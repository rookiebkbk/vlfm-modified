# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
QwenPolicyV3 — 使用 Qwen3-VL 大模型从目标相关性和可探索性两个维度评分观测，
融合为单一探索价值存入 ValueMap，驱动前沿选择。

与 ITMPolicyV2 的核心区别:
  - ITMPolicyV2: BLIP2ITM 计算图文余弦相似度 → 存入 ValueMap
  - QwenPolicyV3: Qwen3-VL 同时输出 target_relevance + explorability，
    融合后存入单通道 ValueMap

融合策略: value = (target_relevance + explorability) / 2
  - 两个归一化分数等权融合
"""

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch import Tensor

from myon.policy.itm_policy_v2 import ITMPolicyV2
from myon.mapping.value_map import ValueMap
from vlfm.vlm.qwen_vl import QwenVLClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass


def fuse_qwen_scores(target: float, exploration: float) -> float:
    """Fuse normalized target and exploration scores with equal weight."""
    return (target + exploration) / 2.0


class QwenPolicyV3(ITMPolicyV2):
    """V3 policy: Qwen3-VL 双维度评分 → 融合 → 单通道 ValueMap。

    继承自 ITMPolicyV2，获得:
      - stuck 检测 / blocked frontier 黑名单
      - 防抖坚持 (commitment-based frontier selection)
      - 环检测 (acyclic enforcement)

    新增:
      - QwenVLClient 替代 BLIP2ITMClient
      - score_observation() 同时获取 target + exploration 评分
      - 等权融合存入单通道 ValueMap
    """

    def __init__(
        self,
        text_prompt: str = "",
        qwen_port: int = 12182,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            text_prompt: 不再用于 ITM cosine，保留参数兼容性。
            qwen_port: Qwen3-VL vLLM 服务端口。
        """
        # 跳过 ITMPolicy.__init__ 中的 BLIP2ITMClient 创建，
        # 但我们仍需要 ValueMap, AcyclicEnforcer 等。
        # ITMPolicyV2 → ITMPolicy → ObjectNavPolicy 的继承链较深，
        # 这里手动覆盖父类中 BLIP2ITM 相关的初始化。
        super().__init__(text_prompt=text_prompt, *args, **kwargs)

        # 用 QwenVLClient 替换父类的 BLIP2ITMClient
        qwen_port = int(os.environ.get("QWEN_PORT", str(qwen_port)))
        self._qwenvl = QwenVLClient(port=qwen_port)

        # 重建 ValueMap 为单通道（父类可能根据 prompt 分隔符创建了多通道）
        self._value_map = ValueMap(
            value_channels=1,
            use_max_confidence=kwargs.get("use_max_confidence", False),
            obstacle_map=self._obstacle_map if kwargs.get("sync_explored_areas", False) else None,
        )

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()

    def _update_value_map(self) -> None:
        """用 Qwen3-VL 对每个观测评分，融合为单一探索价值后更新 ValueMap。

        对 value_map_rgbd 中的每个 RGB 图像:
          1. 调用 Qwen3-VL score_observation(rgb, target_object)
          2. 得到 target_relevance ∈ [0,1], explorability ∈ [0,1]
          3. 等权融合两个分数
          4. 存入单通道 ValueMap
        """
        if not hasattr(self, "_observations_cache"):
            return
        vmrgbd = self._observations_cache.get("value_map_rgbd", [])
        if not vmrgbd:
            return

        all_rgb = [item[0] for item in vmrgbd]

        fused_values = []
        for rgb in all_rgb:
            try:
                scores = self._qwenvl.score_observation(rgb, self._target_object)
                target = scores["target"]
                exploration = scores["exploration"]
                fused = fuse_qwen_scores(target, exploration)
                fused_values.append(fused)
                print(
                    f"[QwenPolicyV3] target={target:.2f} explore={exploration:.2f} "
                    f"fused={fused:.2f} raw={scores['raw'][:80]}"
                )
            except Exception as e:
                print(f"[QwenPolicyV3] QwenVL score_observation failed: {e}. Using 0.0.")
                fused_values.append(0.0)

        for fused_value, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            fused_values, vmrgbd
        ):
            self._value_map.update_map(
                np.array([fused_value], dtype=np.float32),
                depth,
                tf,
                min_depth,
                max_depth,
                fov,
            )

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        """ValueMap 是单通道的，直接排序。"""
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(
            frontiers, 0.5
        )
        return sorted_frontiers, sorted_values
