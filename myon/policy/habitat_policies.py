# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import PolicyConfig
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

from myon.mapping.obstacle_map import ObstacleMap
from myon.utils.geometry_utils import xyz_yaw_to_tf_matrix
from myon.vlm.detections import ObjectDetections

from .itm_policy_v2 import ITMPolicyV2
from .qwen_policy import QwenPolicyV3
from .simple_policy import ITMPolicy, MyonConfig, ObjectNavPolicy


HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
# MP3D_ID_TO_NAME = [
#     "chair",
#     "table|dining table|coffee table|side table|desk",  # "table",
#     "framed photograph",  # "picture",
#     "cabinet",
#     "pillow",  # "cushion",
#     "couch",  # "sofa",
#     "bed",
#     "nightstand",  # "chest of drawers",
#     "potted plant",  # "plant",
#     "sink",
#     "toilet",
#     "stool",
#     "towel",
#     "tv",  # "tv monitor",
#     "shower",
#     "bathtub",
#     "counter",
#     "fireplace",
#     "gym equipment",
#     "seating",
#     "clothes",
# ]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    """Habitat 环境适配层，负责将 ObjectNavPolicy 接入 Habitat 仿真器。
    提供观测缓存、坐标变换、360° 初始化等功能。
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # 由 _reset() 设置
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        policy_config: SimplePolicyConfig = config.habitat_baselines.rl.policy
        kwargs = {k: policy_config[k] for k in SimplePolicyConfig.kwaarg_names}  # type: ignore

        # 从 habitat 配置中提取相机参数
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

        # 同步深度参数
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width

        # 仅在需要保存视频时启用可视化
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", ObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """将数字类别 ID 转为字符串名称，返回 PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        # elif self._dataset_type == "mp3d":
        #     obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
        #     self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        parent_cls: ObjectNavPolicy = super()  # type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """向左转 30 度，共 12 次完成 360° 初始化"""
        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: ObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """获取策略信息用于日志记录"""
        parent_cls: ObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", ObjectNavPolicy], observations: TensorDict) -> None:
        """缓存 RGB、深度图和相机变换矩阵。

        Args:
           observations (TensorDict): 当前时间步的观测。
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS 的西方向为负，因此翻转 y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class HabitatSimplePolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatSimplePolicyV2(HabitatMixin, ITMPolicyV2):
    pass


@baseline_registry.register_policy
class HabitatQwenPolicyV3(HabitatMixin, QwenPolicyV3):
    pass


@dataclass
class SimplePolicyConfig(MyonConfig, PolicyConfig):
    name: str = "HabitatSimplePolicy"


cs = ConfigStore.instance()
cs.store(
    group="habitat_baselines/rl/policy",
    name="simple_policy",
    node=SimplePolicyConfig,
)
