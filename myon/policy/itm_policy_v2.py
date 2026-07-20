# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple, Union
import os

import numpy as np
import torch
from torch import Tensor

from .simple_policy import ITMPolicy, ObjectDetections
from myon.utils.geometry_utils import closest_point_within_threshold
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

class ITMPolicyV2(ITMPolicy):
    """
    ITMPolicy 的变体，覆盖 _get_best_frontier 以实验新的前沿选择策略。

    继承关系:
        ITMPolicyV2 → ITMPolicy → ObjectNavPolicy

    与父类 ITMPolicy 的区别仅在于 _get_best_frontier 的实现。
    act(), _update_value_map(), _sort_frontiers_by_value() 等全部继承自 ITMPolicy。
    """

    def __init__(
        self,
        min_commit_steps: int = 4,
        switch_margin: float = 0.05,
        stuck_patience: int = 20,
        frontier_reached_radius: float = 0.45,
        frontier_progress_epsilon: float = 0.1,
        robot_motion_epsilon: float = 0.03,
        blocked_frontier_radius: float = 0.6,
        blocked_frontier_cooldown: int = 30,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        print("Initializing ITMPolicyV2")

        self._min_commit_steps = min_commit_steps
        self._switch_margin = switch_margin
        self._stuck_patience = stuck_patience
        self._frontier_reached_radius = frontier_reached_radius
        self._frontier_progress_epsilon = frontier_progress_epsilon
        self._robot_motion_epsilon = robot_motion_epsilon
        self._blocked_frontier_radius = blocked_frontier_radius
        self._blocked_frontier_cooldown = blocked_frontier_cooldown
        self._blocked_frontiers: List[Tuple[np.ndarray, int]] = []

        self._reset_frontier_commitment_state()


    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:

        os.environ["DEBUG_INFO"] = ""
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # 检查上次前沿的 stuck 状态
        last_idx = -1
        last_is_stuck = False
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            last_idx = closest_point_within_threshold(
                sorted_pts, self._last_frontier, threshold=0.5
            )
            if last_idx != -1:
                last_frontier = sorted_pts[last_idx]
                self._update_frontier_progress(last_frontier)
                last_is_stuck = self._is_frontier_stuck(last_frontier)

                if last_is_stuck:
                    self._block_frontier(last_frontier)
                    print(f"Last frontier is stuck. Blocking it {self._blocked_frontier_cooldown} steps and choosing a new one.")

        # 选非循环且未被 加入黑名单 的最优前沿
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                if self._is_frontier_blocked(frontier):
                    continue
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        # 全被 skip 或全循环 → 选最近的非 blocked 前沿
        if best_frontier_idx is None:
            print("All frontiers are cyclic or blocked. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic or blocked. "

            candidate_indices = [
                i
                for i in range(len(sorted_pts))
                if not self._is_frontier_blocked(sorted_pts[i])
            ]

            # 所有候选都被排除 → 退回全体候选
            if len(candidate_indices) == 0:
                candidate_indices = list(range(len(sorted_pts)))

            best_frontier_idx = min(
                candidate_indices,
                key=lambda i: np.linalg.norm(sorted_pts[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]

        # 防抖坚持：仅当上次前沿未 stuck 且仍在列表中
        if last_idx != -1 and not last_is_stuck and not self._is_frontier_blocked(sorted_pts[last_idx]):
            curr_value = sorted_values[last_idx]
            value_gap = best_value - curr_value


            should_stick = (
                self._frontier_commit_steps < self._min_commit_steps
                or value_gap < self._switch_margin
            )
            if should_stick:
                print("Sticking to last point.")
                os.environ["DEBUG_INFO"] += "Sticking to last point. "
                best_frontier_idx = last_idx
                best_frontier = sorted_pts[best_frontier_idx]
                best_value = sorted_values[best_frontier_idx]
                self._frontier_commit_steps += 1
            else:
                self._reset_frontier_commitment_state()
        else:
            self._reset_frontier_commitment_state()

        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value
    
    def _is_frontier_stuck(self, frontier: np.ndarray) -> bool:

        """
        检查机器人是否在追踪当前前沿时卡住了。
        卡住的条件：
        1. 机器人距离前沿小于 frontier_reached_radius → 已到达
        2. 连续多步未接近当前追踪的前沿 → 卡
        3. 机器人运动不足 → 卡"""

        robot_xy = self._observations_cache["robot_xy"]
        dist = np.linalg.norm(frontier - robot_xy)

        if dist < self._frontier_reached_radius:
            return True

        # 连续多步未接近当前追踪的frontier或机器人运动不足 → 认为 stuck
        no_progress = self._frontier_no_progress_steps >= self._stuck_patience
        low_motion = self._frontier_low_motion_steps >= self._stuck_patience * 2

        return no_progress or low_motion

    def _reset_frontier_commitment_state(self) -> None:
        self._frontier_commit_steps = 0
        self._last_frontier_distance = None
        self._frontier_no_progress_steps = 0
        self._frontier_low_motion_steps = 0
        self._last_robot_xy_for_frontier = None

    def _block_frontier(self, frontier: np.ndarray) -> None:
        """将前沿加入黑名单，cooldown 步内不可再选。"""
        if self._is_frontier_blocked(frontier):
            return

        unblock_step = self._num_steps + self._blocked_frontier_cooldown
        self._blocked_frontiers.append((frontier.copy(), unblock_step))


    def _is_frontier_blocked(self, frontier: np.ndarray) -> bool:
        """检查前沿是否在黑名单中。"""
        self._blocked_frontiers = [
            (blocked_frontier, unblock_step)
            for blocked_frontier, unblock_step in self._blocked_frontiers
            if self._num_steps < unblock_step
        ]

        return any(
            np.linalg.norm(frontier - blocked_frontier)
            < self._blocked_frontier_radius
            for blocked_frontier, _ in self._blocked_frontiers
        )

    def _update_frontier_progress(self, frontier: np.ndarray) -> None:
        robot_xy = self._observations_cache["robot_xy"]

        dist = np.linalg.norm(frontier - robot_xy)

        if self._last_frontier_distance is not None:
            progress = self._last_frontier_distance - dist
            if progress < self._frontier_progress_epsilon:
                self._frontier_no_progress_steps += 1
            else:
                self._frontier_no_progress_steps = 0

        if self._last_robot_xy_for_frontier is not None:
            motion = np.linalg.norm(robot_xy - self._last_robot_xy_for_frontier)
            if motion < self._robot_motion_epsilon:
                self._frontier_low_motion_steps += 1
            else:
                self._frontier_low_motion_steps = 0

        self._last_frontier_distance = dist
        self._last_robot_xy_for_frontier = robot_xy.copy()

    def _reset(self) -> None:
        super()._reset()
        self._reset_frontier_commitment_state()
        self._blocked_frontiers = []