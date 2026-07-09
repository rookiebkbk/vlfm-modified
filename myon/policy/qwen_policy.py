# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
QwenPolicyV3: uses Qwen3-VL for target relevance scoring and geometric heuristics
for explorability scoring, combining both in a 2-channel ValueMap.

Inherits from ITMPolicyV2 to get its stuck detection, blocked frontier, and
commitment-based frontier selection logic.
"""

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch import Tensor

from myon.policy.itm_policy_v2 import ITMPolicyV2
from vlfm.utils.geometry_utils import extract_yaw
from vlfm.utils.img_utils import place_img_in_img, rotate_image
from vlfm.vlm.qwen_vl import QwenVLClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class QwenPolicyV3(ITMPolicyV2):
    """V3 policy using Qwen3-VL for target relevance + geometric explorability.

    Extends ITMPolicyV2 with a 2-channel ValueMap:
      - Channel 0: target relevance (QwenVL VLM scoring)
      - Channel 1: explorability (geometric heuristic from obstacle map)

    The _reduce_values method gates between the two: if any frontier has high
    target relevance (>= exploration_thresh), use target values; otherwise,
    use explorability values to efficiently cover new area.
    """

    def __init__(
        self,
        text_prompt: str = "Seems like there is a target_object ahead.",
        exploration_thresh: float = 0.35,
        qwen_port: int = 12182,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            text_prompt: Prompt template for target relevance scoring.
                "target_object" is replaced with the actual target name.
            exploration_thresh: Threshold for switching between target-seeking
                and exploration modes. If max target relevance across frontiers
                is below this, use explorability scores instead.
            qwen_port: Port of the Qwen3-VL vLLM server.
        """
        super().__init__(text_prompt=text_prompt, *args, **kwargs)

        # Replace BLIP2ITMClient with QwenVLClient
        self._qwenvl = QwenVLClient(port=qwen_port)

        # Override ValueMap: 2 channels for [target_relevance, explorability]
        self._value_map._value_map = np.zeros(
            (self._value_map._value_map.shape[0],
             self._value_map._value_map.shape[1],
             2),
            dtype=np.float32,
        )
        self._value_map._value_channels = 2

        self._exploration_thresh = exploration_thresh

        # Custom visualization: show ch0 if above thresh, else show max across channels
        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            first_channel = arr[:, :, 0]
            max_values = np.max(arr, axis=2)
            mask = first_channel > exploration_thresh
            result = np.where(mask, first_channel, max_values)
            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _update_value_map(self) -> None:
        """Update value map with QwenVL target relevance + geometric explorability.

        For each cached RGB observation:
          - Channel 0: QwenVL.cosine(rgb, prompt) for target relevance
          - Channel 1: geometric ratio of unknown area in FOV (explorability)
        """
        if not hasattr(self, "_observations_cache"):
            return
        vmrgbd = self._observations_cache.get("value_map_rgbd", [])
        if not vmrgbd:
            return

        all_rgb = [i[0] for i in vmrgbd]

        # Compute target relevance scores via QwenVL
        prompts = [
            p.replace("target_object", self._target_object.replace("|", "/"))
            for p in self._text_prompt.split(PROMPT_SEPARATOR)
        ]
        # Use the first prompt for target relevance (channel 0)
        primary_prompt = prompts[0] if prompts else "Describe this scene."

        cosines = []
        for rgb in all_rgb:
            try:
                target_score = self._qwenvl.cosine(rgb, primary_prompt)
            except Exception as e:
                print(f"[QwenPolicyV3] QwenVL cosine failed: {e}. Using default 0.0.")
                target_score = 0.0

            # Compute explorability geometrically
            exploration_score = self._compute_explorability()

            cosines.append([target_score, exploration_score])

        for cosine_pair, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, vmrgbd
        ):
            self._value_map.update_map(
                np.array(cosine_pair), depth, tf, min_depth, max_depth, fov
            )

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """Reduce 2-channel values to a single value per frontier.

        If any frontier has target relevance above the threshold, use target
        values. Otherwise, use explorability values to prioritize new area.
        """
        target_values = [v[0] for v in values]
        max_target = max(target_values)

        if max_target < self._exploration_thresh:
            return [v[1] for v in values]  # explorability
        else:
            return target_values  # target relevance

    def _sort_frontiers_by_value(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        """Sort frontiers by 2-channel ValueMap values with reduce_fn."""
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(
            frontiers, 0.5, reduce_fn=self._reduce_values
        )
        return sorted_frontiers, sorted_values

    def _compute_explorability(self) -> float:
        """Compute how much new/unexplored area the current view reveals.

        Uses the obstacle map's explored_area to compute the ratio of newly
        explored (previously unknown) pixels covered by the current FOV.

        Returns:
            Float in [0, 1] where 1.0 means entirely new area and 0.0 means
            already fully explored area.
        """
        if not hasattr(self, "_obstacle_map") or self._obstacle_map is None:
            return 0.0

        try:
            vmrgbd = self._observations_cache.get("value_map_rgbd", [])
            if not vmrgbd:
                return 0.0

            # Use the latest RGB-D observation
            _, depth, tf_camera, min_depth, max_depth, fov = vmrgbd[-1]

            # Get explored area from obstacle map
            # 0 = unknown, 1 = explored, 2 = obstacle (obstacle_map values)
            explored = self._obstacle_map.explored_area.astype(np.uint8)

            # Create FOV cone mask in obstacle map pixel space
            fov_mask = self._get_fov_mask_in_map(
                depth, tf_camera, min_depth, max_depth, fov
            )

            if fov_mask is None or np.sum(fov_mask) == 0:
                return 0.0

            # Count unknown (not explored) pixels within the FOV
            unknown_in_fov = np.sum((explored == 0) & (fov_mask > 0))
            total_in_fov = np.sum(fov_mask > 0)

            if total_in_fov == 0:
                return 0.0

            return min(1.0, unknown_in_fov / total_in_fov)

        except Exception as e:
            print(f"[QwenPolicyV3] explorability computation failed: {e}")
            return 0.0

    def _get_fov_mask_in_map(
        self,
        depth: np.ndarray,
        tf_camera: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        """Project the current FOV into obstacle map pixel coordinates.

        Follows the same pattern as ValueMap._localize_new_data / _process_local_data.

        Returns:
            Boolean mask in obstacle map pixel space, True where FOV covers.
        """
        # Use ValueMap's internal method for FOV projection
        cone_mask = self._value_map._process_local_data(
            depth, fov, min_depth, max_depth
        )

        # Rotate to match camera orientation
        yaw = extract_yaw(tf_camera)
        cone_mask = rotate_image(cone_mask, -yaw)

        # Place at correct world position
        cam_x = tf_camera[0, 3]
        cam_y = tf_camera[1, 3]
        px = int(cam_x * self._value_map.pixels_per_meter) + self._value_map._episode_pixel_origin[0]
        py = int(-cam_y * self._value_map.pixels_per_meter) + self._value_map._episode_pixel_origin[1]

        # Place the cone mask into a full-size map
        full_mask = np.zeros(
            (self._value_map._value_map.shape[0], self._value_map._value_map.shape[1]),
            dtype=np.float32,
        )
        fov_mask = place_img_in_img(full_mask, cone_mask, px, py)

        return fov_mask
