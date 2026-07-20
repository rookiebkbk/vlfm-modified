# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from habitat.core.logging import logger
from torch import Tensor

from myon.mapping.object_point_cloud_map import ObjectPointCloudMap
from myon.mapping.obstacle_map import ObstacleMap
from myon.mapping.value_map import ValueMap
from myon.obs_transformers.utils import image_resize
from myon.policy.utils.acyclic_enforcer import AcyclicEnforcer
from myon.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from myon.utils.geometry_utils import closest_point_within_threshold, get_fov, rho_theta
from myon.vlm.blip2itm import BLIP2ITMClient
from myon.vlm.detections import ObjectDetections
from myon.vlm.sam import MobileSAMClient
from myon.vlm.yolov8 import YOLOv8Client
from vlfm.vlm.qwen_vl import QwenVLClient


try:
    from habitat_baselines.common.tensor_dict import TensorDict
    from vlfm.policy.base_policy import BasePolicy
except Exception:
    class BasePolicy:  # type: ignore
        pass

PROMPT_SEPARATOR = "|"

class BaseObjectNavPolicy(BasePolicy):
    pass

class ObjectNavPolicy(BaseObjectNavPolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # 由 _update_object_map() 设置
    _stop_action: Union[Tensor, Any] = None  # 必须由子类设置
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        use_vqa_verification: bool = False,
        vqa_trigger_threshold: float = 0.5,
        vqa_positive_yolo_threshold: float = 0.7,
        vqa_negative_yolo_threshold: float = 0.9,
        vqa_fail_open: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not (
            0.0 <= vqa_trigger_threshold
            <= vqa_positive_yolo_threshold
            <= vqa_negative_yolo_threshold
            <= 1.0
        ):
            raise ValueError(
                "VQA thresholds must satisfy 0 <= trigger <= positive <= negative <= 1"
            )
        self._coco_object_detector = YOLOv8Client(port=int(os.environ.get("YOLOV8_PORT", "12186")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        vqa_port = int(os.environ.get("QWEN_VQA_PORT", "12184"))
        self._vqa_verifier = QwenVLClient(port=vqa_port) if use_vqa_verification else None

        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self._use_vqa_verification = use_vqa_verification
        self._vqa_trigger_threshold = vqa_trigger_threshold
        self._vqa_positive_yolo_threshold = vqa_positive_yolo_threshold
        self._vqa_negative_yolo_threshold = vqa_negative_yolo_threshold
        self._vqa_fail_open = vqa_fail_open
        self._vqa_verifications: List[Dict[str, Any]] = []
        self._vqa_step_info: Dict[str, Any] = {}

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:

        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        self._log_vqa_step()
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        if not self._done_initializing:  # 初始化阶段
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # 尚未找到目标物体
            mode = "explore"
            pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._policy_info["mode"] = mode
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        self._vqa_verifications = []
        self._vqa_step_info = {}
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }
        if self._use_vqa_verification:
            policy_info["vqa_verifications"] = self._vqa_verifications
            policy_info["vqa_step"] = self._vqa_step_info

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # 如果 _object_masks 非零，获取分割轮廓并绘制到 RGB 和深度图上
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _set_vqa_step_info(
        self,
        *,
        triggered: bool,
        result: str,
        accepted_count: int,
        max_yolo_confidence: Union[float, None] = None,
        selected_threshold: Union[float, None] = None,
        raw: str = "",
        error: str = "",
    ) -> None:
        self._vqa_step_info = {
            "step": self._num_steps,
            "target": self._target_object.split("|")[0],
            "enabled": self._use_vqa_verification,
            "triggered": triggered,
            "result": result,
            "verified": True if result == "YES" else False if result == "NO" else None,
            "max_yolo_confidence": max_yolo_confidence,
            "selected_threshold": selected_threshold,
            "accepted_count": accepted_count,
            "target_detected_this_step": accepted_count > 0,
            "raw": raw,
            "error": error,
        }
        self._vqa_verifications = [self._vqa_step_info]

    def _log_vqa_step(self) -> None:
        if not self._use_vqa_verification:
            return

        target_in_object_map = self._object_map.has_object(self._target_object)
        detected_this_step = self._vqa_step_info["target_detected_this_step"]
        if detected_this_step:
            final_decision = "TARGET_ACCEPTED_THIS_STEP"
        elif target_in_object_map:
            final_decision = "TARGET_ALREADY_MAPPED"
        else:
            final_decision = "NO_TARGET"

        self._vqa_step_info.update(
            {
                "target_in_object_map": target_in_object_map,
                "final_target_recognized": detected_this_step or target_in_object_map,
                "final_decision": final_decision,
            }
        )
        log_info = {
            "step": self._vqa_step_info["step"],
            "target": self._vqa_step_info["target"],
            "vqa_triggered": self._vqa_step_info["triggered"],
            "vqa_result": self._vqa_step_info["result"],
            "target_detected_this_step": detected_this_step,
            "final_target_recognized": self._vqa_step_info["final_target_recognized"],
            "final_decision": final_decision,
        }
        for key in (
            "max_yolo_confidence",
            "selected_threshold",
            "accepted_count",
            "raw",
            "error",
        ):
            value = self._vqa_step_info[key]
            if value not in (None, ""):
                log_info[key] = value
        logger.info(f"[VQA_STEP] {json.dumps(log_info, sort_keys=True)}")

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        if self._use_vqa_verification and self._object_map.has_object(self._target_object):
            self._set_vqa_step_info(
                triggered=False,
                result="SKIPPED_ALREADY_MAPPED",
                accepted_count=0,
            )
            return ObjectDetections(
                boxes=torch.empty((0, 4)),
                logits=torch.empty(0),
                phrases=[],
                image_source=img,
                fmt="xyxy",
            )

        target_classes = self._target_object.split("|")

        if self._use_vqa_verification:
            detections = self._coco_object_detector.predict(
                img,
                conf_thres=self._vqa_trigger_threshold,
            )
        else:
            detections = self._coco_object_detector.predict(img)
        detections.filter_by_class(target_classes)

        if not self._use_vqa_verification:
            detections.filter_by_conf(self._coco_threshold)
            self._vqa_step_info = {}
            self._vqa_verifications = []
            return detections

        if detections.num_detections == 0:
            self._set_vqa_step_info(
                triggered=False,
                result="NOT_TRIGGERED",
                accepted_count=0,
            )
            return detections

        max_confidence = float(detections.logits.max().item())
        if max_confidence < self._vqa_trigger_threshold:
            detections.filter_by_conf(self._vqa_trigger_threshold)
            self._set_vqa_step_info(
                triggered=False,
                result="NOT_TRIGGERED",
                accepted_count=0,
                max_yolo_confidence=max_confidence,
            )
            return detections

        verified: Union[bool, None] = None
        use_positive_threshold = self._vqa_fail_open
        raw_response = ""
        error = ""
        try:
            if self._vqa_verifier is None:
                raise RuntimeError("VQA verifier is not configured")
            result = self._vqa_verifier.verify_detection(img, target_classes[0])
            verified = bool(result["verified"])
            use_positive_threshold = verified
            raw_response = str(result.get("raw", ""))
        except Exception as exc:
            error = str(exc)

        selected_threshold = (
            self._vqa_positive_yolo_threshold
            if use_positive_threshold
            else self._vqa_negative_yolo_threshold
        )
        detections.filter_by_conf(selected_threshold)
        result_name = "ERROR" if verified is None else ("YES" if verified else "NO")
        self._set_vqa_step_info(
            triggered=True,
            result=result_name,
            accepted_count=detections.num_detections,
            max_yolo_confidence=max_confidence,
            selected_threshold=selected_threshold,
            raw=raw_response,
            error=error,
        )

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)

        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """从观测中提取 RGB、深度和相机变换矩阵。子类必须实现。"""
        raise NotImplementedError


class ITMPolicy(ObjectNavPolicy):

    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}
    
    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine( #BLIP2ITMClient 实例，计算文本和图像的余弦相似度
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

# 设置默认值，被yaml配置文件覆盖
@dataclass
class MyonConfig:
    """myon 策略基础配置，字段对应 ObjectNavPolicy.__init__ 和 ITMPolicy.__init__ 的参数。"""
    name: str = "HabitatSimplePolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    obstacle_map_area_threshold: float = 1.5
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18
    min_commit_steps: int = 4
    switch_margin: float = 0.05
    stuck_patience: int = 20
    frontier_reached_radius: float = 0.45
    frontier_progress_epsilon: float = 0.1
    robot_motion_epsilon: float = 0.03
    blocked_frontier_radius: float = 0.6
    blocked_frontier_cooldown: int = 30
    use_vqa_verification: bool = False
    vqa_trigger_threshold: float = 0.5
    vqa_positive_yolo_threshold: float = 0.7
    vqa_negative_yolo_threshold: float = 0.9
    vqa_fail_open: bool = False

    @classmethod
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(MyonConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="myon_config_base", node=MyonConfig())
