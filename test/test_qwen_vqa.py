# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import torch

from myon.policy.simple_policy import ObjectNavPolicy
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.qwen_vl import (
    build_verification_prompt,
    crop_detection,
    parse_verification_response,
)
from vlfm.vlm.server_wrapper import image_to_str, str_to_image


class FakeObjectMap:
    def __init__(self, has_object: bool = False) -> None:
        self._has_object = has_object

    def has_object(self, target_object: str) -> bool:
        return self._has_object


class FakeDetector:
    def __init__(self, detections: ObjectDetections) -> None:
        self.detections = detections
        self.calls = 0

    def predict(self, image: np.ndarray, **kwargs: Any) -> ObjectDetections:
        self.calls += 1
        return self.detections


class FakeVerifier:
    def __init__(self, results: List[Any]) -> None:
        self.results = iter(results)
        self.images: List[np.ndarray] = []

    def verify_detection(self, image: np.ndarray, target_object: str) -> Dict[str, Any]:
        self.images.append(image)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def make_policy(
    detections: ObjectDetections,
    verifier: FakeVerifier,
    fail_open: bool = False,
    use_vqa: bool = True,
) -> ObjectNavPolicy:
    policy = object.__new__(ObjectNavPolicy)
    policy._target_object = "chair"
    policy._object_map = FakeObjectMap()
    policy._coco_object_detector = FakeDetector(detections)
    policy._coco_threshold = 0.8
    policy._use_vqa_verification = use_vqa
    policy._vqa_verifier = verifier
    policy._vqa_trigger_threshold = 0.5
    policy._vqa_positive_yolo_threshold = 0.7
    policy._vqa_negative_yolo_threshold = 0.9
    policy._vqa_fail_open = fail_open
    policy._vqa_verifications = []
    policy._vqa_step_info = {}
    policy._num_steps = 0
    return policy


def test_parse_verification_response() -> None:
    assert parse_verification_response("YES")
    assert parse_verification_response("Answer: yes.")
    assert not parse_verification_response("NO")
    assert not parse_verification_response("uncertain")


def test_verification_prompt_allows_distant_and_occluded_targets() -> None:
    prompt = build_verification_prompt("chair")
    assert "distant or partially occluded" in prompt
    assert "more likely to be a \"chair\" than another object" in prompt
    assert "too ambiguous to identify" not in prompt
    assert "directly visible" in prompt
    assert "only likely to be nearby" in prompt
    assert "appears only in a photo or on a screen" in prompt


def test_crop_detection_supports_normalized_boxes() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_detection(image, [0.25, 0.2, 0.75, 0.8], padding=0.0)
    assert crop.shape == (60, 100, 3)


def test_image_transport_accepts_default_jpeg_quality() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    decoded = str_to_image(image_to_str(image))
    assert decoded.shape == image.shape


def test_vqa_yes_uses_positive_threshold_on_full_image() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.7],
                [0.5, 0.2, 0.9, 0.8],
                [0.2, 0.2, 0.3, 0.3],
            ]
        ),
        logits=torch.tensor([0.95, 0.75, 0.65]),
        phrases=["chair", "chair", "chair"],
        image_source=image,
        fmt="xyxy",
    )
    verifier = FakeVerifier([{"verified": True, "raw": "YES"}])
    policy = make_policy(detections, verifier)

    filtered = policy._get_object_detections(image)

    assert filtered.num_detections == 2
    assert torch.allclose(filtered.logits, torch.tensor([0.95, 0.75]))
    assert len(verifier.images) == 1
    assert verifier.images[0] is image
    assert policy._vqa_verifications[0]["selected_threshold"] == 0.7
    assert policy._vqa_verifications[0]["accepted_count"] == 2


def test_vqa_no_uses_negative_threshold() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor([[0.1, 0.1, 0.4, 0.7], [0.5, 0.2, 0.9, 0.8]]),
        logits=torch.tensor([0.95, 0.85]),
        phrases=["chair", "chair"],
        image_source=image,
        fmt="xyxy",
    )
    verifier = FakeVerifier([{"verified": False, "raw": "NO"}])
    policy = make_policy(detections, verifier)

    filtered = policy._get_object_detections(image)

    assert filtered.num_detections == 1
    assert torch.allclose(filtered.logits, torch.tensor([0.95]))
    assert len(verifier.images) == 1
    assert policy._vqa_verifications[0]["selected_threshold"] == 0.9


def test_vqa_is_not_called_below_trigger_threshold() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor([[0.1, 0.1, 0.4, 0.7]]),
        logits=torch.tensor([0.49]),
        phrases=["chair"],
        image_source=image,
        fmt="xyxy",
    )
    verifier = FakeVerifier([])
    policy = make_policy(detections, verifier)

    filtered = policy._get_object_detections(image)

    assert filtered.num_detections == 0
    assert verifier.images == []
    assert not policy._vqa_verifications[0]["triggered"]


def test_already_mapped_target_has_clear_final_decision() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.empty((0, 4)),
        logits=torch.empty(0),
        phrases=[],
        image_source=image,
        fmt="xyxy",
    )
    policy = make_policy(detections, FakeVerifier([]))
    policy._object_map = FakeObjectMap(has_object=True)

    filtered = policy._get_object_detections(image)
    policy._log_vqa_step()

    assert filtered.num_detections == 0
    assert policy._coco_object_detector.calls == 0
    assert not policy._vqa_step_info["triggered"]
    assert not policy._vqa_step_info["target_detected_this_step"]
    assert policy._vqa_step_info["target_in_object_map"]
    assert policy._vqa_step_info["final_target_recognized"]
    assert policy._vqa_step_info["final_decision"] == "TARGET_ALREADY_MAPPED"


def test_vqa_error_respects_fail_open() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor([[0.1, 0.1, 0.4, 0.7]]),
        logits=torch.tensor([0.95]),
        phrases=["chair"],
        image_source=image,
        fmt="xyxy",
    )
    policy = make_policy(
        detections,
        FakeVerifier([RuntimeError("service unavailable")]),
        fail_open=True,
    )

    filtered = policy._get_object_detections(image)

    assert filtered.num_detections == 1
    assert policy._vqa_verifications[0]["error"] == "service unavailable"
    assert policy._vqa_verifications[0]["verified"] is None
    assert policy._vqa_verifications[0]["selected_threshold"] == 0.7


def test_vqa_error_rejects_by_default() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor([[0.1, 0.1, 0.4, 0.7], [0.5, 0.2, 0.9, 0.8]]),
        logits=torch.tensor([0.95, 0.85]),
        phrases=["chair", "chair"],
        image_source=image,
        fmt="xyxy",
    )
    policy = make_policy(
        detections,
        FakeVerifier([RuntimeError("service unavailable")]),
    )

    filtered = policy._get_object_detections(image)

    assert filtered.num_detections == 1
    assert torch.allclose(filtered.logits, torch.tensor([0.95]))
    assert policy._vqa_verifications[0]["verified"] is None
    assert policy._vqa_verifications[0]["selected_threshold"] == 0.9


def test_disabling_vqa_preserves_coco_threshold_filtering() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = ObjectDetections(
        boxes=torch.tensor([[0.1, 0.1, 0.4, 0.7], [0.5, 0.2, 0.9, 0.8]]),
        logits=torch.tensor([0.81, 0.79]),
        phrases=["chair", "chair"],
        image_source=image,
        fmt="xyxy",
    )
    verifier = FakeVerifier([])
    policy = make_policy(detections, verifier, use_vqa=False)

    with patch("myon.policy.simple_policy.logger.info") as log_info:
        filtered = policy._get_object_detections(image)
        policy._log_vqa_step()

    assert filtered.num_detections == 1
    assert torch.allclose(filtered.logits, torch.tensor([0.81]))
    assert verifier.images == []
    assert policy._vqa_step_info == {}
    assert policy._vqa_verifications == []
    log_info.assert_not_called()
