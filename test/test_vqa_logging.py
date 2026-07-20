# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Dict

from myon.final.utils.final_trainer import (
    extract_scalars_from_info as extract_final_scalars,
)
from myon.utils.vlfm_trainer import (
    extract_scalars_from_info as extract_vlfm_scalars,
)


def test_vqa_diagnostics_are_not_treated_as_episode_metrics() -> None:
    info: Dict[str, Any] = {
        "success": 1,
        "distance_to_goal": 0.25,
        "vqa_step": {
            "triggered": False,
            "result": "NOT_TRIGGERED",
            "verified": None,
            "max_yolo_confidence": None,
            "selected_threshold": None,
            "target_detected_this_step": False,
        },
        "vqa_verifications": [],
    }

    extractors: tuple[Callable[[Dict[str, Any]], Dict[str, float]], ...] = (
        extract_final_scalars,
        extract_vlfm_scalars,
    )
    for extract in extractors:
        assert extract(info) == {"success": 1.0, "distance_to_goal": 0.25}
