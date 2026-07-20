# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from myon.policy.qwen_policy import fuse_qwen_scores
from vlfm.vlm.qwen_vl import QwenVLModel, build_observation_scoring_prompt


def test_scores_are_fused_with_equal_weight() -> None:
    assert fuse_qwen_scores(1.0, 0.0) == 0.5
    assert fuse_qwen_scores(0.0, 1.0) == 0.5
    assert fuse_qwen_scores(0.75, 0.25) == 0.5


def test_scoring_prompt_forbids_stairs() -> None:
    prompt = build_observation_scoring_prompt("chair")
    assert "must not go up or down stairs" in prompt
    assert "If stairs are required, score both 1" in prompt
    assert "EXPLORE=<1-5> TARGET=<1-5>" in prompt


def test_one_to_five_scores_are_normalized() -> None:
    model = object.__new__(QwenVLModel)
    model.generate = lambda *args, **kwargs: "EXPLORE=5 TARGET=1"

    result = model.score_observation(np.zeros((8, 8, 3), dtype=np.uint8), "chair")

    assert result["target"] == 0.0
    assert result["exploration"] == 1.0
