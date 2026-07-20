# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
Qwen vision-language server and client for visual question answering.

Uses transformers directly (no vLLM dependency) and supports compatible Qwen
vision-language checkpoints such as Qwen3.5-2B and Qwen3-VL-8B.

Usage:
    # Start the server:
    python -m vlfm.vlm.qwen_vl --port 12182

    # Use the client:
    client = QwenVLClient(port=12182)
    answer = client.ask(image, "Is this a chair?")
    scores = client.score_observation(image, "chair")
"""

import re
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import requests
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, image_to_str, str_to_image


def crop_detection(
    image: np.ndarray,
    box: Sequence[float],
    padding: float = 0.2,
) -> np.ndarray:
    """Crop a normalized or absolute xyxy detection with surrounding context."""
    if padding < 0:
        raise ValueError("padding must be non-negative")

    height, width = image.shape[:2]
    xyxy = np.asarray(box, dtype=np.float32).reshape(4).copy()
    if np.max(np.abs(xyxy)) <= 1.0 + 1e-6:
        xyxy *= np.array([width, height, width, height], dtype=np.float32)

    x1, y1, x2, y2 = xyxy
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    x1 = max(0, int(np.floor(x1 - box_width * padding)))
    y1 = max(0, int(np.floor(y1 - box_height * padding)))
    x2 = min(width, int(np.ceil(x2 + box_width * padding)))
    y2 = min(height, int(np.ceil(y2 + box_height * padding)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid detection box: {list(box)}")
    return image[y1:y2, x1:x2].copy()


def parse_verification_response(response: str) -> bool:
    """Parse the first explicit YES/NO token, rejecting malformed answers."""
    match = re.search(r"\b(YES|NO)\b", response, re.IGNORECASE)
    return match is not None and match.group(1).upper() == "YES"


def build_verification_prompt(target_object: str) -> str:
    """Build a concise full-image target verification prompt."""
    return (
        f'Is at least one real "{target_object}" directly visible in this image? '
        f'A distant or partially occluded object counts as YES if it is more '
        f'likely to be a "{target_object}" than another object. '
        f'Answer NO if it is only likely to be nearby, appears only in a photo or '
        f'on a screen, or the candidate is more likely a different object. '
        f'Answer exactly YES or NO.'
    )


def build_observation_scoring_prompt(target_object: str) -> str:
    """Build the target/exploration scoring prompt used by Qwen policies."""
    return (
        f'Score this direction from 1 to 5.\n'
        f'EXPLORE: Does it lead to a new area, such as through a door or corridor?\n'
        f'TARGET: How relevant is it to the current target "{target_object}"?\n'
        f'The robot must not go up or down stairs. If stairs are required, score both 1.\n'
        f'Answer exactly: EXPLORE=<1-5> TARGET=<1-5>'
    )


# Server


class QwenVLModel:
    """Load a compatible Qwen vision-language checkpoint for inference."""

    def __init__(self, model_path: str = "/root/objnav/vlfm/qwen/qwen3.5-2B", device: Any = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"[QwenVL] Loading model from {model_path} on {device}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("[QwenVL] Model loaded!")

    def generate(self, image: np.ndarray, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a text response given an image and prompt."""
        pil_img = Image.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kwargs = {}
        if "enable_thinking" in (self.processor.chat_template or ""):
            template_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.processor(text=text, images=[pil_img], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """VQA: ask a question about an image."""
        if prompt is None or prompt == "":
            prompt = "Describe this image in detail."
        return self.generate(image, prompt, max_new_tokens=128)

    def verify_detection(self, image: np.ndarray, target_object: str) -> Dict[str, Any]:
        """Verify that the target is visible in the full observation."""
        prompt = build_verification_prompt(target_object)
        response = self.generate(image, prompt, max_new_tokens=4)
        verified = parse_verification_response(response)
        print(
            f"[QwenVL] verify_detection({target_object}): "
            f"verified={verified}, raw={response!r}"
        )
        return {"verified": verified, "raw": response}

    def score_observation(self, image: np.ndarray, target_object: str) -> dict:
        """Score an observation on target relevance and explorability.

        Returns dict with keys: target, exploration, raw
        """
        prompt = build_observation_scoring_prompt(target_object)
        response = self.generate(image, prompt, max_new_tokens=32)

        result = {"target": 0.0, "exploration": 0.0, "raw": response}

        # Parse the 1-5 ratings and normalize them to [0, 1].
        target_match = re.search(r"TARGET\s*=\s*(\d+)", response, re.IGNORECASE)
        explore_match = re.search(r"EXPLORE\s*=\s*(\d+)", response, re.IGNORECASE)

        if target_match:
            target_score = max(1, min(5, int(target_match.group(1))))
            result["target"] = (target_score - 1) / 4.0
        if explore_match:
            exploration_score = max(1, min(5, int(explore_match.group(1))))
            result["exploration"] = (exploration_score - 1) / 4.0

        print(f"[QwenVL] score_observation({target_object}): {result}")
        return result


# Client


class QwenVLClient:
    """HTTP client that talks to the QwenVL Flask server."""

    def __init__(
        self,
        port: int = 12182,
        base_url: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 1,
    ) -> None:
        if base_url is None:
            base_url = f"http://localhost:{port}"
        self.url = f"{base_url}/qwen_vl"
        self.timeout = timeout
        self.max_retries = max_retries
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

    def _request(self, image: np.ndarray, method: str, **kwargs: Any) -> Dict[str, Any]:
        payload = {
            "image": image_to_str(image),
            "method": method,
            **kwargs,
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2)
        raise RuntimeError(f"QwenVL request failed: {last_error}")

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = ""
        response = self._request(image, "ask", prompt=prompt)
        return response["response"]

    def score_observation(self, image: np.ndarray, target_object: str) -> dict:
        response = self._request(
            image,
            "score_observation",
            target_object=target_object,
        )
        return {
            "target": float(response["target"]),
            "exploration": float(response["exploration"]),
            "raw": response.get("raw", ""),
        }

    def verify_detection(self, image: np.ndarray, target_object: str) -> Dict[str, Any]:
        response = self._request(
            image,
            "verify_detection",
            target_object=target_object,
        )
        return {
            "verified": bool(response["verified"]),
            "raw": response.get("raw", ""),
        }


# Main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--model-path", type=str, default="/root/objnav/vlfm/qwen/qwen3.5-2B")
    args = parser.parse_args()

    print("[QwenVL] Starting server...")

    class QwenVLServer(ServerMixin, QwenVLModel):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            method = payload.get("method", "ask")

            if method == "score_observation":
                result = self.score_observation(image, payload["target_object"])
                return {
                    "response": result["raw"],
                    "target": result["target"],
                    "exploration": result["exploration"],
                    "raw": result["raw"],
                }
            elif method == "verify_detection":
                result = self.verify_detection(image, payload["target_object"])
                return {
                    "verified": result["verified"],
                    "raw": result["raw"],
                }
            else:
                # ask (VQA)
                response = self.ask(image, payload.get("prompt"))
                return {"response": response}

    server = QwenVLServer(model_path=args.model_path)
    host_model(server, name="qwen_vl", port=args.port)
