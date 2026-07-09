# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
Qwen3-VL client for visual question answering and image-text matching.

Uses vLLM's OpenAI-compatible API server to serve Qwen3-VL-8B-Instruct.
Replaces both BLIP2 (VQA) and BLIP2ITM (image-text cosine similarity).

Usage:
    # Start the vLLM server first:
    vllm serve /root/objnav/vlfm/qwen --port 12182

    # Then use the client:
    client = QwenVLClient(port=12182)
    answer = client.ask(image, "Is this a chair?")
    score = client.cosine(image, "a red chair")
"""

import base64
import io
import time
from typing import Optional

import cv2
import numpy as np
import requests


class QwenVLClient:
    """HTTP client for Qwen3-VL served via vLLM OpenAI-compatible API.

    Args:
        port: Port number of the vLLM server (default 12182, replaces BLIP2ITM port).
        base_url: Base URL of the vLLM server. Overrides port if provided.
        model_name: Name of the model. vLLM auto-detects this.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries on failure.
    """

    def __init__(
        self,
        port: int = 12182,
        base_url: Optional[str] = None,
        model_name: str = "Qwen3-VL-8B-Instruct",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        if base_url is None:
            base_url = f"http://localhost:{port}"
        self.url = f"{base_url}/v1/chat/completions"
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert a numpy image (RGB) to a base64-encoded data URL string."""
        # Convert RGB to BGR for cv2
        if image.shape[-1] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image
        _, buffer = cv2.imencode(".jpg", img_bgr)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"

    def _build_payload(self, image: np.ndarray, prompt: str) -> dict:
        """Build the OpenAI-compatible chat completions payload."""
        img_url = self._image_to_base64(image)
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

    def _request(self, payload: dict) -> str:
        """Send a request to the vLLM server with retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"

            if attempt < self.max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[QwenVLClient] {last_error}. Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"QwenVLClient request failed: {last_error}")

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """Ask a question about an image. Replaces BLIP2.ask().

        Args:
            image: RGB image as numpy array (H, W, 3).
            prompt: The question to ask about the image.

        Returns:
            The model's text response.
        """
        if prompt is None or prompt == "":
            prompt = "Describe this image in detail."

        payload = self._build_payload(image, prompt)
        return self._request(payload)

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """Compute relevance score between image and text. Replaces BLIP2ITM.cosine().

        Args:
            image: RGB image as numpy array (H, W, 3).
            txt: Text description to compare against.

        Returns:
            Float relevance score between 0.0 and 1.0.
        """
        prompt = (
            f'On a scale of 0 to 100, how relevant is this image to the description '
            f'"{txt}"? '
            f'Answer with ONLY a single integer number from 0 (completely irrelevant) '
            f'to 100 (perfect match). No other text.'
        )
        payload = self._build_payload(image, prompt)
        # Use slightly higher max_tokens since we need to parse a number
        payload["max_tokens"] = 16

        response = self._request(payload)

        # Parse the integer from the response
        try:
            # Extract the first integer found in the response
            digits = "".join(c for c in response if c.isdigit() or c == "-")
            score = int(digits)
            return max(0.0, min(1.0, score / 100.0))
        except (ValueError, TypeError):
            print(f"[QwenVLClient] Failed to parse score from: '{response}'. Returning 0.0.")
            return 0.0
