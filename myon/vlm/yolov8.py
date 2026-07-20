# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np
import torch

from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.server_wrapper import (
    ServerMixin,
    host_model,
    send_request,
    str_to_image,
)

class YOLOv8:
    """YOLOv8 object detector, replacement for YOLOv7.

    Uses the ultralytics YOLO package which supports YOLOv8/v9/v10/v11.
    """

    def __init__(
        self,
        weights: str = "yolov8x.pt",
        image_size: int = 640,
        device: Optional[torch.device] = None,
    ):
        from ultralytics import YOLO

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = YOLO(weights)
        self.image_size = image_size

        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, self.image_size, self.image_size).to(self.device)
            for _ in range(3):
                self.model(dummy_img, verbose=False)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        """Run detection on an RGB image.

        Args:
            image: RGB image as numpy array (H, W, 3).
            conf_thres: Confidence threshold for NMS filtering.
            iou_thres: IoU threshold for NMS.

        Returns:
            ObjectDetections with normalized xyxy boxes.
        """
        results = self.model.predict(
            image,
            imgsz=self.image_size,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return ObjectDetections(
                boxes=torch.empty((0, 4)),
                logits=torch.empty(0),
                phrases=[],
                image_source=image,
                fmt="xyxy",
            )

        h, w = image.shape[:2]
        xyxy = boxes.xyxy.cpu()
        xyxy[:, [0, 2]] /= w
        xyxy[:, [1, 3]] /= h

        logits = boxes.conf.cpu()
        phrases = [self.model.names[int(c)] for c in boxes.cls.cpu()]

        return ObjectDetections(xyxy, logits, phrases, image_source=image, fmt="xyxy")


class YOLOv8Client:
    """HTTP client for a remote YOLOv8 server."""

    def __init__(self, port: int = 12186):
        self.url = f"http://localhost:{port}/yolov8"

    def predict(
        self,
        image_numpy: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        return ObjectDetections.from_json(response, image_source=image_numpy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12186)
    parser.add_argument("--weights", type=str, default="yolov8x.pt")
    args = parser.parse_args()

    print("Loading YOLOv8...")

    class YOLOv8Server(ServerMixin, YOLOv8):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            conf = payload.get("conf_thres", 0.25)
            iou = payload.get("iou_thres", 0.45)
            return self.predict(image, conf_thres=conf, iou_thres=iou).to_json()

    server = YOLOv8Server(weights=args.weights)
    print("Model loaded!")
    print(f"Hosting YOLOv8 on port {args.port}...")
    host_model(server, name="yolov8", port=args.port)
