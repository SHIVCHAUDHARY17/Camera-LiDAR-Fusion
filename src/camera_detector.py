"""
YOLO camera detector wrapper.

Thin layer over Ultralytics YOLO that:
  - loads a pretrained model once
  - applies the configured confidence + class-id filter
  - returns Detection2D objects ready for fusion
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from src.detections import Detection2D


class CameraDetector:
    """YOLO inference on KITTI image_2 frames (or any RGB image)."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.3,
        classes: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            weights:    path to YOLO weights (auto-downloaded by ultralytics
                        on first use if name is a known checkpoint).
            confidence: minimum detection score to keep.
            classes:    list of class IDs to keep. None = keep everything.
                        For KITTI we use COCO ids: person(0), bicycle(1),
                        car(2), motorcycle(3), bus(5), truck(7).
            device:     "cpu" or "cuda" / "cuda:0". On a 4GB GTX 1650 with
                        PointPillars also on the GPU, "cpu" is the safe
                        default for YOLOv8n.
        """
        self.model = YOLO(weights)
        self.confidence = float(confidence)
        self.classes = list(classes) if classes is not None else None
        self.device = device
        # ultralytics exposes COCO class names as a {id: name} dict
        self.class_names: dict[int, str] = self.model.names

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[Detection2D]:
        """
        Run detection on a single image.

        Args:
            image: path to image file, or HxWx3 BGR numpy array (OpenCV format).

        Returns:
            List of Detection2D filtered by confidence and class.
        """
        source = str(image) if isinstance(image, (str, Path)) else image

        results = self.model.predict(
            source=source,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection2D] = []
        if not results:
            return detections

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = result.boxes.conf.cpu().numpy()  # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for box, score, cid in zip(boxes_xyxy, scores, class_ids):
                detections.append(
                    Detection2D(
                        bbox=box.astype(np.float64),
                        confidence=float(score),
                        class_id=int(cid),
                        class_name=self.class_names.get(int(cid), str(cid)),
                        source="camera",
                    )
                )

        return detections
