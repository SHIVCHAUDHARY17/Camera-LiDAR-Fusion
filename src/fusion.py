"""
Late fusion of camera and LiDAR 2D detections.

Pipeline:
  1. Map camera (COCO) and LiDAR (KITTI 3-class) detections to a unified
     class taxonomy: Car, Pedestrian, Cyclist.
  2. Within each class group, build a 2D IoU matrix between camera and
     LiDAR boxes.
  3. Solve the assignment problem with the Hungarian algorithm
     (scipy.optimize.linear_sum_assignment) to find the best one-to-one
     matching.
  4. Drop matches below the IoU threshold.
  5. For real matches: emit a FusedDetection with weighted-average bbox
     and weighted-average confidence.
  6. For unmatched detections: emit as-is if confidence > min_confidence.

Late fusion = each sensor's detector runs independently; we only combine
final detections. Easy to add/remove sensors without retraining anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detections import Detection2D


# Unified taxonomy used internally by the fusion module.
UNIFIED_CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

# COCO class id (camera/YOLO output) -> unified class id.
# Camera classes we keep: person, bicycle, car, motorcycle, bus, truck.
COCO_TO_UNIFIED = {
    0: 1,  # person      -> Pedestrian
    1: 2,  # bicycle     -> Cyclist
    2: 0,  # car         -> Car
    3: 2,  # motorcycle  -> Cyclist
    5: 0,  # bus         -> Car
    7: 0,  # truck       -> Car
}


def unified_class_id(detection: Detection2D) -> int:
    """
    Translate a detection's class_id into the unified taxonomy.

    LiDAR detections (StubLidarDetector) already use unified ids (0/1/2).
    Camera detections (YOLO) use COCO ids and need translation.
    Returns -1 for detections we cannot map.
    """
    if detection.source == "lidar":
        return detection.class_id
    return COCO_TO_UNIFIED.get(detection.class_id, -1)


# ---------------------------------------------------------------------- #
# IoU
# ---------------------------------------------------------------------- #


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    2D IoU between two boxes [x_min, y_min, x_max, y_max].
    Returns 0.0 if they don't overlap or have zero area.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of bboxes.

    Args:
        boxes_a: (M, 4)
        boxes_b: (N, 4)

    Returns:
        (M, N) array of IoU values in [0, 1].
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    a = boxes_a[:, None, :]  # (M, 1, 4)
    b = boxes_b[None, :, :]  # (1, N, 4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(
        a[..., 3] - a[..., 1], 0.0, None
    )
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(
        b[..., 3] - b[..., 1], 0.0, None
    )
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


# ---------------------------------------------------------------------- #
# Fused detection container
# ---------------------------------------------------------------------- #


@dataclass
class FusedDetection:
    """
    Output of the fusion module.

    Three flavours via `source`:
      "fused"       - matched camera + LiDAR pair, blended bbox + score
      "camera_only" - camera detection that didn't match any LiDAR box
      "lidar_only"  - LiDAR detection that didn't match any camera box
    """

    bbox: np.ndarray  # (4,) [x_min, y_min, x_max, y_max]
    confidence: float
    class_id: int  # unified taxonomy
    class_name: str
    source: str  # "fused" | "camera_only" | "lidar_only"
    iou: float = 0.0  # IoU of the match (0 if unmatched)
    camera_det: Optional[Detection2D] = None
    lidar_det: Optional[Detection2D] = None


# ---------------------------------------------------------------------- #
# Fusion engine
# ---------------------------------------------------------------------- #


class Fusion:
    """Late fusion of camera + LiDAR detections via 2D IoU + Hungarian."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        camera_weight: float = 0.4,
        lidar_weight: float = 0.6,
        min_confidence: float = 0.2,
    ) -> None:
        if abs((camera_weight + lidar_weight) - 1.0) > 1e-6:
            raise ValueError(
                f"camera_weight + lidar_weight must sum to 1.0, "
                f"got {camera_weight + lidar_weight:.3f}"
            )
        self.iou_threshold = float(iou_threshold)
        self.camera_weight = float(camera_weight)
        self.lidar_weight = float(lidar_weight)
        self.min_confidence = float(min_confidence)

    def fuse(
        self,
        camera_dets: Sequence[Detection2D],
        lidar_dets: Sequence[Detection2D],
    ) -> List[FusedDetection]:
        """
        Match camera + LiDAR detections and return a fused detection list.

        Matching is class-aware: a camera person never matches a LiDAR Car
        even if the boxes overlap.
        """
        cam_groups = self._group_by_unified_class(camera_dets)
        lid_groups = self._group_by_unified_class(lidar_dets)

        fused: List[FusedDetection] = []
        matched_cam: set[int] = set()
        matched_lid: set[int] = set()

        # Per-class Hungarian assignment
        for cls_id in set(cam_groups) | set(lid_groups):
            cam_list = cam_groups.get(cls_id, [])
            lid_list = lid_groups.get(cls_id, [])
            if not cam_list or not lid_list:
                continue  # no possible matches in this class

            cam_boxes = np.stack([d.bbox for _, d in cam_list])
            lid_boxes = np.stack([d.bbox for _, d in lid_list])
            iou_mat = bbox_iou_matrix(cam_boxes, lid_boxes)  # (M, N)

            # Hungarian minimises cost. We want to MAX iou -> minimise -iou.
            row_idx, col_idx = linear_sum_assignment(-iou_mat)

            for r, c in zip(row_idx, col_idx):
                iou = float(iou_mat[r, c])
                if iou < self.iou_threshold:
                    continue  # below threshold = not a real match

                cam_global, cam_det = cam_list[r]
                lid_global, lid_det = lid_list[c]
                fused.append(self._fuse_pair(cam_det, lid_det, iou))
                matched_cam.add(cam_global)
                matched_lid.add(lid_global)

        # Unmatched camera detections
        for i, det in enumerate(camera_dets):
            if i in matched_cam or det.confidence < self.min_confidence:
                continue
            cls_id = unified_class_id(det)
            if cls_id < 0:
                continue
            fused.append(
                FusedDetection(
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    class_id=cls_id,
                    class_name=UNIFIED_CLASS_NAMES[cls_id],
                    source="camera_only",
                    camera_det=det,
                )
            )

        # Unmatched LiDAR detections
        for i, det in enumerate(lidar_dets):
            if i in matched_lid or det.confidence < self.min_confidence:
                continue
            cls_id = unified_class_id(det)
            if cls_id < 0:
                continue
            fused.append(
                FusedDetection(
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    class_id=cls_id,
                    class_name=UNIFIED_CLASS_NAMES[cls_id],
                    source="lidar_only",
                    lidar_det=det,
                )
            )

        return fused

    # ------------------------------------------------------------------ #

    def _fuse_pair(
        self, cam: Detection2D, lid: Detection2D, iou: float
    ) -> FusedDetection:
        """
        Combine a matched camera/LiDAR pair into one FusedDetection.

        bbox     = weighted average of the two boxes (a compromise between
                   sensors; could alternatively favour camera for image-space
                   output - left as future tuning)
        score    = weighted average of confidences
        class_id = same for both by construction (class-aware matching)
        """
        bbox = self.camera_weight * cam.bbox + self.lidar_weight * lid.bbox
        score = self.camera_weight * cam.confidence + self.lidar_weight * lid.confidence
        cls_id = unified_class_id(lid)
        return FusedDetection(
            bbox=bbox,
            confidence=float(score),
            class_id=cls_id,
            class_name=UNIFIED_CLASS_NAMES[cls_id],
            source="fused",
            iou=iou,
            camera_det=cam,
            lidar_det=lid,
        )

    @staticmethod
    def _group_by_unified_class(
        detections: Sequence[Detection2D],
    ) -> Dict[int, List[Tuple[int, Detection2D]]]:
        """
        Group detections by unified class id, preserving each detection's
        original index so we can later mark it matched/unmatched.

        Returns: {class_id: [(global_idx, det), ...]}
        """
        groups: Dict[int, List[Tuple[int, Detection2D]]] = {}
        for i, det in enumerate(detections):
            cls = unified_class_id(det)
            if cls < 0:
                continue
            groups.setdefault(cls, []).append((i, det))
        return groups
