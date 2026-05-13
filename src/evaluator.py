"""
Per-class Average Precision evaluator for camera, LiDAR, and fused detections.

Matches predictions against KITTI ground-truth bounding boxes using 2D IoU,
then computes 11-point interpolated AP per class.

Classes: Car, Pedestrian, Cyclist (standard KITTI 3-class evaluation).
IoU threshold: 0.5 (KITTI standard for Pedestrian/Cyclist; Car normally uses 0.7
but we use 0.5 uniformly for simplicity).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.fusion import FusedDetection, UNIFIED_CLASS_NAMES, unified_class_id
from src.detections import Detection2D


# KITTI GT class -> unified class id (same mapping as lidar_detector)
GT_CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}

EVAL_CLASSES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


# ---------------------------------------------------------------------- #
# GT loader
# ---------------------------------------------------------------------- #


def load_gt_boxes(label_path: str | Path) -> Dict[int, List[np.ndarray]]:
    """
    Load ground-truth 2D bounding boxes from a KITTI label_2 file.

    Returns: {class_id: [bbox_array, ...]} where bbox is [x1, y1, x2, y2].
    Skips DontCare, Tram, Misc.
    """
    boxes: Dict[int, List[np.ndarray]] = defaultdict(list)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in GT_CLASS_MAP:
                continue
            cls_id = GT_CLASS_MAP[cls]
            bbox = np.array([float(x) for x in parts[4:8]], dtype=np.float64)
            boxes[cls_id].append(bbox)
    return dict(boxes)


# ---------------------------------------------------------------------- #
# IoU helper (local — avoids importing fusion module's version)
# ---------------------------------------------------------------------- #


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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


# ---------------------------------------------------------------------- #
# Evaluator
# ---------------------------------------------------------------------- #


class KITTIEvaluator:
    """
    Accumulates predictions across scenes and computes per-class AP.

    Usage:
        ev = KITTIEvaluator()
        for scene_id in val_split:
            gt = load_gt_boxes(label_path)
            ev.add_scene(gt, camera_dets, lidar_dets, fused_dets)
        ev.print_summary()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        # source -> class_id -> [(confidence, is_tp), ...]
        self._records: Dict[str, Dict[int, List[Tuple[float, bool]]]] = {
            "camera": defaultdict(list),
            "lidar": defaultdict(list),
            "fused": defaultdict(list),
        }
        # class_id -> total GT count across all scenes
        self._n_gt: Dict[int, int] = defaultdict(int)

    def add_scene(
        self,
        gt_boxes: Dict[int, List[np.ndarray]],
        camera_dets: Sequence[Detection2D],
        lidar_dets: Sequence[Detection2D],
        fused_dets: Sequence[FusedDetection],
    ) -> None:
        """Process one scene: match predictions to GT, record TP/FP."""
        for cls_id, boxes in gt_boxes.items():
            self._n_gt[cls_id] += len(boxes)

        self._match(
            "camera",
            gt_boxes,
            [(unified_class_id(d), d.bbox, d.confidence) for d in camera_dets],
        )
        self._match(
            "lidar",
            gt_boxes,
            [(unified_class_id(d), d.bbox, d.confidence) for d in lidar_dets],
        )
        self._match(
            "fused",
            gt_boxes,
            [(d.class_id, d.bbox, d.confidence) for d in fused_dets],
        )

    def _match(
        self,
        source: str,
        gt_boxes: Dict[int, List[np.ndarray]],
        preds: List[Tuple[int, np.ndarray, float]],
    ) -> None:
        """Greedy IoU matching: highest-confidence predictions claim GT boxes first."""
        by_cls: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
        for cls_id, bbox, conf in preds:
            if cls_id >= 0:
                by_cls[cls_id].append((conf, bbox))

        for cls_id, cls_preds in by_cls.items():
            gt_list = list(gt_boxes.get(cls_id, []))
            matched_gt: set[int] = set()

            # Process highest-confidence predictions first
            for conf, pred_box in sorted(cls_preds, key=lambda x: -x[0]):
                best_iou, best_idx = 0.0, -1
                for i, gt_box in enumerate(gt_list):
                    if i in matched_gt:
                        continue
                    iou = _iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i

                if best_iou >= self.iou_threshold and best_idx >= 0:
                    matched_gt.add(best_idx)
                    self._records[source][cls_id].append((conf, True))  # TP
                else:
                    self._records[source][cls_id].append((conf, False))  # FP

    def _ap(self, source: str, cls_id: int) -> float:
        """11-point interpolated AP for one source and class."""
        records = self._records[source].get(cls_id, [])
        n_gt = self._n_gt.get(cls_id, 0)
        if not records or n_gt == 0:
            return 0.0

        sorted_records = sorted(records, key=lambda x: -x[0])
        tp_cum = fp_cum = 0
        precisions, recalls = [], []
        for _, is_tp in sorted_records:
            tp_cum += int(is_tp)
            fp_cum += int(not is_tp)
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)

        ap = 0.0
        for thr in np.linspace(0.0, 1.0, 11):
            prec_vals = [p for p, r in zip(precisions, recalls) if r >= thr]
            ap += max(prec_vals) if prec_vals else 0.0
        return ap / 11.0

    def print_summary(self) -> None:
        """Print AP comparison table for all classes and sources."""
        print("\n" + "=" * 62)
        print(
            f"{'Evaluation Results (IoU threshold = ' + str(self.iou_threshold) + ')':^62}"
        )
        print("=" * 62)
        print(f"{'Class':<14} {'Camera AP':>12} {'LiDAR AP':>12} {'Fused AP':>12}")
        print("-" * 62)

        map_scores: Dict[str, List[float]] = defaultdict(list)
        for cls_id, cls_name in EVAL_CLASSES.items():
            aps = {s: self._ap(s, cls_id) for s in ["camera", "lidar", "fused"]}
            for s, v in aps.items():
                map_scores[s].append(v)
            print(
                f"{cls_name:<14}"
                f"{aps['camera']:>12.3f}"
                f"{aps['lidar']:>12.3f}"
                f"{aps['fused']:>12.3f}"
            )

        print("-" * 62)
        print(
            f"{'mAP':<14}"
            f"{np.mean(map_scores['camera']):>12.3f}"
            f"{np.mean(map_scores['lidar']):>12.3f}"
            f"{np.mean(map_scores['fused']):>12.3f}"
        )
        print("=" * 62)
        print(
            "\nNote: LiDAR uses a ground-truth-backed stub with 10% dropout"
            "\nand Gaussian noise. Camera AP reflects real YOLOv8n performance."
            "\nFused AP shows cross-sensor verification benefit.\n"
        )
