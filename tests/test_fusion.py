"""
Unit tests for Fusion, bbox_iou, and bbox_iou_matrix.

All tests use synthetic Detection2D objects — no KITTI data or GPU required.
Covers: IoU math, Hungarian matching, class-aware gating, confidence
weighting, unmatched handling, and edge cases.
"""

import numpy as np
import pytest

from src.detections import Detection2D
from src.fusion import Fusion, bbox_iou, bbox_iou_matrix


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def cam(bbox, conf, coco_id):
    """Shorthand for a camera detection."""
    return Detection2D(
        bbox=np.array(bbox, dtype=np.float64),
        confidence=conf,
        class_id=coco_id,
        class_name="dummy",
        source="camera",
    )


def lid(bbox, conf, unified_id):
    """Shorthand for a LiDAR detection (already uses unified class ids)."""
    return Detection2D(
        bbox=np.array(bbox, dtype=np.float64),
        confidence=conf,
        class_id=unified_id,
        class_name="dummy",
        source="lidar",
    )


# ---------------------------------------------------------------------- #
# IoU tests
# ---------------------------------------------------------------------- #


def test_iou_identical_boxes():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert bbox_iou(
        np.array([0.0, 0.0, 5.0, 5.0]),
        np.array([10.0, 10.0, 20.0, 20.0]),
    ) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # Intersection = [5,5,10,10] = 25, union = 100+100-25 = 175
    assert bbox_iou(
        np.array([0.0, 0.0, 10.0, 10.0]),
        np.array([5.0, 5.0, 15.0, 15.0]),
    ) == pytest.approx(25.0 / 175.0)


def test_iou_matrix_shape():
    a = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]])  # (2, 4)
    b = np.array(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]]
    )  # (3, 4)
    mat = bbox_iou_matrix(a, b)
    assert mat.shape == (2, 3)


def test_iou_matrix_diagonal_is_one_for_identical():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    mat = bbox_iou_matrix(boxes, boxes)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[1, 1] == pytest.approx(1.0)
    assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------- #
# Fusion edge cases
# ---------------------------------------------------------------------- #


def test_fusion_both_empty():
    assert Fusion().fuse([], []) == []


def test_fusion_camera_only():
    out = Fusion().fuse([cam([0, 0, 10, 10], 0.9, 2)], [])
    assert len(out) == 1
    assert out[0].source == "camera_only"
    assert out[0].confidence == pytest.approx(0.9)


def test_fusion_lidar_only():
    out = Fusion().fuse([], [lid([0, 0, 10, 10], 0.8, 0)])
    assert len(out) == 1
    assert out[0].source == "lidar_only"
    assert out[0].confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------- #
# Fusion matching tests
# ---------------------------------------------------------------------- #


def test_fusion_matched_pair():
    """Camera car + LiDAR Car with identical bboxes should fuse."""
    c = cam([0.0, 0.0, 10.0, 10.0], 0.8, 2)  # COCO car (2) -> unified Car (0)
    l = lid([0.0, 0.0, 10.0, 10.0], 0.9, 0)  # unified Car (0)
    f = Fusion(iou_threshold=0.5, camera_weight=0.4, lidar_weight=0.6)
    out = f.fuse([c], [l])
    assert len(out) == 1
    assert out[0].source == "fused"
    assert out[0].iou == pytest.approx(1.0)
    assert out[0].confidence == pytest.approx(0.4 * 0.8 + 0.6 * 0.9)


def test_fusion_class_mismatch_prevents_fusion():
    """Camera person and LiDAR Car should NOT fuse even with perfect bbox overlap."""
    c = cam([0.0, 0.0, 10.0, 10.0], 0.9, 0)  # COCO person (0) -> Pedestrian (1)
    l = lid([0.0, 0.0, 10.0, 10.0], 0.9, 0)  # unified Car (0)
    f = Fusion(iou_threshold=0.1, min_confidence=0.0)
    out = f.fuse([c], [l])
    sources = {r.source for r in out}
    assert "fused" not in sources
    assert "camera_only" in sources
    assert "lidar_only" in sources


def test_fusion_below_iou_threshold_stays_unmatched():
    """Boxes with low overlap should not be fused."""
    c = cam([0.0, 0.0, 10.0, 10.0], 0.8, 2)
    l = lid([9.0, 9.0, 19.0, 19.0], 0.8, 0)  # tiny overlap
    f = Fusion(iou_threshold=0.9, min_confidence=0.0)
    out = f.fuse([c], [l])
    sources = {r.source for r in out}
    assert "fused" not in sources


def test_fusion_min_confidence_filters_unmatched():
    """Unmatched detections below min_confidence should be dropped."""
    c = cam([0.0, 0.0, 10.0, 10.0], 0.1, 2)  # very low confidence
    out = Fusion(min_confidence=0.5).fuse([c], [])
    assert len(out) == 0


def test_fusion_invalid_weights_raises():
    with pytest.raises(ValueError):
        Fusion(camera_weight=0.5, lidar_weight=0.6)  # sum = 1.1, not 1.0


def test_fusion_one_to_one_matching():
    """Hungarian algorithm should give each detection at most one match."""
    # Two camera cars and two LiDAR Cars, all overlapping each other somewhat
    c1 = cam([0.0, 0.0, 20.0, 20.0], 0.8, 2)
    c2 = cam([30.0, 0.0, 50.0, 20.0], 0.7, 2)
    l1 = lid([1.0, 1.0, 21.0, 21.0], 0.9, 0)  # best match for c1
    l2 = lid([31.0, 1.0, 51.0, 21.0], 0.85, 0)  # best match for c2
    f = Fusion(iou_threshold=0.3)
    out = f.fuse([c1, c2], [l1, l2])
    fused_count = sum(1 for r in out if r.source == "fused")
    assert fused_count == 2  # both pairs matched, no double-assignments
