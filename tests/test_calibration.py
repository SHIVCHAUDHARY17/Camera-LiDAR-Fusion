"""
Unit tests for KITTICalibration.

All tests use a synthetic calibration so no KITTI data files are required.
This makes them safe to run in CI without the dataset present.

Synthetic calibration:
  - Focal length 700, principal point (640, 190)
  - Tr_velo_to_cam maps LiDAR x-forward -> camera z (depth),
    so a point at (10, 0, 0) in LiDAR has depth=10 in camera.
"""

import numpy as np
import pytest

from src.calibration import KITTICalibration


# ---------------------------------------------------------------------- #
# Fixture
# ---------------------------------------------------------------------- #


def make_calib() -> KITTICalibration:
    """
    Synthetic calibration with clean expected values.

    LiDAR frame: x=forward, y=left, z=up
    Camera frame: x=right,   y=down,  z=forward

    Rotation: cam_x = -lid_y, cam_y = -lid_z, cam_z = lid_x
    So (10, 0, 0) in LiDAR -> (0, 0, 10) in camera -> depth=10 pixels=(640,190)
    """
    P2 = np.array(
        [
            700.0,
            0.0,
            640.0,
            0.0,
            0.0,
            700.0,
            190.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )
    R0_rect = np.eye(3).flatten()
    Tr_velo_to_cam = np.array(
        [
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    return KITTICalibration(P2, R0_rect, Tr_velo_to_cam)


# ---------------------------------------------------------------------- #
# Matrix shape tests
# ---------------------------------------------------------------------- #


def test_matrix_shapes():
    c = make_calib()
    assert c.P2.shape == (3, 4)
    assert c.R0_rect.shape == (3, 3)
    assert c.Tr_velo_to_cam.shape == (3, 4)
    assert c.proj_lidar_to_image.shape == (3, 4)


# ---------------------------------------------------------------------- #
# Projection tests
# ---------------------------------------------------------------------- #


def test_point_in_front_has_positive_depth():
    c = make_calib()
    pts = np.array([[10.0, 0.0, 0.0]])  # 10m forward in LiDAR
    _, depths, mask = c.lidar_to_image(pts)
    assert depths[0] > 0
    assert mask[0] is True or mask[0] == True


def test_point_behind_camera_is_masked():
    c = make_calib()
    pts = np.array([[-5.0, 0.0, 0.0]])  # behind car in LiDAR = behind camera
    _, depths, mask = c.lidar_to_image(pts)
    assert depths[0] < 0
    assert mask[0] is False or mask[0] == False


def test_on_axis_point_projects_to_principal_point():
    """
    A LiDAR point at (10, 0, 0) maps to camera (0, 0, 10).
    With principal point (640, 190), it should project exactly there.
    """
    c = make_calib()
    pts = np.array([[10.0, 0.0, 0.0]])
    pixels, _, _ = c.lidar_to_image(pts)
    assert pixels[0, 0] == pytest.approx(640.0, abs=1e-6)
    assert pixels[0, 1] == pytest.approx(190.0, abs=1e-6)


def test_output_shapes_batch():
    c = make_calib()
    n = 50
    pts = np.random.randn(n, 3)
    pts[:, 0] = np.abs(pts[:, 0]) + 1.0  # all x>0, all points in front
    pixels, depths, mask = c.lidar_to_image(pts)
    assert pixels.shape == (n, 2)
    assert depths.shape == (n,)
    assert mask.shape == (n,)


def test_image_shape_filter():
    """Points outside image bounds should appear in mask=False."""
    c = make_calib()
    # Point far to the side: large y in LiDAR -> large x in camera -> outside image
    pts = np.array([[10.0, 100.0, 0.0]])
    _, _, mask = c.lidar_to_image(pts, image_shape=(375, 1242))
    assert mask[0] == False


def test_invalid_input_raises():
    c = make_calib()
    with pytest.raises(ValueError):
        c.lidar_to_image(np.ones((5, 4)))  # should be (N, 3)


# ---------------------------------------------------------------------- #
# 3D box corner tests
# ---------------------------------------------------------------------- #


def test_box_corners_shape():
    corners = KITTICalibration._box_3d_corners(
        center=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    assert corners.shape == (8, 3)


def test_box_corners_bottom_face_at_z_zero():
    corners = KITTICalibration._box_3d_corners(
        center=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([4.0, 2.0, 2.0]),
        yaw=0.0,
    )
    assert np.all(corners[:4, 2] == pytest.approx(0.0))


def test_box_corners_top_face_at_height():
    corners = KITTICalibration._box_3d_corners(
        center=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([4.0, 2.0, 2.0]),
        yaw=0.0,
    )
    assert np.all(corners[4:, 2] == pytest.approx(2.0))


def test_box_corners_extents():
    corners = KITTICalibration._box_3d_corners(
        center=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([4.0, 2.0, 2.0]),
        yaw=0.0,
    )
    assert np.max(np.abs(corners[:, 0])) == pytest.approx(2.0)  # l/2 = 2
    assert np.max(np.abs(corners[:, 1])) == pytest.approx(1.0)  # w/2 = 1
