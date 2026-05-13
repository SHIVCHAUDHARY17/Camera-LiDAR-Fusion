"""
KITTI calibration utilities.

A KITTI calib_2/000XXX.txt file contains:
    P0, P1, P2, P3    : 3x4 perspective projection matrices for cameras 0-3
    R0_rect           : 3x3 rectifying rotation (aligns the stereo pair)
    Tr_velo_to_cam    : 3x4 rigid transform Velodyne -> camera 0 frame
    Tr_imu_to_velo    : 3x4 rigid transform IMU -> Velodyne frame  (unused here)

We only need P2, R0_rect, Tr_velo_to_cam for projecting LiDAR -> image_2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class KITTICalibration:
    """Loader and projector for a single KITTI scene's calibration."""

    def __init__(
        self,
        P2: np.ndarray,
        R0_rect: np.ndarray,
        Tr_velo_to_cam: np.ndarray,
    ) -> None:
        # Store raw matrices in their native shapes.
        self.P2 = P2.reshape(3, 4)
        self.R0_rect = R0_rect.reshape(3, 3)
        self.Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)

        # Build 4x4 homogeneous versions so the projection chain is one multiply.
        self.R0_rect_4x4 = np.eye(4)
        self.R0_rect_4x4[:3, :3] = self.R0_rect

        self.Tr_velo_to_cam_4x4 = np.eye(4)
        self.Tr_velo_to_cam_4x4[:3, :4] = self.Tr_velo_to_cam

        # Pre-compute full 3x4 LiDAR -> image_2 projection.
        # Applied to a homogeneous LiDAR point [x, y, z, 1]^T it returns
        # [u*z, v*z, z]^T in image_2.
        self.proj_lidar_to_image = self.P2 @ self.R0_rect_4x4 @ self.Tr_velo_to_cam_4x4

    # ------------------------------------------------------------------ #
    # File loading
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "KITTICalibration":
        """Load calibration from a KITTI calib .txt file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        data: dict[str, np.ndarray] = {}
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key.strip()] = np.array(
                        [float(x) for x in value.split()], dtype=np.float64
                    )
                except ValueError:
                    # Skip non-numeric lines (e.g. comments)
                    continue

        missing = [k for k in ("P2", "R0_rect", "Tr_velo_to_cam") if k not in data]
        if missing:
            raise ValueError(f"Calibration file {path} is missing keys: {missing}")

        return cls(
            P2=data["P2"],
            R0_rect=data["R0_rect"],
            Tr_velo_to_cam=data["Tr_velo_to_cam"],
        )

    # ------------------------------------------------------------------ #
    # LiDAR -> camera / image projection
    # ------------------------------------------------------------------ #

    def lidar_to_camera(self, points_lidar: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR points (N, 3) into the rectified cam-0 frame (N, 3).

        Useful when you need the depth (z value) of a LiDAR point in camera
        coordinates without going all the way to pixels.
        """
        if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {points_lidar.shape}")

        n = points_lidar.shape[0]
        points_h = np.hstack([points_lidar, np.ones((n, 1))])  # (N, 4)
        points_cam_h = (self.R0_rect_4x4 @ self.Tr_velo_to_cam_4x4 @ points_h.T).T
        return points_cam_h[:, :3]

    def lidar_to_image(
        self,
        points_lidar: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points (N, 3) into image_2 pixel coordinates.

        Args:
            points_lidar: (N, 3) array in Velodyne frame.
            image_shape:  optional (height, width). If given, the returned mask
                          is True only for points that ALSO fall inside the image.

        Returns:
            pixels: (N, 2) array of (u, v) pixel coords (float).
            depths: (N,)   depth in rectified camera frame (z, in metres).
            mask:   (N,)   bool. True for points in front of the camera
                          (and inside image bounds, if image_shape given).
        """
        if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {points_lidar.shape}")

        n = points_lidar.shape[0]
        points_h = np.hstack([points_lidar, np.ones((n, 1))])  # (N, 4)
        proj = self.proj_lidar_to_image @ points_h.T  # (3, N)

        depths = proj[2, :]
        # Avoid divide-by-zero for points exactly at the image plane;
        # those are filtered by the mask anyway.
        safe_depths = np.where(np.abs(depths) < 1e-6, 1e-6, depths)
        u = proj[0, :] / safe_depths
        v = proj[1, :] / safe_depths
        pixels = np.stack([u, v], axis=1)  # (N, 2)

        mask = depths > 0  # in front of the camera

        if image_shape is not None:
            h, w = image_shape
            in_bounds = (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < w)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < h)
            )
            mask = mask & in_bounds

        return pixels, depths, mask

    # ------------------------------------------------------------------ #
    # 3D bounding box projection
    # ------------------------------------------------------------------ #

    def project_box_3d_to_image(
        self,
        center: np.ndarray,
        dimensions: np.ndarray,
        yaw: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project a 3D bounding box (in LiDAR frame) into image_2.

        Args:
            center:      (3,) (x, y, z) - bottom-centre of box, LiDAR frame.
            dimensions:  (3,) (length, width, height).
            yaw:         rotation around z-axis in radians.
            image_shape: optional (height, width) for clipping the 2D bbox.

        Returns:
            corners_2d: (8, 2) projected corner pixel coords.
            bbox_2d:    (4,)   (x_min, y_min, x_max, y_max) tightest 2D bbox
                               around the projected corners. Clipped to
                               image bounds if image_shape is provided.
        """
        corners_3d = self._box_3d_corners(center, dimensions, yaw)  # (8, 3)
        corners_2d, _, _ = self.lidar_to_image(corners_3d)

        x_min = float(np.min(corners_2d[:, 0]))
        y_min = float(np.min(corners_2d[:, 1]))
        x_max = float(np.max(corners_2d[:, 0]))
        y_max = float(np.max(corners_2d[:, 1]))

        if image_shape is not None:
            h, w = image_shape
            x_min = max(0.0, min(x_min, w - 1))
            x_max = max(0.0, min(x_max, w - 1))
            y_min = max(0.0, min(y_min, h - 1))
            y_max = max(0.0, min(y_max, h - 1))

        bbox_2d = np.array([x_min, y_min, x_max, y_max], dtype=np.float64)
        return corners_2d, bbox_2d

    @staticmethod
    def _box_3d_corners(
        center: np.ndarray, dimensions: np.ndarray, yaw: float
    ) -> np.ndarray:
        """
        Compute 8 corners of a 3D box in LiDAR frame.

        PointPillars / KITTI LiDAR convention:
          - centre is the BOTTOM-centre of the box (sits on the ground)
          - dimensions are (length along x, width along y, height along z)
          - yaw rotates around the z-axis (vertical, "heading" angle)

        Corner order (z-up, looking down):
          0,1,2,3 = bottom face
          4,5,6,7 = top face (same x,y as 0..3, z + h)
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        dimensions = np.asarray(dimensions, dtype=np.float64).reshape(3)
        l, w, h = dimensions

        # Canonical box at origin, axis-aligned.
        x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        z = np.array([0, 0, 0, 0, h, h, h, h])
        corners = np.stack([x, y, z], axis=0)  # (3, 8)

        # Rotate around z-axis (yaw).
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        corners = R @ corners  # (3, 8)

        # Translate to box centre.
        corners += center.reshape(3, 1)

        return corners.T  # (8, 3)
