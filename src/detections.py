"""
Shared detection types for camera and LiDAR detectors.

Keeping these in one place means the fusion module doesn't need separate
code paths for camera vs LiDAR detections - they are interchangeable at
the 2D-bbox level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection2D:
    """A 2D detection in image space, optionally backed by a 3D box."""

    bbox: np.ndarray  # (4,) [x_min, y_min, x_max, y_max] in pixels
    confidence: float  # detector score in [0, 1]
    class_id: int  # source-specific class id (COCO for camera, KITTI-ish for LiDAR)
    class_name: str  # human-readable label
    source: str = "camera"  # "camera" | "lidar"

    # Optional 3D info, populated for LiDAR detections.
    # When present, bbox is the projection of the 3D box into image space.
    center_3d: Optional[np.ndarray] = None  # (3,) (x, y, z) bottom-centre, LiDAR frame
    dimensions_3d: Optional[np.ndarray] = None  # (3,) (length, width, height)
    yaw: Optional[float] = None  # rotation around z-axis, radians
    corners_2d: Optional[np.ndarray] = None  # (8, 2) projected corner pixels

    def area(self) -> float:
        """Area of the 2D bbox in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
