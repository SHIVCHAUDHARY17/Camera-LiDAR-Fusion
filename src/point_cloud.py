"""
Velodyne point cloud utilities for KITTI.

Reads raw .bin files and provides basic range filtering.
The actual projection onto image space is handled by KITTICalibration,
which was already built and tested — this module just handles I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def read_velodyne_bin(path: Union[str, Path]) -> np.ndarray:
    """
    Read a KITTI Velodyne .bin file.

    Returns:
        (N, 4) float32 array: [x, y, z, intensity]
        LiDAR frame: x=forward, y=left, z=up
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Velodyne file not found: {path}")
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def filter_fov(
    points: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 60.0,
    y_min: float = -30.0,
    y_max: float = 30.0,
) -> np.ndarray:
    """
    Keep only points within a rectangular forward-facing field of view.

    Removes points behind the vehicle (x < 0) and outliers.
    """
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    return points[mask]
