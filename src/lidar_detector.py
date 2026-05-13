"""
Stub LiDAR detector backed by KITTI ground-truth labels.

Reads KITTI label_2/<scene>.txt and returns Detection2D objects with realistic
detector noise (dropout + bbox jitter + sampled confidence scores).

Why a stub: there is no clean drop-in PointPillars ONNX for KITTI on Windows.
Fusion math is independent of detector quality - using ground truth + noise
lets us validate the fusion pipeline end-to-end without spending days on
detector setup. The output shape (Detection2D with 2D bbox + 3D box info)
matches what a real PointPillars detector would emit.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.calibration import KITTICalibration
from src.detections import Detection2D


# Map KITTI label classes to a 3-class output, matching what real KITTI
# PointPillars implementations (Open3D-ML, OpenPCDet) emit.
KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 0,  # group with car
    "Truck": 0,  # group with car
    "Pedestrian": 1,
    "Person_sitting": 1,  # group with pedestrian
    "Cyclist": 2,
    # "Tram", "Misc", "DontCare" are intentionally skipped
}
CLASS_ID_TO_NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


class StubLidarDetector:
    """Simulates PointPillars by reading KITTI labels + adding noise."""

    def __init__(
        self,
        kitti_root: Union[str, Path],
        dropout: float = 0.10,
        bbox_jitter_std: float = 3.0,
        confidence_alpha: float = 8.0,
        confidence_beta: float = 2.0,
        score_threshold: float = 0.3,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Args:
            kitti_root:        path containing calib/, label_2/, etc.
            dropout:           fraction of GT objects to randomly drop (0..1)
            bbox_jitter_std:   stddev of pixel noise added to each bbox corner
            confidence_alpha:  Beta distribution alpha for confidence sampling
            confidence_beta:   Beta distribution beta for confidence sampling
                               (default Beta(8, 2) -> mean ~0.8, mostly 0.6-0.95)
            score_threshold:   drop detections below this score
            seed:              RNG seed (None = non-deterministic)
        """
        self.kitti_root = Path(kitti_root)
        self.dropout = float(dropout)
        self.bbox_jitter_std = float(bbox_jitter_std)
        self.confidence_alpha = float(confidence_alpha)
        self.confidence_beta = float(confidence_beta)
        self.score_threshold = float(score_threshold)
        self.rng = np.random.default_rng(seed)

    def detect(self, scene_id: str) -> List[Detection2D]:
        """
        Return simulated LiDAR detections for one KITTI scene.

        Args:
            scene_id: 6-digit zero-padded string, e.g. "000000".
        """
        label_path = self.kitti_root / "label_2" / f"{scene_id}.txt"
        calib_path = self.kitti_root / "calib" / f"{scene_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not calib_path.is_file():
            raise FileNotFoundError(f"Calib file not found: {calib_path}")

        calib = KITTICalibration.from_file(calib_path)
        labels = self._parse_label_file(label_path)

        detections: List[Detection2D] = []
        for lbl in labels:
            cls_name = lbl["type"]
            if cls_name not in KITTI_CLASS_MAP:
                continue  # skip Tram, Misc, DontCare

            # 1. Dropout - simulate detector miss
            if self.rng.random() < self.dropout:
                continue

            # 2. Sample a confidence score, threshold it
            conf = float(self.rng.beta(self.confidence_alpha, self.confidence_beta))
            if conf < self.score_threshold:
                continue

            # 3. Jitter the 2D bbox corners
            bbox = lbl["bbox_2d"] + self.rng.normal(0.0, self.bbox_jitter_std, size=4)
            x1, y1, x2, y2 = bbox
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue  # box collapsed to nothing - skip

            # 4. Convert 3D box from camera frame (KITTI label) to LiDAR frame
            #    so it matches what a real PointPillars would emit.
            center_lidar, dims_lwh, yaw_lidar = self._cam_to_lidar(
                lbl["loc_cam"], lbl["dims_lwh"], lbl["rotation_y"], calib
            )

            # 5. Project 3D box back to image (8 corners, for visualization)
            corners_2d, _ = calib.project_box_3d_to_image(
                center_lidar, dims_lwh, yaw_lidar
            )

            class_id = KITTI_CLASS_MAP[cls_name]
            detections.append(
                Detection2D(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
                    confidence=conf,
                    class_id=class_id,
                    class_name=CLASS_ID_TO_NAME[class_id],
                    source="lidar",
                    center_3d=center_lidar,
                    dimensions_3d=dims_lwh,
                    yaw=yaw_lidar,
                    corners_2d=corners_2d,
                )
            )

        return detections

    # ------------------------------------------------------------------ #
    # KITTI label file parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_label_file(path: Path) -> List[dict]:
        """
        Parse a KITTI label_2/<scene>.txt file.

        KITTI label line format (15 fields):
            type truncated occluded alpha
            bbox_left bbox_top bbox_right bbox_bottom        [pixels, image]
            height width length                              [metres]
            loc_x loc_y loc_z                                [metres, camera frame]
            rotation_y                                       [radians, camera y-axis]

        loc is the BOTTOM-CENTRE of the 3D box, in rectified camera coords.
        """
        labels: List[dict] = []
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                # KITTI dimension order is (h, w, l). We store (l, w, h)
                # to match LiDAR convention (length along x, width along y,
                # height along z).
                dims_lwh = np.array(
                    [float(parts[10]), float(parts[9]), float(parts[8])],
                    dtype=np.float64,
                )
                labels.append(
                    {
                        "type": parts[0],
                        "bbox_2d": np.array(
                            [float(x) for x in parts[4:8]], dtype=np.float64
                        ),
                        "dims_lwh": dims_lwh,
                        "loc_cam": np.array(
                            [float(parts[11]), float(parts[12]), float(parts[13])],
                            dtype=np.float64,
                        ),
                        "rotation_y": float(parts[14]),
                    }
                )
        return labels

    # ------------------------------------------------------------------ #
    # Camera frame -> LiDAR frame conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cam_to_lidar(
        loc_cam: np.ndarray,
        dims_lwh: np.ndarray,
        rotation_y: float,
        calib: KITTICalibration,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Convert a KITTI label's 3D box from rectified camera frame to LiDAR frame.

        Forward chain:  lidar -- Tr_velo_to_cam --> cam0 -- R0_rect --> rect_cam
        Inverse:        rect_cam -- R0_rect^T --> cam0 -- (R^T, -t) --> lidar

        Yaw conversion: KITTI rotation_y is around the camera's y-axis (down).
        LiDAR yaw is around the z-axis (up). The standard formula used by
        OpenPCDet, mmdet3d, and Open3D-ML is:
            yaw_lidar = -rotation_y - pi/2

        Dimensions don't change numerically when switching frames - only the
        axis labels do (height stays height, etc.).
        """
        # rectified cam -> unrectified cam0
        loc_unrect = calib.R0_rect.T @ loc_cam

        # unrectified cam0 -> lidar (inverse of rigid transform)
        R = calib.Tr_velo_to_cam[:3, :3]
        t = calib.Tr_velo_to_cam[:3, 3]
        loc_lidar = R.T @ (loc_unrect - t)

        yaw_lidar = -rotation_y - np.pi / 2.0
        return loc_lidar, dims_lwh, float(yaw_lidar)
