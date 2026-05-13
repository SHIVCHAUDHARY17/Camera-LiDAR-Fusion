"""
KITTI tracking sequence loader and stub LiDAR detector.

Mirrors the detection-dataset StubLidarDetector (src/lidar_detector.py) but
works on the tracking dataset layout where all frames in a sequence share one
label file and one calibration file, and images live under image_02/<seq>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.calibration import KITTICalibration
from src.detections import Detection2D

KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}
CLASS_ID_TO_NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def load_sequence_labels(label_path: Union[str, Path]) -> Dict[int, List[dict]]:
    """
    Parse a KITTI tracking label file.

    Tracking label format (space-separated, 17 fields):
        frame track_id type truncated occluded alpha
        left top right bottom        [pixels, image_2]
        height width length          [metres]
        x y z                        [metres, rectified camera frame, bottom-centre]
        rotation_y                   [radians, camera y-axis]

    Returns {frame_id: [list of label dicts]}.
    Each label dict keys: track_id, type, bbox_2d, dims_lwh, loc_cam, rotation_y.
    dims_lwh stores (length, width, height) matching LiDAR convention.
    """
    label_path = Path(label_path)
    labels: Dict[int, List[dict]] = {}

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue

            frame_id = int(parts[0])
            # KITTI tracking dimension order: height=10, width=11, length=12
            dims_lwh = np.array(
                [float(parts[12]), float(parts[11]), float(parts[10])],
                dtype=np.float64,
            )
            lbl = {
                "track_id": int(parts[1]),
                "type": parts[2],
                "bbox_2d": np.array(
                    [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])],
                    dtype=np.float64,
                ),
                "dims_lwh": dims_lwh,
                "loc_cam": np.array(
                    [float(parts[13]), float(parts[14]), float(parts[15])],
                    dtype=np.float64,
                ),
                "rotation_y": float(parts[16]),
            }
            labels.setdefault(frame_id, []).append(lbl)

    return labels


def load_sequence_calib(calib_path: Union[str, Path]) -> KITTICalibration:
    """
    Load KITTI tracking calibration, normalising variant key names.

    Accepts R_rect or R0_rect, and Tr_velo_to_cam or Tr_velo_cam.
    Returns a KITTICalibration ready for projection.
    """
    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    data: dict = {}
    with calib_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support both "key: v1 v2 ..." (detection) and "key v1 v2 ..." (tracking)
            if ":" in line:
                key, value = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, value = parts[0], parts[1]
            try:
                data[key.strip()] = np.array(
                    [float(x) for x in value.split()], dtype=np.float64
                )
            except ValueError:
                continue

    if "P2" not in data:
        raise ValueError(f"Calibration file {calib_path} missing P2")

    r0 = data.get("R0_rect") or data.get("R_rect")
    if r0 is None:
        raise ValueError(f"Calibration file {calib_path} missing R0_rect / R_rect")

    tr = data.get("Tr_velo_to_cam") or data.get("Tr_velo_cam")
    if tr is None:
        raise ValueError(
            f"Calibration file {calib_path} missing Tr_velo_to_cam / Tr_velo_cam"
        )

    return KITTICalibration(P2=data["P2"], R0_rect=r0, Tr_velo_to_cam=tr)


def get_sequence_frames(image_dir: Union[str, Path], sequence_id: str) -> List[str]:
    """
    Return sorted list of zero-padded frame ID strings for a sequence.

    Scans image_dir/sequence_id/*.png and returns the file stems in order.
    """
    seq_dir = Path(image_dir) / sequence_id
    if not seq_dir.is_dir():
        return []
    return sorted(p.stem for p in seq_dir.glob("*.png"))


def _cam_to_lidar(
    loc_cam: np.ndarray,
    dims_lwh: np.ndarray,
    rotation_y: float,
    calib: KITTICalibration,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a KITTI 3D box from rectified camera frame to LiDAR frame.

    Mirrors StubLidarDetector._cam_to_lidar exactly so noise statistics
    are identical between detection and tracking stubs.
    """
    loc_unrect = calib.R0_rect.T @ loc_cam
    R = calib.Tr_velo_to_cam[:3, :3]
    t = calib.Tr_velo_to_cam[:3, 3]
    loc_lidar = R.T @ (loc_unrect - t)
    yaw_lidar = -rotation_y - np.pi / 2.0
    return loc_lidar, dims_lwh, float(yaw_lidar)


class StubSequenceLidarDetector:
    """
    Stub LiDAR detector for KITTI tracking sequences.

    Reads the sequence's GT label file once at construction, then serves
    per-frame detections with the same noise model as StubLidarDetector:
      - 10 % object dropout
      - Gaussian bbox jitter (std = 3 px per corner)
      - Beta(8, 2) confidence sampling, threshold at 0.3
      - Deterministic RNG with seed = 42

    Expected dataset layout under kitti_tracking_root/training/:
        label_02/<sequence_id>.txt
        calib/<sequence_id>.txt
    """

    def __init__(
        self,
        kitti_tracking_root: Union[str, Path],
        sequence_id: str,
        dropout: float = 0.10,
        bbox_jitter_std: float = 3.0,
        confidence_alpha: float = 8.0,
        confidence_beta: float = 2.0,
        score_threshold: float = 0.3,
        seed: Optional[int] = 42,
    ) -> None:
        self.root = Path(kitti_tracking_root)
        self.sequence_id = sequence_id
        self.dropout = float(dropout)
        self.bbox_jitter_std = float(bbox_jitter_std)
        self.confidence_alpha = float(confidence_alpha)
        self.confidence_beta = float(confidence_beta)
        self.score_threshold = float(score_threshold)
        self.rng = np.random.default_rng(seed)

        label_path = self.root / "training" / "label_02" / f"{sequence_id}.txt"
        calib_path = self.root / "training" / "calib" / f"{sequence_id}.txt"

        self._labels = load_sequence_labels(label_path)
        self._calib = load_sequence_calib(calib_path)

    def detect(self, frame_id: Union[int, str]) -> List[Detection2D]:
        """
        Return simulated LiDAR detections for one frame.

        Args:
            frame_id: integer index or zero-padded string (e.g. 0 or "000000").
        """
        fid = int(frame_id)
        frame_labels = self._labels.get(fid, [])

        detections: List[Detection2D] = []
        for lbl in frame_labels:
            cls_name = lbl["type"]
            if cls_name not in KITTI_CLASS_MAP:
                continue

            if self.rng.random() < self.dropout:
                continue

            conf = float(self.rng.beta(self.confidence_alpha, self.confidence_beta))
            if conf < self.score_threshold:
                continue

            bbox = lbl["bbox_2d"] + self.rng.normal(0.0, self.bbox_jitter_std, size=4)
            x1, y1, x2, y2 = bbox
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            center_lidar, dims_lwh, yaw_lidar = _cam_to_lidar(
                lbl["loc_cam"], lbl["dims_lwh"], lbl["rotation_y"], self._calib
            )
            corners_2d, _ = self._calib.project_box_3d_to_image(
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
