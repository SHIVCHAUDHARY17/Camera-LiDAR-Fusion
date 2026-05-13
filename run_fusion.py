"""
Camera-LiDAR Fusion - command-line entry point.

Runs the full pipeline on one KITTI scene and saves up to four outputs:
  <scene>_inputs.jpg         raw camera + LiDAR detections
  <scene>_fused.jpg          fused output colour-coded by source
  <scene>_lidar_on_image.jpg LiDAR points projected onto image (if velodyne present)
  <scene>_bev.jpg            bird's eye view with point cloud + detection boxes

Usage:
    python run_fusion.py --scene 000000
    python run_fusion.py --scene 000050 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.camera_detector import CameraDetector
from src.calibration import KITTICalibration
from src.fusion import Fusion
from src.lidar_detector import StubLidarDetector
from src import visualizer


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera-LiDAR late fusion on one KITTI scene"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="6-digit zero-padded scene id, e.g. 000000",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kitti_root = Path(cfg["kitti"]["root"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load image
    # ------------------------------------------------------------------ #
    image_path = kitti_root / "image_2" / f"{args.scene}.png"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            f"Has data_object_image_2.zip been extracted?"
        )
    image = cv2.imread(str(image_path))
    print(f"[load]   {image_path.name}  shape={image.shape}")

    # ------------------------------------------------------------------ #
    # 2. Camera detector (YOLO)
    # ------------------------------------------------------------------ #
    cam_cfg = cfg["camera_detector"]
    camera = CameraDetector(
        weights=cam_cfg["weights"],
        confidence=cam_cfg["confidence"],
        classes=cam_cfg["classes"],
    )
    camera_dets = camera.detect(image)
    print(f"[camera] {len(camera_dets):>2d} detections")

    # ------------------------------------------------------------------ #
    # 3. LiDAR detector (stub)
    # ------------------------------------------------------------------ #
    lid_cfg = cfg["lidar_detector"]
    lidar = StubLidarDetector(
        kitti_root=str(kitti_root),
        dropout=lid_cfg["dropout"],
        bbox_jitter_std=lid_cfg["bbox_jitter_std"],
        confidence_alpha=lid_cfg["confidence_alpha"],
        confidence_beta=lid_cfg["confidence_beta"],
        score_threshold=lid_cfg["score_threshold"],
        seed=lid_cfg["seed"],
    )
    lidar_dets = lidar.detect(args.scene)
    print(f"[lidar]  {len(lidar_dets):>2d} detections")

    # ------------------------------------------------------------------ #
    # 4. Fusion
    # ------------------------------------------------------------------ #
    fus_cfg = cfg["fusion"]
    fuser = Fusion(
        iou_threshold=fus_cfg["iou_threshold"],
        camera_weight=fus_cfg["camera_weight"],
        lidar_weight=fus_cfg["lidar_weight"],
        min_confidence=fus_cfg["min_confidence"],
    )
    fused = fuser.fuse(camera_dets, lidar_dets)
    n_fused = sum(1 for f in fused if f.source == "fused")
    n_cam_only = sum(1 for f in fused if f.source == "camera_only")
    n_lid_only = sum(1 for f in fused if f.source == "lidar_only")
    print(
        f"[fusion] {len(fused):>2d} total  "
        f"({n_fused} fused, {n_cam_only} camera-only, {n_lid_only} lidar-only)"
    )

    # ------------------------------------------------------------------ #
    # 5. Standard image visualisations
    # ------------------------------------------------------------------ #
    cv2.imwrite(
        str(out_dir / f"{args.scene}_inputs.jpg"),
        visualizer.render_inputs(image, camera_dets, lidar_dets),
    )
    cv2.imwrite(
        str(out_dir / f"{args.scene}_fused.jpg"),
        visualizer.render_fused(image, fused),
    )
    print(f"[save]   outputs/{args.scene}_inputs.jpg")
    print(f"[save]   outputs/{args.scene}_fused.jpg")

    # ------------------------------------------------------------------ #
    # 6. Point cloud visualisations (requires velodyne data)
    # ------------------------------------------------------------------ #
    velodyne_path = kitti_root / "velodyne" / f"{args.scene}.bin"
    calib = KITTICalibration.from_file(kitti_root / "calib" / f"{args.scene}.txt")

    if velodyne_path.is_file():
        from src.point_cloud import read_velodyne_bin, filter_fov

        points = read_velodyne_bin(velodyne_path)
        points = filter_fov(points)
        print(f"[velodyne] {len(points):,} points after FOV filter")

        # A: LiDAR points projected onto camera image
        cv2.imwrite(
            str(out_dir / f"{args.scene}_lidar_on_image.jpg"),
            visualizer.render_lidar_on_image(image, points, calib),
        )
        print(f"[save]   outputs/{args.scene}_lidar_on_image.jpg")

        # B: Bird's eye view with point cloud + detections
        cv2.imwrite(
            str(out_dir / f"{args.scene}_bev.jpg"),
            visualizer.render_bev(points, fused),
        )
        print(f"[save]   outputs/{args.scene}_bev.jpg")

    else:
        print(f"[velodyne] not found — rendering BEV without point cloud")
        # B: BEV with detection boxes only (no point cloud background)
        empty = np.zeros((0, 4), dtype=np.float32)
        cv2.imwrite(
            str(out_dir / f"{args.scene}_bev.jpg"),
            visualizer.render_bev(empty, fused),
        )
        print(f"[save]   outputs/{args.scene}_bev.jpg")


if __name__ == "__main__":
    main()
