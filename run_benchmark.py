"""
Benchmark camera-only, LiDAR-only, and fused detection across a val split.

Prints per-class AP and mAP for each sensor modality — the quantitative
result showing how fusion compares to individual sensors.

Usage:
    python run_benchmark.py                     # 100 scenes (default)
    python run_benchmark.py --max_scenes 50     # faster run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from src.camera_detector import CameraDetector
from src.evaluator import KITTIEvaluator, load_gt_boxes
from src.fusion import Fusion
from src.lidar_detector import StubLidarDetector


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_val_split(split_file: Path, kitti_root: Path, max_scenes: int) -> list[str]:
    if split_file.is_file():
        with open(split_file) as f:
            scenes = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(scenes)} scenes from {split_file}")
    else:
        image_dir = kitti_root / "image_2"
        scenes = sorted(p.stem for p in image_dir.glob("*.png"))
        print(f"No split file found — using first {max_scenes} available scenes.")
    return scenes[:max_scenes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark camera / LiDAR / fused AP on KITTI val split"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=100,
        help="Maximum number of scenes to evaluate (default: 100)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    kitti_root = Path(cfg["kitti"]["root"])

    scenes = load_val_split(
        Path(cfg["kitti"]["split_file"]), kitti_root, args.max_scenes
    )

    # ------------------------------------------------------------------ #
    # Initialise detectors and fusion
    # ------------------------------------------------------------------ #
    cam_cfg = cfg["camera_detector"]
    camera = CameraDetector(
        weights=cam_cfg["weights"],
        confidence=cam_cfg["confidence"],
        classes=cam_cfg["classes"],
    )

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

    fus_cfg = cfg["fusion"]
    fuser = Fusion(
        iou_threshold=fus_cfg["iou_threshold"],
        camera_weight=fus_cfg["camera_weight"],
        lidar_weight=fus_cfg["lidar_weight"],
        min_confidence=fus_cfg["min_confidence"],
    )

    evaluator = KITTIEvaluator(iou_threshold=0.5)

    # ------------------------------------------------------------------ #
    # Evaluate
    # ------------------------------------------------------------------ #
    processed = 0
    for i, scene_id in enumerate(scenes):
        image_path = kitti_root / "image_2" / f"{scene_id}.png"
        label_path = kitti_root / "label_2" / f"{scene_id}.txt"

        if not image_path.is_file() or not label_path.is_file():
            continue

        image = cv2.imread(str(image_path))
        gt_boxes = load_gt_boxes(label_path)

        camera_dets = camera.detect(image)
        lidar_dets = lidar.detect(scene_id)
        fused_dets = fuser.fuse(camera_dets, lidar_dets)

        evaluator.add_scene(gt_boxes, camera_dets, lidar_dets, fused_dets)
        processed += 1

        if processed % 10 == 0:
            print(f"  {processed}/{len(scenes)} scenes processed...")

    print(f"\nFinished: {processed} scenes evaluated.")
    evaluator.print_summary()


if __name__ == "__main__":
    main()
