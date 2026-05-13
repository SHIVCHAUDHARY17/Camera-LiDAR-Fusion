"""
Multi-object tracking across a sequence of KITTI scenes.

Saves per-scene outputs to outputs/tracking/:
  <scene>_bev.jpg            BEV with point cloud, tracks, trails, velocity arrows
  <scene>_camera_tracked.jpg camera photo with projected 3D boxes and track IDs

Optionally compiles BEV frames into tracking.avi video.

Usage:
    python run_tracker.py --start 000000 --end 000030
    python run_tracker.py --start 000000 --end 000050 --video
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
from src.point_cloud import filter_fov, read_velodyne_bin
from src.tracker import MultiObjectTracker
from src import visualizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalman filter MOT across a KITTI scene sequence"
    )
    parser.add_argument("--start", type=str, default="000000")
    parser.add_argument("--end", type=str, default="000030")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--video", action="store_true", help="Compile BEV frames into tracking.avi"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    kitti_root = Path(cfg["kitti"]["root"])
    out_dir = Path(cfg["output"]["dir"]) / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [f"{i:06d}" for i in range(int(args.start), int(args.end) + 1)]
    print(f"Tracking {len(scenes)} scenes  ({args.start} → {args.end})\n")

    # ------------------------------------------------------------------ #
    # Initialise pipeline
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

    tracker = MultiObjectTracker(
        max_age=5,
        min_hits=1,
        distance_threshold=3.0,
    )

    bev_frames: list[np.ndarray] = []
    scenes_processed = 0
    all_track_ids: set[int] = set()
    active_counts: list[int] = []

    # ------------------------------------------------------------------ #
    # Process sequence
    # ------------------------------------------------------------------ #
    for scene_id in scenes:
        image_path = kitti_root / "image_2" / f"{scene_id}.png"
        velodyne_path = kitti_root / "velodyne" / f"{scene_id}.bin"
        calib_path = kitti_root / "calib" / f"{scene_id}.txt"

        if not image_path.is_file():
            print(f"  [{scene_id}] image not found — skip")
            continue

        # Detection + fusion
        image = cv2.imread(str(image_path))
        camera_dets = camera.detect(image)
        lidar_dets = lidar.detect(scene_id)
        fused = fuser.fuse(camera_dets, lidar_dets)

        # Tracking
        active_tracks = tracker.update(fused)

        # Point cloud
        if velodyne_path.is_file():
            points = filter_fov(read_velodyne_bin(velodyne_path))
        else:
            points = np.zeros((0, 4), dtype=np.float32)

        # Load calibration for 3D projection
        calib = KITTICalibration.from_file(calib_path)

        # ---- BEV with tracks + velocity arrows ---- #
        bev = visualizer.render_bev_tracks(points, active_tracks)
        cv2.putText(
            bev,
            f"Scene {scene_id}",
            (5, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (100, 100, 100),
            1,
        )
        cv2.imwrite(str(out_dir / f"{scene_id}_bev.jpg"), bev)
        bev_frames.append(bev)

        # ---- Camera image with projected 3D tracks ---- #
        # Coasting tracks (time_since_update > 0) carry a stale 3-D position from a
        # previous scene; projecting them onto this frame's image would place boxes at
        # wrong pixel locations because the scenes are not temporally consecutive.
        current_tracks = [t for t in active_tracks if t.time_since_update == 0]
        cam_tracked = visualizer.render_camera_tracks(image, current_tracks, calib)
        cv2.imwrite(str(out_dir / f"{scene_id}_camera_tracked.jpg"), cam_tracked)

        # ---- Combined: camera_tracked on top, BEV on bottom ---- #
        target_w = cam_tracked.shape[1]
        bev_h_scaled = int(bev.shape[0] * target_w / bev.shape[1])
        bev_resized = cv2.resize(bev, (target_w, bev_h_scaled))
        combined = np.vstack([cam_tracked, bev_resized])
        cv2.imwrite(str(out_dir / f"{scene_id}_combined.jpg"), combined)

        # Accumulate summary stats
        scenes_processed += 1
        all_track_ids.update(t.id for t in active_tracks)
        active_counts.append(len(active_tracks))

        n_fused = sum(1 for f in fused if f.source == "fused")
        print(
            f"  [{scene_id}]  fused={n_fused:>2d}  "
            f"active_tracks={len(active_tracks):>2d}"
        )

    print(f"\nSaved {len(bev_frames)} frames → {out_dir}/")

    # ------------------------------------------------------------------ #
    # End-of-run summary
    # ------------------------------------------------------------------ #
    avg_active = sum(active_counts) / len(active_counts) if active_counts else 0.0
    max_active = max(active_counts) if active_counts else 0
    print("\n── Tracking summary ──────────────────────────")
    print(f"  Scenes processed    : {scenes_processed}")
    print(f"  Unique track IDs    : {len(all_track_ids)}")
    print(f"  Avg active / frame  : {avg_active:.1f}")
    print(f"  Max active (1 frame): {max_active}")
    print("──────────────────────────────────────────────")

    # Optional video
    if args.video and bev_frames:
        video_path = out_dir / "tracking.avi"
        h, w = bev_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            5,
            (w, h),
        )
        for frame in bev_frames:
            writer.write(frame)
        writer.release()
        print(f"Video → {video_path}")


if __name__ == "__main__":
    main()
