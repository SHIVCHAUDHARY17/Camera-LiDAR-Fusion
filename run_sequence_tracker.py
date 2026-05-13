"""
Track objects across a KITTI tracking sequence using real consecutive frames.

Images are at 10 Hz so the Kalman filter produces real velocity estimates.
Per-frame outputs are saved to outputs/sequence_<SEQUENCE>/:
    <frame>_camera_tracked.jpg  camera image with projected 3D tracks
    <frame>_bev.jpg             bird's eye view with Kalman tracks
    <frame>_combined.jpg        camera on top, BEV on bottom

Prints a full summary including MOTA (Multiple Object Tracking Accuracy).

Usage:
    python run_sequence_tracker.py --sequence 0001
    python run_sequence_tracker.py --sequence 0001 --start_frame 0 --end_frame 50 --video
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.camera_detector import CameraDetector
from src.fusion import Fusion
from src.kitti_seq_loader import (
    KITTI_CLASS_MAP,
    StubSequenceLidarDetector,
    get_sequence_frames,
    load_sequence_calib,
    load_sequence_labels,
)
from src.tracker import MultiObjectTracker
from src import visualizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """2D IoU between [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def compute_mota(
    gt_by_frame: dict,
    pred_by_frame: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute MOTA = 1 - (FP + FN + IDS) / total_GT over all frames.

    gt_by_frame:   {frame_id: [label dicts with 'track_id' and 'bbox_2d']}
    pred_by_frame: {frame_id: [dicts with 'track_id' and 'bbox']}

    Matching is greedy, highest-IoU-first, with IoU >= iou_threshold.
    IDS is counted when the predicted track ID matched to a GT track ID
    changes between frames.
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_ids = 0
    gt_to_pred: dict[int, int] = {}  # gt_track_id -> last matched pred_track_id

    all_frames = sorted(set(list(gt_by_frame.keys())) | set(list(pred_by_frame.keys())))

    for frame_id in all_frames:
        raw_gt = gt_by_frame.get(frame_id, [])
        preds = pred_by_frame.get(frame_id, [])

        gt_valid = [g for g in raw_gt if g["type"] in KITTI_CLASS_MAP]
        n_gt = len(gt_valid)
        n_pred = len(preds)
        total_gt += n_gt

        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue

        # Build IoU matrix and match greedily (highest IoU first)
        iou_mat = np.array(
            [[_bbox_iou(g["bbox_2d"], p["bbox"]) for p in preds] for g in gt_valid],
            dtype=np.float64,
        )
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        matches: list[tuple[int, int]] = []

        pairs = sorted(
            ((iou_mat[gi, pi], gi, pi) for gi in range(n_gt) for pi in range(n_pred)),
            reverse=True,
        )
        for iou_val, gi, pi in pairs:
            if iou_val < iou_threshold:
                break
            if gi in matched_gt or pi in matched_pred:
                continue
            matches.append((gi, pi))
            matched_gt.add(gi)
            matched_pred.add(pi)

        total_fp += n_pred - len(matches)
        total_fn += n_gt - len(matches)

        for gi, pi in matches:
            gt_tid = gt_valid[gi]["track_id"]
            pred_tid = preds[pi]["track_id"]
            prev = gt_to_pred.get(gt_tid)
            if prev is not None and prev != pred_tid:
                total_ids += 1
            gt_to_pred[gt_tid] = pred_tid

    mota = 1.0 - (total_fp + total_fn + total_ids) / max(total_gt, 1)
    return {
        "mota": mota,
        "total_gt": total_gt,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_ids": total_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalman MOT on a KITTI tracking sequence (real 10 Hz frames)"
    )
    parser.add_argument("--sequence", type=str, default="0001")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=100)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--kitti_root",
        type=str,
        default="E:/kitti/tracking",
        help="Root of KITTI tracking dataset (contains training/)",
    )
    parser.add_argument(
        "--video", action="store_true", help="Compile combined frames into sequence.avi"
    )
    parser.add_argument(
        "--no_dropout",
        action="store_true",
        help="Set dropout=0 and score_threshold=0.1 on stub LiDAR detector, "
             "and lower fusion min_confidence to 0.1, to evaluate true Kalman MOTA",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracking_root = Path(args.kitti_root)
    out_dir = Path(cfg["output"]["dir"]) / f"sequence_{args.sequence}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = tracking_root / "training" / "image_02"
    calib_path = tracking_root / "training" / "calib" / f"{args.sequence}.txt"
    label_path = tracking_root / "training" / "label_02" / f"{args.sequence}.txt"

    all_frames = get_sequence_frames(image_dir, args.sequence)
    frames = [f for f in all_frames if args.start_frame <= int(f) <= args.end_frame]

    if not frames:
        print(f"No frames found for sequence {args.sequence} in range "
              f"[{args.start_frame}, {args.end_frame}]")
        return

    mode_tag = " [NO-DROPOUT]" if args.no_dropout else ""
    print(f"Sequence {args.sequence}: {len(frames)} frames "
          f"({frames[0]} → {frames[-1]}){mode_tag}\n")

    gt_labels = load_sequence_labels(label_path) if label_path.is_file() else {}
    calib = load_sequence_calib(calib_path)

    cam_cfg = cfg["camera_detector"]
    camera = CameraDetector(
        weights=cam_cfg["weights"],
        confidence=cam_cfg["confidence"],
        classes=cam_cfg["classes"],
    )

    lid_cfg = cfg["lidar_detector"]
    lidar = StubSequenceLidarDetector(
        kitti_tracking_root=str(tracking_root),
        sequence_id=args.sequence,
        dropout=0.0 if args.no_dropout else lid_cfg["dropout"],
        bbox_jitter_std=lid_cfg["bbox_jitter_std"],
        confidence_alpha=lid_cfg["confidence_alpha"],
        confidence_beta=lid_cfg["confidence_beta"],
        score_threshold=0.1 if args.no_dropout else lid_cfg["score_threshold"],
        seed=lid_cfg["seed"],
    )

    fus_cfg = cfg["fusion"]
    fuser = Fusion(
        iou_threshold=fus_cfg["iou_threshold"],
        camera_weight=fus_cfg["camera_weight"],
        lidar_weight=fus_cfg["lidar_weight"],
        min_confidence=0.1 if args.no_dropout else fus_cfg["min_confidence"],
    )

    tracker = MultiObjectTracker(max_age=5, min_hits=1, distance_threshold=3.0)

    combined_frames: list[np.ndarray] = []
    frames_processed = 0
    all_track_ids: set[int] = set()
    active_counts: list[int] = []
    pred_by_frame: dict[int, list] = {}

    for frame_str in frames:
        frame_id = int(frame_str)
        image_path = image_dir / args.sequence / f"{frame_str}.png"

        if not image_path.is_file():
            print(f"  [{frame_str}] image not found — skip")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [{frame_str}] failed to read image — skip")
            continue

        camera_dets = camera.detect(image)
        lidar_dets = lidar.detect(frame_id)
        fused = fuser.fuse(camera_dets, lidar_dets)
        active_tracks = tracker.update(fused)

        # BEV (tracking dataset has no velodyne bin files)
        points = np.zeros((0, 4), dtype=np.float32)
        bev = visualizer.render_bev_tracks(points, active_tracks)
        cv2.putText(
            bev,
            f"Seq {args.sequence}  Frame {frame_str}",
            (5, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (100, 100, 100),
            1,
        )
        cv2.imwrite(str(out_dir / f"{frame_str}_bev.jpg"), bev)

        # Camera: only project tracks with a fresh detection (time_since_update == 0)
        current_tracks = [t for t in active_tracks if t.time_since_update == 0]
        cam_tracked = visualizer.render_camera_tracks(image, current_tracks, calib)
        cv2.imwrite(str(out_dir / f"{frame_str}_camera_tracked.jpg"), cam_tracked)

        # Combined
        target_w = cam_tracked.shape[1]
        bev_resized = cv2.resize(
            bev, (target_w, int(bev.shape[0] * target_w / bev.shape[1]))
        )
        combined = np.vstack([cam_tracked, bev_resized])
        cv2.imwrite(str(out_dir / f"{frame_str}_combined.jpg"), combined)
        combined_frames.append(combined)

        # Collect predictions for MOTA: tracked objects + camera-only fused detections.
        # Camera-only detections have no center_3d so the tracker skips them, but they
        # are still valid predictions against GT and must be included to avoid phantom FN.
        pred_by_frame[frame_id] = [
            {"track_id": t.id, "bbox": t.last_detection.bbox}
            for t in active_tracks
        ] + [
            {"track_id": -1, "bbox": f.bbox}
            for f in fused if f.source == "camera_only"
        ]

        frames_processed += 1
        all_track_ids.update(t.id for t in active_tracks)
        active_counts.append(len(active_tracks))

        n_fused = sum(1 for f in fused if f.source == "fused")
        print(f"  [{frame_str}]  fused={n_fused:>2d}  active_tracks={len(active_tracks):>2d}")

    # Only evaluate MOTA over frames that were actually processed.
    # gt_labels covers the full sequence; without this filter, unprocessed
    # frames contribute pure FN and make the score artificially low.
    gt_for_mota = {f: v for f, v in gt_labels.items()
                   if args.start_frame <= f <= args.end_frame}
    mota = compute_mota(gt_for_mota, pred_by_frame)
    avg_active = sum(active_counts) / len(active_counts) if active_counts else 0.0
    max_active = max(active_counts) if active_counts else 0

    print(f"\nSaved {frames_processed} frames → {out_dir}/")
    print("\n── Tracking summary ────────────────────────────────")
    print(f"  Sequence            : {args.sequence}")
    print(f"  Mode                : {'no-dropout (dropout=0, thr=0.1, fus_min_conf=0.1)' if args.no_dropout else 'default (dropout=0.1, thr=0.3, fus_min_conf=0.2)'}")
    print(f"  Frames processed    : {frames_processed}")
    print(f"  Unique track IDs    : {len(all_track_ids)}")
    print(f"  Avg active / frame  : {avg_active:.1f}")
    print(f"  Max active (1 frame): {max_active}")
    print("\n── MOTA ────────────────────────────────────────────")
    print(f"  MOTA score          : {mota['mota']:.4f}")
    print(f"  Total GT objects    : {mota['total_gt']}")
    print(f"  False positives (FP): {mota['total_fp']}")
    print(f"  False negatives (FN): {mota['total_fn']}")
    print(f"  ID switches   (IDS) : {mota['total_ids']}")
    print("────────────────────────────────────────────────────")

    if args.video and combined_frames:
        video_path = out_dir / "sequence.avi"
        h, w = combined_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            10,  # real KITTI frame rate
            (w, h),
        )
        for frame in combined_frames:
            writer.write(frame)
        writer.release()
        print(f"Video → {video_path}")


if __name__ == "__main__":
    main()
