"""
Visualization helpers for fusion outputs.

Render modes:
  - render_inputs:          raw camera + LiDAR detections
  - render_fused:           fused output colour-coded by source
  - render_lidar_on_image:  Velodyne point cloud projected onto camera image
  - render_bev:             bird's eye view — detections only
  - render_bev_tracks:      BEV — Kalman tracks with trails + velocity arrows
  - render_camera_tracks:   camera image — 3D boxes + track IDs projected on photo
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.detections import Detection2D
from src.fusion import FusedDetection


# BGR colours
COLOR_CAMERA = (255, 120, 0)
COLOR_LIDAR = (0, 220, 100)
COLOR_FUSED = (0, 0, 255)
COLOR_3D_BOX = (0, 255, 255)
COLOR_TEXT_FG = (0, 0, 0)


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #


def _put_label(image, text, anchor_xy, bg_color):
    x, y = anchor_xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_top = max(0, y - th - 6)
    cv2.rectangle(image, (x, y_top), (x + tw + 4, y_top + th + 6), bg_color, -1)
    cv2.putText(
        image,
        text,
        (x + 2, y_top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_TEXT_FG,
        1,
        cv2.LINE_AA,
    )


def _draw_box(image, bbox, label, color, thickness=2):
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    _put_label(image, label, (x1, y1), color)


def _draw_3d_wireframe(image, corners_2d, color, thickness=1):
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    pts = corners_2d.astype(int)
    h, w = image.shape[:2]
    for i, j in edges:
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        p2 = (int(pts[j, 0]), int(pts[j, 1]))
        if not (-2000 <= p1[0] <= w + 2000 and -2000 <= p2[0] <= w + 2000):
            continue
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def _get_3d_info(det):
    c = getattr(det, "center_3d", None)
    d = getattr(det, "dimensions_3d", None)
    y = getattr(det, "yaw", None)
    if c is not None:
        return c, d, y
    lidar_det = getattr(det, "lidar_det", None)
    if lidar_det is not None:
        return (
            getattr(lidar_det, "center_3d", None),
            getattr(lidar_det, "dimensions_3d", None),
            getattr(lidar_det, "yaw", None),
        )
    return None, None, None


def _bev_box_corners(center_xy, lw, yaw):
    l, w = float(lw[0]), float(lw[1])
    x = np.array([l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[c, -s], [s, c]]) @ np.stack([x, y])
    corners[0] += center_xy[0]
    corners[1] += center_xy[1]
    return corners.T


def _bev_base(points_lidar, x_range, y_range, scale):
    """Shared BEV canvas: point cloud dots + range grid."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    bev_h = int((x_max - x_min) * scale)
    bev_w = int((y_max - y_min) * scale)
    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    def to_bev(lx, ly):
        return int((y_max - ly) * scale), int((x_max - lx) * scale)

    if points_lidar.shape[0] > 0:
        pts = points_lidar[::5, :3]
        mask = (
            (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
        )
        for pt in pts[mask]:
            col, row = to_bev(pt[0], pt[1])
            if 0 <= col < bev_w and 0 <= row < bev_h:
                cv2.circle(bev, (col, row), 1, (55, 55, 55), -1)

    for dist in range(10, int(x_max), 10):
        _, row_g = to_bev(dist, 0)
        cv2.line(bev, (0, row_g), (bev_w, row_g), (35, 35, 35), 1)
        cv2.putText(
            bev,
            f"{dist}m",
            (4, row_g - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (80, 80, 80),
            1,
        )

    return bev, to_bev, bev_h, bev_w


def _draw_ego(bev, to_bev, bev_w):
    col_e, row_e = to_bev(0, 0)
    cv2.rectangle(
        bev, (col_e - 5, row_e - 10), (col_e + 5, row_e + 10), (255, 255, 255), -1
    )
    cv2.putText(
        bev,
        "EGO",
        (col_e + 8, row_e + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        bev,
        "^ FORWARD",
        (bev_w // 2 - 35, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (160, 160, 160),
        1,
    )


# ---------------------------------------------------------------------- #
# Image-plane renderers
# ---------------------------------------------------------------------- #


def render_inputs(image, camera_dets, lidar_dets):
    out = image.copy()
    for d in camera_dets:
        _draw_box(out, d.bbox, f"CAM {d.class_name} {d.confidence:.2f}", COLOR_CAMERA)
    for d in lidar_dets:
        _draw_box(out, d.bbox, f"LID {d.class_name} {d.confidence:.2f}", COLOR_LIDAR)
        if d.corners_2d is not None:
            _draw_3d_wireframe(out, d.corners_2d, COLOR_3D_BOX)
    return out


def render_fused(image, fused, show_3d_corners=True):
    out = image.copy()
    for f in fused:
        if f.source == "fused":
            color, tag = COLOR_FUSED, "FUSED"
        elif f.source == "camera_only":
            color, tag = COLOR_CAMERA, "CAM"
        else:
            color, tag = COLOR_LIDAR, "LID"
        label = f"{tag} {f.class_name} {f.confidence:.2f}"
        if f.iou > 0:
            label += f" iou={f.iou:.2f}"
        _draw_box(out, f.bbox, label, color)
        if (
            show_3d_corners
            and f.lidar_det is not None
            and f.lidar_det.corners_2d is not None
        ):
            _draw_3d_wireframe(out, f.lidar_det.corners_2d, COLOR_3D_BOX)
    return out


# ---------------------------------------------------------------------- #
# Feature A — LiDAR point cloud on camera image
# ---------------------------------------------------------------------- #


def render_lidar_on_image(image, points_lidar, calib, max_depth=50.0, dot_radius=2):
    out = image.copy()
    h, w = image.shape[:2]
    pixels, depths, mask = calib.lidar_to_image(points_lidar[:, :3], image_shape=(h, w))
    keep = mask & (depths > 0) & (depths <= max_depth)
    if not np.any(keep):
        return out
    pixels_f = pixels[keep]
    depths_f = depths[keep]
    norm = 1.0 - np.clip(depths_f / max_depth, 0.0, 1.0)
    colors = cv2.applyColorMap(
        (norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO
    ).reshape(-1, 3)
    order = np.argsort(depths_f)[::-1]
    pixels_f = pixels_f[order]
    colors = colors[order]
    u = np.clip(pixels_f[:, 0].astype(int), 0, w - 1)
    v = np.clip(pixels_f[:, 1].astype(int), 0, h - 1)
    for dy in range(-dot_radius, dot_radius + 1):
        for dx in range(-dot_radius, dot_radius + 1):
            if dx * dx + dy * dy <= dot_radius * dot_radius:
                out[np.clip(v + dy, 0, h - 1), np.clip(u + dx, 0, w - 1)] = colors
    return out


# ---------------------------------------------------------------------- #
# Feature B — Static BEV (no tracking)
# ---------------------------------------------------------------------- #


def render_bev(
    points_lidar, detections, x_range=(0.0, 50.0), y_range=(-25.0, 25.0), scale=10
):
    bev, to_bev, bev_h, bev_w = _bev_base(points_lidar, x_range, y_range, scale)
    for det in detections:
        source = getattr(det, "source", "lidar")
        class_name = getattr(det, "class_name", "?")
        color = (
            COLOR_FUSED
            if source == "fused"
            else COLOR_CAMERA
            if source == "camera_only"
            else COLOR_LIDAR
        )
        c3d, dims, yaw = _get_3d_info(det)
        if c3d is None or dims is None or yaw is None:
            continue
        corners = _bev_box_corners(c3d[:2], dims[:2], yaw)
        bev_pts = np.array([to_bev(c[0], c[1]) for c in corners], dtype=np.int32)
        cv2.drawContours(bev, [bev_pts], 0, color, 2)
        col_c, row_c = to_bev(c3d[0], c3d[1])
        cv2.circle(bev, (col_c, row_c), 3, color, -1)
        if 0 <= col_c < bev_w and 2 <= row_c < bev_h:
            cv2.putText(
                bev,
                class_name[0],
                (col_c + 5, row_c),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    _draw_ego(bev, to_bev, bev_w)
    return bev


# ---------------------------------------------------------------------- #
# Feature C — BEV with Kalman tracks + velocity arrows (B updated)
# ---------------------------------------------------------------------- #


def render_bev_tracks(
    points_lidar,
    tracks,
    trail_length=15,
    x_range=(0.0, 50.0),
    y_range=(-25.0, 25.0),
    scale=10,
    predict_seconds=3.0,
):
    """
    BEV with Kalman-tracked objects.

    Each track: unique colour, fading trail, velocity arrow showing
    predicted position in predict_seconds, ID + speed label.
    Coasting tracks marked with *.
    """
    bev, to_bev, bev_h, bev_w = _bev_base(points_lidar, x_range, y_range, scale)

    for track in tracks:
        color = track.color
        coasting = track.time_since_update > 0

        # Fading trail
        trail = track.history[-trail_length:]
        if len(trail) >= 2:
            pts_bev = [to_bev(p[0], p[1]) for p in trail]
            for i in range(1, len(pts_bev)):
                alpha = (i / len(pts_bev)) * 0.8
                fade = tuple(int(c * alpha) for c in color)
                cv2.line(bev, pts_bev[i - 1], pts_bev[i], fade, 1, cv2.LINE_AA)

        # Detection box
        c3d, dims, yaw = _get_3d_info(track.last_detection)
        if c3d is not None and dims is not None and yaw is not None:
            corners = _bev_box_corners(c3d[:2], dims[:2], yaw)
            bev_pts = np.array([to_bev(c[0], c[1]) for c in corners], dtype=np.int32)
            cv2.drawContours(bev, [bev_pts], 0, color, 1 if coasting else 2)

        # Current position
        cx, cy = track.position
        col_c, row_c = to_bev(cx, cy)
        cv2.circle(bev, (col_c, row_c), 4, color, -1)

        # Velocity arrow — predicted position in predict_seconds
        vx, vy = track.velocity
        speed = track.speed
        if speed > 0.3:  # only draw arrow if actually moving
            pred_x = cx + vx * predict_seconds
            pred_y = cy + vy * predict_seconds
            col_a, row_a = to_bev(pred_x, pred_y)
            x_min, x_max = x_range
            y_min, y_max = y_range
            if 0 <= col_a < bev_w and 0 <= row_a < bev_h:
                cv2.arrowedLine(
                    bev,
                    (col_c, row_c),
                    (col_a, row_a),
                    color,
                    2,
                    tipLength=0.3,
                    line_type=cv2.LINE_AA,
                )

        # Label
        label = f"ID{track.id} {track.class_name[0]} {speed:.1f}m/s"
        if coasting:
            label += "*"
        if 0 <= col_c < bev_w - 60 and 5 <= row_c < bev_h:
            cv2.putText(
                bev,
                label,
                (col_c + 6, row_c - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

    _draw_ego(bev, to_bev, bev_w)
    cv2.putText(
        bev,
        f"{len(tracks)} active tracks",
        (bev_w - 90, bev_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (140, 140, 140),
        1,
    )
    return bev


# ---------------------------------------------------------------------- #
# Feature D — Camera image with projected 3D tracks
# ---------------------------------------------------------------------- #


def render_camera_tracks(image, tracks, calib):
    """
    Project Kalman-tracked 3D boxes onto the camera image.

    For each active track: coloured 3D wireframe box with track ID,
    class name, and velocity estimate overlaid on the real photo.

    This is the standard autonomous driving perception visualisation —
    3D understanding projected back into image space.

    Args:
        image:  HxWx3 BGR camera image
        tracks: confirmed KalmanTracker objects
        calib:  KITTICalibration for this scene
    """
    out = image.copy()
    h, w = out.shape[:2]

    for track in tracks:
        color = track.color
        c3d, dims, yaw = _get_3d_info(track.last_detection)
        if c3d is None or dims is None or yaw is None:
            continue

        # Project 3D box onto image
        corners_2d, bbox_2d = calib.project_box_3d_to_image(
            c3d, dims, yaw, image_shape=(h, w)
        )

        # Draw 3D wireframe
        _draw_3d_wireframe(out, corners_2d, color, thickness=2)

        # Draw 2D bbox outline (thin)
        x1, y1, x2, y2 = bbox_2d.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        # Ground contact point — project bottom-centre of 3D box (c3d is already
        # bottom-centre in LiDAR frame) onto the image and mark with a filled circle.
        gc_pixels, gc_depths, gc_mask = calib.lidar_to_image(
            np.array([c3d], dtype=np.float32), image_shape=(h, w)
        )
        if gc_mask[0] and gc_depths[0] > 0:
            gx, gy = int(gc_pixels[0, 0]), int(gc_pixels[0, 1])
            cv2.circle(out, (gx, gy), 5, color, -1)

        # Label: ID, class, speed
        label = f"ID{track.id} {track.class_name} {track.speed:.1f}m/s"
        if track.time_since_update > 0:
            label += "*"
        _put_label(out, label, (x1, y1), color)

    return out
