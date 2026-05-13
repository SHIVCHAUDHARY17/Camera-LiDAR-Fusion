"""
Multi-object tracker using a Kalman filter with constant-velocity model.

State:       [x, y, vx, vy] in LiDAR BEV frame (metres)
Measurement: [x, y] from fused/LiDAR detections (3D position required)

Track lifecycle:
  Birth  — unmatched detection creates a new track
  Update — matched detection corrects Kalman state
  Coast  — unmatched track: predict forward, no correction (up to max_age frames)
  Death  — coasted too long: remove track

Matching: Hungarian algorithm on Euclidean distance in BEV space.
Camera-only detections (no 3D info) are skipped — cannot localise in BEV.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic unique BGR colour for a track ID."""
    rng = np.random.default_rng(track_id * 17 + 7)
    r, g, b = rng.integers(80, 255, size=3)
    return (int(b), int(g), int(r))  # BGR


class KalmanTracker:
    """Single-object track with constant-velocity Kalman filter."""

    _id_counter: int = 1

    def __init__(self, detection, dt: float = 1.0) -> None:
        self.id = KalmanTracker._id_counter
        KalmanTracker._id_counter += 1

        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.last_detection = detection

        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        self.history: List[Tuple[float, float]] = []

        # ---- Kalman matrices ----------------------------------------- #
        # State transition: constant velocity
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix: observe [x, y] only
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise: velocity is less certain than position
        self.Q = np.diag([0.25, 0.25, 1.0, 1.0])

        # Measurement noise: ~1 m detector accuracy in LiDAR frame
        self.R = np.diag([1.0, 1.0])

        # Initial state: position from detection, velocity = 0
        pos = self._get_bev_position(detection)
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)

        # High initial velocity uncertainty
        self.P = np.diag([2.0, 2.0, 20.0, 20.0])

    # ------------------------------------------------------------------ #
    # Kalman predict + update
    # ------------------------------------------------------------------ #

    def predict(self) -> np.ndarray:
        """Advance state by one timestep without a measurement."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        self.history.append((float(self.x[0]), float(self.x[1])))
        return self.x[:2]

    def update(self, detection) -> None:
        """Correct state with a matched detection measurement."""
        pos = self._get_bev_position(detection)
        z = pos.reshape(2, 1)

        y = z - (self.H @ self.x).reshape(2, 1)  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.last_detection = detection
        self.history.append((float(self.x[0]), float(self.x[1])))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.x[2] ** 2 + self.x[3] ** 2))

    @property
    def color(self) -> Tuple[int, int, int]:
        return _track_color(self.id)

    # ------------------------------------------------------------------ #
    # Position extraction helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_bev_position(detection) -> np.ndarray:
        """
        Extract (x, y) in LiDAR BEV frame from a FusedDetection or Detection2D.
        Raises ValueError if no 3D info is available (camera-only detections).
        """
        # FusedDetection — 3D info is on lidar_det
        lidar_det = getattr(detection, "lidar_det", None)
        if lidar_det is not None:
            c = getattr(lidar_det, "center_3d", None)
            if c is not None:
                return np.array([c[0], c[1]], dtype=np.float64)

        # Detection2D — center_3d is directly on the object
        c = getattr(detection, "center_3d", None)
        if c is not None:
            return np.array([c[0], c[1]], dtype=np.float64)

        raise ValueError("Detection has no 3D position — cannot track in BEV")


# ---------------------------------------------------------------------- #
# Multi-object tracker
# ---------------------------------------------------------------------- #


class MultiObjectTracker:
    """
    Manages a set of KalmanTracker instances across frames.

    Each call to update() processes one frame:
    1. Predict all existing tracks forward by one timestep
    2. Build a distance matrix between detections and predicted positions
    3. Hungarian algorithm finds the optimal one-to-one assignment
    4. Update matched tracks; create new tracks for unmatched detections
    5. Remove tracks that have coasted for more than max_age frames
    6. Return confirmed tracks (hits >= min_hits)
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 1,
        distance_threshold: float = 3.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers: List[KalmanTracker] = []
        self.frame_count = 0

    def update(self, detections: Sequence) -> List[KalmanTracker]:
        """
        Update tracker with one frame of fused detections.

        Returns confirmed active tracks (hits >= min_hits).
        Camera-only detections (no center_3d) are silently skipped.
        """
        self.frame_count += 1

        # Keep only detections with 3D position (skip camera_only)
        trackable = []
        for d in detections:
            try:
                KalmanTracker._get_bev_position(d)
                trackable.append(d)
            except ValueError:
                continue

        # Step 1: predict all existing tracks
        predicted = [t.predict() for t in self.trackers]

        # Step 2–3: match and update
        matched, unmatched_dets, _ = self._match(trackable, predicted)

        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(trackable[det_idx])

        # Step 4: birth new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanTracker(trackable[det_idx]))

        # Step 5: kill old tracks
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        # Step 6: return confirmed tracks
        return [t for t in self.trackers if t.hits >= self.min_hits]

    def _match(
        self,
        detections: List,
        predicted: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to predicted track positions using
        Euclidean distance + Hungarian algorithm.
        """
        if not self.trackers or not detections:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        det_pos = np.array(
            [KalmanTracker._get_bev_position(d) for d in detections]
        )  # (D, 2)
        trk_pos = np.array(predicted)  # (T, 2)

        # (D, T) distance matrix
        diff = det_pos[:, None, :] - trk_pos[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))

        row_idx, col_idx = linear_sum_assignment(dist)

        matched, matched_dets, matched_trks = [], set(), set()
        for r, c in zip(row_idx, col_idx):
            if dist[r, c] <= self.distance_threshold:
                matched.append((r, c))
                matched_dets.add(r)
                matched_trks.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_trks = [i for i in range(len(self.trackers)) if i not in matched_trks]

        return matched, unmatched_dets, unmatched_trks
