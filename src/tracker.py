"""
tracker.py — High-performance multi-object tracker with velocity, heading, and trajectory dynamics.
Includes class-agnostic NMS to eliminate duplicate detections.
"""
from __future__ import annotations
import math
import numpy as np
from typing import NamedTuple

class Detection(NamedTuple):
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls_name: str
    conf: float

def iou(boxA: tuple[float, float, float, float], boxB: tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

class Track:
    def __init__(self, track_id: int, det: Detection, t_sec: float):
        self.track_id = track_id
        self.cls_name = det.cls_name
        self.conf = det.conf
        self.history: list[tuple[float, tuple[float, float, float, float]]] = [(t_sec, det.bbox)]
        self.last_t = t_sec
        self.last_bbox = det.bbox
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.accel = 0.0
        self.misses = 0
        self.hit_count = 1
        self.is_stationary = False
        self.stationary_since: float | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.last_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.last_bbox
        return ((x1 + x2) / 2.0, y2)

    def update(self, det: Detection, t_sec: float):
        dt = t_sec - self.last_t
        if dt > 1e-4:
            c_prev = self.centroid
            x1, y1, x2, y2 = det.bbox
            c_curr = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            raw_vx = (c_curr[0] - c_prev[0]) / dt
            raw_vy = (c_curr[1] - c_prev[1]) / dt
            
            # Exponential smoothing for velocity
            alpha = 0.6
            self.vx = alpha * raw_vx + (1 - alpha) * self.vx
            self.vy = alpha * raw_vy + (1 - alpha) * self.vy
            prev_speed = self.speed
            self.speed = math.hypot(self.vx, self.vy)
            self.accel = (self.speed - prev_speed) / dt

            # Stationary check
            if self.speed < 8.0:
                if self.stationary_since is None:
                    self.stationary_since = t_sec
                self.is_stationary = True
            else:
                self.is_stationary = False
                self.stationary_since = None

        self.last_t = t_sec
        self.last_bbox = det.bbox
        self.conf = det.conf
        self.history.append((t_sec, det.bbox))
        self.misses = 0
        self.hit_count += 1

    def stationary_duration(self, current_t: float) -> float:
        if self.is_stationary and self.stationary_since is not None:
            return max(0.0, current_t - self.stationary_since)
        return 0.0


class RoadTracker:
    """Multi-object tracker associating detections across frames with class-agnostic NMS."""
    def __init__(self, max_misses: int = 5, iou_thresh: float = 0.20):
        self.next_id = 1
        self.tracks: list[Track] = []
        self.dead_tracks: list[Track] = []
        self.max_misses = max_misses
        self.iou_thresh = iou_thresh

    def update(self, detections: list[Detection], t_sec: float) -> list[Track]:
        # 1. Class-Agnostic Non-Maximum Suppression to eliminate duplicate boxes
        dets = sorted(detections, key=lambda d: -d.conf)
        filtered_dets: list[Detection] = []
        for d in dets:
            keep = True
            for kept in filtered_dets:
                if iou(d.bbox, kept.bbox) > 0.45:
                    keep = False
                    break
            if keep:
                filtered_dets.append(d)
        detections = filtered_dets

        # 2. First frame initialization
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(self.next_id, det, t_sec))
                self.next_id += 1
            return self.tracks

        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        # 3. Match greedy descending IoU
        pairs = []
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                score = iou(track.last_bbox, det.bbox)
                if score >= self.iou_thresh:
                    pairs.append((score, t_idx, d_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)
        for score, t_idx, d_idx in pairs:
            if t_idx in unmatched_tracks and d_idx in unmatched_dets:
                matches.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_dets.remove(d_idx)

        # 4. Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx], t_sec)

        # 5. Handle unmatched tracks (coasting)
        active_tracks = []
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.misses += 1
            if track.misses <= self.max_misses:
                active_tracks.append(track)
            else:
                self.dead_tracks.append(track)

        for t_idx, _ in matches:
            active_tracks.append(self.tracks[t_idx])

        # 6. Handle new tracks
        for d_idx in unmatched_dets:
            new_track = Track(self.next_id, detections[d_idx], t_sec)
            self.next_id += 1
            active_tracks.append(new_track)

        self.tracks = active_tracks
        return self.tracks

    def get_all_tracks(self) -> list[Track]:
        return self.tracks + self.dead_tracks
