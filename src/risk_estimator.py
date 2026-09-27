"""
risk_estimator.py — Causal, real-time accident anticipation risk estimator (Part B).
Uses Time-to-Collision (TTC), relative closing dynamics, and trajectory convergence.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.tracker import RoadTracker, Detection
from src.scene import SceneGeometry

HORIZON_SEC = 5.0
BASE_DIR = Path(__file__).resolve().parent.parent


class RiskEstimator:
    """Part B — causal accident anticipation.
    Step sees frames one by one in chronological order and returns P(accident in next 5s) in [0, 1].
    Strictly causal: uses only past and current frames.
    """
    def __init__(self, weights_path: str = "yolov8n.pt", stride: int = 5):
        p1 = BASE_DIR / "weights" / weights_path
        p2 = BASE_DIR / weights_path
        self.weights_path = str(p1 if p1.exists() else (p2 if p2.exists() else weights_path))
        self.stride = stride
        self.model: YOLO | None = None
        self.tracker: RoadTracker | None = None
        self.scene: SceneGeometry | None = None
        self.frame_idx = 0
        self.last_score = 0.0
        self.meta: dict = {}

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video."""
        self.meta = meta
        if self.model is None:
            self.model = YOLO(self.weights_path)
        self.tracker = RoadTracker(max_misses=4, iou_thresh=0.25)
        self.scene = SceneGeometry(meta.get("width", 3840), meta.get("height", 2160))
        self.frame_idx = 0
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Frame: BGR uint8 (H, W, 3). Return P(accident within 5 s) in [0, 1]."""
        self.frame_idx += 1

        # Skip frames to stay comfortably inside time budget (stride=5 gives ~6 fps)
        if (self.frame_idx - 1) % self.stride != 0:
            return float(self.last_score)

        if self.model is None or self.tracker is None:
            return 0.0

        # Downsample frame for fast detection (640 width)
        h, w = frame.shape[:2]
        inf_w = 640
        inf_h = int(h * (inf_w / w))
        small = cv2.resize(frame, (inf_w, inf_h))
        scale_x = w / float(inf_w)
        scale_y = h / float(inf_h)

        results = self.model(small, conf=0.25, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            name = self.model.names[cls_id]
            if name in ["car", "truck", "bus", "motorcycle", "person"]:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                orig_bbox = (bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y)
                detections.append(Detection(orig_bbox, name, float(box.conf[0].item())))

        active_tracks = self.tracker.update(detections, t_sec)

        # Compute Pairwise TTC & Closing Dynamics
        max_pair_risk = 0.0
        n = len(active_tracks)

        for i in range(n):
            tr1 = active_tracks[i]
            c1 = tr1.centroid
            for j in range(i + 1, n):
                tr2 = active_tracks[j]
                c2 = tr2.centroid

                # Part B evaluates vehicle accidents (two motorized vehicles)
                if tr1.cls_name not in ["car", "truck", "bus"] or tr2.cls_name not in ["car", "truck", "bus"]:
                    continue

                # Ignore nascent tracks with velocity jitter (require established tracks)
                if len(tr1.history) < 4 or len(tr2.history) < 4:
                    continue

                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                dist = math.hypot(dx, dy)

                if dist < 5.0 or dist > 250.0:
                    continue

                # Relative velocity vector
                dvx = tr1.vx - tr2.vx
                dvy = tr1.vy - tr2.vy

                # Closing velocity (projection of relative velocity onto connecting vector)
                v_closing = (dx * dvx + dy * dvy) / dist

                # EXCLUSION 1: Same-lane following / red-light queuing
                # Both vehicles aligned in same lane corridor (|x1 - x2| < 75 px), heading downstream (vy >= -2)
                is_same_lane_traffic = (
                    abs(dx) < 75.0
                    and tr1.vy >= -2.0 and tr2.vy >= -2.0
                )

                # EXCLUSION 2: Parallel traffic in adjacent lanes
                cos_heading = (tr1.vx * tr2.vx + tr1.vy * tr2.vy) / (tr1.speed * tr2.speed + 1e-6)
                is_parallel_traffic = (cos_heading > 0.70 and abs(dx) > 60.0)

                # In same-lane traffic, only trigger if severe high-speed closure at point-blank range (< 40 px)
                if is_same_lane_traffic and (v_closing > -80.0 or dist > 45.0):
                    continue

                if is_parallel_traffic:
                    continue

                # TRUE COLLISION TRAJECTORY:
                # High closing speed (> 70 px/s) on intersecting or opposing trajectory at close distance (< 85 px)
                if v_closing < -70.0 and dist < 85.0:
                    closing_speed = abs(v_closing)
                    ttc = dist / closing_speed

                    if ttc <= 1.5:
                        # Calibrated exponential risk function
                        base_risk = math.exp(-ttc / 0.60)
                        speed_factor = min(1.0, closing_speed / 90.0)
                        proximity_factor = min(1.0, 50.0 / (dist + 1e-3))
                        risk_val = float(np.clip(base_risk * speed_factor * proximity_factor, 0.0, 1.0))
                        max_pair_risk = max(max_pair_risk, risk_val)



        # Causal temporal exponential moving average (suppresses single-frame blips)
        alpha = 0.40
        self.last_score = float(np.clip(alpha * max_pair_risk + (1.0 - alpha) * self.last_score, 0.0, 1.0))
        if self.last_score < 0.02:
            self.last_score = 0.0

        return self.last_score
