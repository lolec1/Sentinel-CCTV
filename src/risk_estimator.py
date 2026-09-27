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
        self.track_bev_history: dict[int, tuple[float, float, float]] = {}

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video."""
        self.meta = meta
        if self.model is None:
            self.model = YOLO(self.weights_path)
        self.tracker = RoadTracker(max_misses=4, iou_thresh=0.25)
        self.scene = SceneGeometry(meta.get("width", 3840), meta.get("height", 2160))
        self.frame_idx = 0
        self.last_score = 0.0
        self.track_bev_history = {}

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

        current_bev_state = {}
        for trk in active_tracks:
            # Ground contact point: bottom-center of bounding box
            x1, y1, x2, y2 = trk.last_bbox
            bottom_center = ((x1 + x2) / 2.0, y2)
            
            # Map 2D image coordinates to BEV metric coordinates (in meters)
            x_bev, y_bev = self.scene.image_to_bev(bottom_center)

            vx_bev, vy_bev = 0.0, 0.0
            if trk.track_id in self.track_bev_history:
                prev_t, prev_x, prev_y = self.track_bev_history[trk.track_id]
                dt = t_sec - prev_t
                if dt > 0:
                    vx_bev = (x_bev - prev_x) / dt  # meters / second
                    vy_bev = (y_bev - prev_y) / dt  # meters / second

            current_bev_state[trk.track_id] = {
                "track": trk,
                "pos": np.array([x_bev, y_bev], dtype=np.float32),
                "vel": np.array([vx_bev, vy_bev], dtype=np.float32)
            }

        # Save history for velocity computation on next frame
        self.track_bev_history = {
            tid: (t_sec, state["pos"][0], state["pos"][1]) 
            for tid, state in current_bev_state.items()
        }

        # Compute Pairwise TTC & Closing Dynamics
        max_pair_risk = 0.0
        active_ids = list(current_bev_state.keys())
        n = len(active_ids)

        for i in range(n):
            id1 = active_ids[i]
            st1 = current_bev_state[id1]
            tr1 = st1["track"]

            for j in range(i + 1, n):
                id2 = active_ids[j]
                st2 = current_bev_state[id2]
                tr2 = st2["track"]

                # Part B evaluates vehicle accidents (two motorized vehicles)
                if tr1.cls_name not in ["car", "truck", "bus"] or tr2.cls_name not in ["car", "truck", "bus"]:
                    continue

                # Require established tracks to avoid velocity jitter
                if len(tr1.history) < 4 or len(tr2.history) < 4:
                    continue

                pos1, vel1 = st1["pos"], st1["vel"]
                pos2, vel2 = st2["pos"], st2["vel"]

                rel_pos = pos1 - pos2  # Distance vector in meters
                dist_m = float(np.linalg.norm(rel_pos))

                # Distance bounds in meters (e.g., 0.5m to 40.0m)
                if dist_m < 0.5 or dist_m > 40.0:
                    continue

                rel_vel = vel1 - vel2  # Relative velocity in m/s
                speed1 = float(np.linalg.norm(vel1))
                speed2 = float(np.linalg.norm(vel2))

                # Closing velocity along the distance vector (m/s)
                v_closing = -float(np.dot(rel_pos, rel_vel) / (dist_m + 1e-6))

                # EXCLUSION 1: Same-lane following / queuing
                # Distance along X axis < 2.5 meters (lane width ~3.5m)
                dx_m = abs(pos1[0] - pos2[0])
                is_same_lane_traffic = (dx_m < 2.5 and vel1[1] >= -0.5 and vel2[1] >= -0.5)

                # EXCLUSION 2: Parallel traffic in adjacent lanes
                cos_heading = float(np.dot(vel1, vel2) / (speed1 * speed2 + 1e-6))
                is_parallel_traffic = (cos_heading > 0.70 and dx_m > 2.0)

                # Ignore same-lane traffic unless extreme high-speed closing at point-blank range (< 3m)
                if is_same_lane_traffic and (v_closing < 8.0 or dist_m > 3.0):
                    continue

                if is_parallel_traffic:
                    continue

                # TRUE COLLISION TRAJECTORY:
                # Closing speed > 3.0 m/s (~11 km/h) at close proximity (< 12.0 meters)
                if v_closing > 3.0 and dist_m < 12.0:
                    ttc = dist_m / v_closing  # Time-To-Collision in seconds

                    if ttc <= HORIZON_SEC:
                        # Exponential risk curve based on metric TTC
                        base_risk = math.exp(-ttc / 1.20)
                        speed_factor = min(1.0, v_closing / 10.0)
                        proximity_factor = min(1.0, 8.0 / (dist_m + 1e-3))
                        
                        risk_val = float(np.clip(base_risk * speed_factor * proximity_factor, 0.0, 1.0))
                        max_pair_risk = max(max_pair_risk, risk_val)

        # Causal temporal exponential moving average (suppresses single-frame blips)
        alpha = 0.40
        self.last_score = float(np.clip(alpha * max_pair_risk + (1.0 - alpha) * self.last_score, 0.0, 1.0))
        if self.last_score < 0.02:
            self.last_score = 0.0

        return self.last_score
