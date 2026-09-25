"""
solution.py — Official submission entrypoint for WIUT Hackathon 2026.
Implements detect_events (Part A) and RiskEstimator (Part B).
"""
from __future__ import annotations
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.scene import SceneGeometry, TrafficLightTracker
from src.tracker import RoadTracker, Detection
from src.event_engine import EventEngine
from src.risk_estimator import RiskEstimator

# Official 14 classes
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

# Pre-load detector singleton
_MODEL_CACHE: dict[str, YOLO] = {}

def get_yolo_model(weights: str = "yolov8n.pt") -> YOLO:
    if weights not in _MODEL_CACHE:
        # Check if weights file exists in BASE_DIR or weights/
        p1 = BASE_DIR / weights
        p2 = BASE_DIR / "weights" / weights
        target = str(p1 if p1.exists() else (p2 if p2.exists() else weights))
        _MODEL_CACHE[weights] = YOLO(target)
    return _MODEL_CACHE[weights]


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        List of [start_sec, end_sec, label] with 0 <= start_sec < end_sec <= duration.
        Same-class segments do not overlap.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / float(fps) if fps else 0.0

    model = get_yolo_model("yolov8n.pt")
    scene = SceneGeometry(width, height)
    tracker = RoadTracker(max_misses=5, iou_thresh=0.25)
    tl_tracker = TrafficLightTracker(scene)

    # Frame sampling stride: 3 (~10 fps from 30 fps video) ensures high accuracy
    # while running comfortably within the <= 3x video duration budget.
    stride = 3

    tl_history: list[tuple[float, str]] = []
    frame_idx = 0
    inf_w = 640
    inf_h = int(height * (inf_w / width))
    scale_x = width / float(inf_w)
    scale_y = height / float(inf_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            t_sec = frame_idx / fps

            # 1. Update Traffic Light State
            tl_state = tl_tracker.update(frame, t_sec)
            tl_history.append((t_sec, tl_state))

            # 2. Downscaled YOLO Detection
            small = cv2.resize(frame, (inf_w, inf_h))
            results = model(small, conf=0.25, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                name = model.names[cls_id]
                if name in ["car", "truck", "bus", "motorcycle", "person", "bicycle"]:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    orig_bbox = (bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y)
                    detections.append(Detection(orig_bbox, name, float(box.conf[0].item())))

            # 3. Multi-Object Tracking
            tracker.update(detections, t_sec)

        frame_idx += 1

    cap.release()

    # 4. Event Evaluation across all tracks
    all_tracks = tracker.get_all_tracks()
    engine = EventEngine(scene, duration)
    events = engine.process(all_tracks, tl_history)

    return events
