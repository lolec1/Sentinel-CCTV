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
from src.event_engine import EventEngine, clean_and_merge_segments
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
        # Check if weights file exists in weights/ or BASE_DIR
        p1 = BASE_DIR / "weights" / weights
        p2 = BASE_DIR / weights
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

    # Frame sampling stride: 4 (~7.5 fps from 30 fps video) ensures high accuracy
    # while running comfortably within the <= 3x video duration budget.
    stride = 4

    tl_history: list[tuple[float, str]] = []
    frame_idx = 0
    # Rare-class fallback state
    obstacle_candidates: list[dict] = []
    smoke_candidates: list[dict] = []
    rare_events: list[list] = []

    inf_w = 640
    inf_h = int(height * (inf_w / width))
    scale_x = width / float(inf_w)
    scale_y = height / float(inf_h)

    # Background model for unknown/static foreground objects.
    # It is deliberately slow to adapt so a newly appeared object
    # can remain visible as foreground for many seconds.
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=600,
        varThreshold=32,
        detectShadows=True,
    )

    # Morphological cleanup for foreground masks.
    morph_kernel = np.ones((5, 5), dtype=np.uint8)

    # Scene masks at YOLO resolution.
    road_mask = np.zeros((inf_h, inf_w), dtype=np.uint8)
    intersection_mask = np.zeros((inf_h, inf_w), dtype=np.uint8)


    def scale_polygon(poly: np.ndarray) -> np.ndarray:
        pts = np.asarray(poly, dtype=np.float32).copy()
        pts[:, 0] /= scale_x
        pts[:, 1] /= scale_y
        return np.round(pts).astype(np.int32)


    # Everything where an obstacle can reasonably be located.
    for poly in [
        scene.roadway_sb,
        scene.roadway_nb,
        scene.intersection_zone,
    ]:
        cv2.fillPoly(
            road_mask,
            [scale_polygon(poly)],
            255,
        )

    # Smoke is allowed to occupy the whole intersection area,
    # not only the carriageway.
    cv2.fillPoly(
        intersection_mask,
        [scale_polygon(scene.intersection_zone)],
        255,
    )

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
            active_tracks = tracker.update(detections, t_sec)

            # 4. Rare-class fallback detection
            # YOLOv8n/COCO does not contain road_obstacle or fire_smoke.
            # These classes are therefore detected using temporal
            # foreground analysis and scene geometry.

            # ---------------------------------------------------------
            # 4.1 Background subtraction
            # ---------------------------------------------------------
            fg_raw = bg_subtractor.apply(
                small,
                learningRate=0.002,
            )

            # Keep only strong foreground pixels.
            # MOG2 value 127 corresponds to shadows, which we reject.
            fg_mask = np.where(
                fg_raw == 255,
                255,
                0
            ).astype(np.uint8)

            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                morph_kernel,
            )

            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_CLOSE,
                morph_kernel,
            )

            # Do not try to classify the first few seconds while
            # the background model is still stabilizing.
            if t_sec >= 5.0:

                # -----------------------------------------------------
                # 4.2 Mask known tracked traffic participants
                # -----------------------------------------------------
                occupied_mask = np.zeros(
                    (inf_h, inf_w),
                    dtype=np.uint8,
                )

                for trk in active_tracks:
                    x1, y1, x2, y2 = trk.last_bbox

                    sx1 = max(0, int(x1 / scale_x) - 8)
                    sy1 = max(0, int(y1 / scale_y) - 8)
                    sx2 = min(inf_w - 1, int(x2 / scale_x) + 8)
                    sy2 = min(inf_h - 1, int(y2 / scale_y) + 8)

                    cv2.rectangle(
                        occupied_mask,
                        (sx1, sy1),
                        (sx2, sy2),
                        255,
                        -1,
                    )

                # -----------------------------------------------------
                # 4.3 ROAD OBSTACLE
                # -----------------------------------------------------
                #
                # Candidate requirements:
                # - foreground object
                # - located on roadway
                # - not explained by a tracked vehicle/person
                # - approximately stationary
                # - remains present for >= 15 seconds

                obstacle_mask = cv2.bitwise_and(
                    fg_mask,
                    road_mask,
                )

                obstacle_mask = cv2.bitwise_and(
                    obstacle_mask,
                    cv2.bitwise_not(occupied_mask),
                )

                contours, _ = cv2.findContours(
                    obstacle_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                current_obstacles = []

                for contour in contours:
                    area = cv2.contourArea(contour)

                    # Very small blobs are camera noise.
                    # Extremely large blobs are usually background
                    # segmentation failures.
                    if area < 80.0 or area > 12000.0:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    if w < 8 or h < 8:
                        continue

                    cx = x + w / 2.0
                    cy = y + h / 2.0

                    current_obstacles.append({
                        "cx": cx,
                        "cy": cy,
                        "area": float(area),
                        "bbox": (x, y, w, h),
                    })

                # Match current contours to previous candidates.
                matched_indices = set()

                for candidate in obstacle_candidates:
                    best_idx = None
                    best_dist = float("inf")

                    for idx, obs in enumerate(current_obstacles):
                        if idx in matched_indices:
                            continue

                        dx = obs["cx"] - candidate["cx"]
                        dy = obs["cy"] - candidate["cy"]
                        dist = float(np.hypot(dx, dy))

                        # A stationary object should not jump
                        # significantly between sampled frames.
                        if dist <= 35.0 and dist < best_dist:
                            best_dist = dist
                            best_idx = idx

                    if best_idx is not None:
                        obs = current_obstacles[best_idx]
                        matched_indices.add(best_idx)

                        # If the candidate moved substantially,
                        # it is probably a moving unknown object,
                        # not a stationary road obstacle.
                        movement = float(
                            np.hypot(
                                obs["cx"] - candidate["start_cx"],
                                obs["cy"] - candidate["start_cy"],
                            )
                        )

                        if movement > 45.0:
                            candidate["start_t"] = t_sec
                            candidate["start_cx"] = obs["cx"]
                            candidate["start_cy"] = obs["cy"]
                            candidate["confirmed"] = False

                        candidate["cx"] = obs["cx"]
                        candidate["cy"] = obs["cy"]
                        candidate["area"] = obs["area"]
                        candidate["bbox"] = obs["bbox"]
                        candidate["last_seen"] = t_sec

                    elif t_sec - candidate["last_seen"] > 1.5:
                        # Candidate disappeared.
                        if candidate["confirmed"]:
                            rare_events.append([
                                candidate["start_t"],
                                candidate["last_seen"],
                                "road_obstacle",
                            ])

                        candidate["expired"] = True

                # Remove expired candidates.
                obstacle_candidates = [
                    c for c in obstacle_candidates
                    if not c.get("expired", False)
                ]

                # Create new candidates.
                for idx, obs in enumerate(current_obstacles):
                    if idx in matched_indices:
                        continue

                    obstacle_candidates.append({
                        "start_t": t_sec,
                        "last_seen": t_sec,
                        "start_cx": obs["cx"],
                        "start_cy": obs["cy"],
                        "cx": obs["cx"],
                        "cy": obs["cy"],
                        "area": obs["area"],
                        "bbox": obs["bbox"],
                        "confirmed": False,
                    })

                # Confirm only after 15 seconds of spatial persistence.
                for candidate in obstacle_candidates:
                    if candidate["confirmed"]:
                        continue

                    duration_candidate = (
                        candidate["last_seen"]
                        - candidate["start_t"]
                    )

                    if duration_candidate >= 15.0:
                        movement = float(
                            np.hypot(
                                candidate["cx"] - candidate["start_cx"],
                                candidate["cy"] - candidate["start_cy"],
                            )
                        )

                        if movement <= 45.0:
                            candidate["confirmed"] = True

                # -----------------------------------------------------
                # 4.4 FIRE / SMOKE
                # -----------------------------------------------------
                #
                # Required chromaticity:
                # S < 40
                # 50 <= V <= 180
                #
                # Combined with foreground detection so ordinary
                # gray asphalt is not treated as smoke.

                hsv = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2HSV,
                )

                smoke_color_mask = cv2.inRange(
                    hsv,
                    np.array([0, 0, 50], dtype=np.uint8),
                    np.array([180, 40, 180], dtype=np.uint8),
                )

                smoke_mask = cv2.bitwise_and(
                    fg_mask,
                    smoke_color_mask,
                )

                smoke_mask = cv2.bitwise_and(
                    smoke_mask,
                    intersection_mask,
                )

                # ---------------------------------------------------------
                # CRITICAL: never classify tracked traffic participants
                # as smoke.
                #
                # MOG2 sees approaching vehicles as rapidly growing
                # foreground regions. Their gray/low-saturation pixels
                # can satisfy the smoke HSV condition.
                #
                # Dilate the occupied boxes so the smoke contour cannot
                # hug the edges of a moving vehicle.
                # ---------------------------------------------------------

                smoke_forbidden_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (31, 31),
                )

                smoke_forbidden_mask = cv2.dilate(
                    occupied_mask,
                    smoke_forbidden_kernel,
                )

                smoke_mask = cv2.bitwise_and(
                    smoke_mask,
                    cv2.bitwise_not(smoke_forbidden_mask),
                )

                smoke_mask = cv2.morphologyEx(
                    smoke_mask,
                    cv2.MORPH_OPEN,
                    morph_kernel,
                )

                smoke_mask = cv2.morphologyEx(
                    smoke_mask,
                    cv2.MORPH_CLOSE,
                    morph_kernel,
                )

                smoke_contours, _ = cv2.findContours(
                    smoke_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                current_smoke = []

                for contour in smoke_contours:
                    area = cv2.contourArea(contour)

                    if area < 120.0 or area > 30000.0:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    if w < 10 or h < 10:
                        continue

                    cx = x + w / 2.0
                    cy = y + h / 2.0

                    # Reject contours whose center is still inside
                    # a tracked-object exclusion zone.
                    cx_i = int(np.clip(round(cx), 0, inf_w - 1))
                    cy_i = int(np.clip(round(cy), 0, inf_h - 1))

                    if smoke_forbidden_mask[cy_i, cx_i] != 0:
                        continue

                    current_smoke.append({
                        "cx": cx,
                        "cy": cy,
                        "area": float(area),
                        "bbox": (x, y, w, h),
                    })

                matched_smoke = set()

                for candidate in smoke_candidates:
                    best_idx = None
                    best_dist = float("inf")

                    for idx, smoke in enumerate(current_smoke):
                        if idx in matched_smoke:
                            continue

                        dist = float(
                            np.hypot(
                                smoke["cx"] - candidate["cx"],
                                smoke["cy"] - candidate["cy"],
                            )
                        )

                        if dist <= 70.0 and dist < best_dist:
                            best_dist = dist
                            best_idx = idx

                    if best_idx is not None:
                        smoke = current_smoke[best_idx]
                        matched_smoke.add(best_idx)

                        candidate["cx"] = smoke["cx"]
                        candidate["cy"] = smoke["cy"]
                        candidate["area"] = smoke["area"]
                        candidate["bbox"] = smoke["bbox"]
                        candidate["last_seen"] = t_sec
                        candidate["history"].append(
                            (t_sec, smoke["area"])
                        )

                        # Keep only a short temporal history.
                        candidate["history"] = [
                            item
                            for item in candidate["history"]
                            if t_sec - item[0] <= 5.0
                        ]

                        if len(candidate["history"]) >= 3:
                            first_t, first_area = candidate["history"][0]
                            last_t, last_area = candidate["history"][-1]

                            dt = last_t - first_t

                            if dt > 0.5:
                                growth = last_area / max(first_area, 1.0)
                                growth_rate = (
                                    last_area - first_area
                                ) / dt

                                if (
                                    growth >= 1.5
                                    and growth_rate > 30.0
                                ):
                                    candidate["confirmed"] = True

                    elif t_sec - candidate["last_seen"] > 1.5:
                        if candidate["confirmed"]:
                            rare_events.append([
                                candidate["start_t"],
                                candidate["last_seen"],
                                "fire_smoke",
                            ])

                        candidate["expired"] = True

                smoke_candidates = [
                    c for c in smoke_candidates
                    if not c.get("expired", False)
                ]

                # Create new smoke candidates.
                for idx, smoke in enumerate(current_smoke):
                    if idx in matched_smoke:
                        continue

                    smoke_candidates.append({
                        "start_t": t_sec,
                        "last_seen": t_sec,
                        "cx": smoke["cx"],
                        "cy": smoke["cy"],
                        "area": smoke["area"],
                        "bbox": smoke["bbox"],
                        "history": [
                            (t_sec, smoke["area"])
                        ],
                        "confirmed": False,
                    })

        frame_idx += 1
    # Close rare-class candidates that are still active at EOF.
    for candidate in obstacle_candidates:
        if candidate["confirmed"]:
            rare_events.append([
                candidate["start_t"],
                candidate["last_seen"],
                "road_obstacle",
            ])

    for candidate in smoke_candidates:
        if candidate["confirmed"]:
            rare_events.append([
                candidate["start_t"],
                candidate["last_seen"],
                "fire_smoke",
            ])

    cap.release()

    # 4. Event Evaluation across all tracks

    all_tracks = tracker.get_all_tracks()
    engine = EventEngine(scene, duration)
    events = engine.process(all_tracks, tl_history)

    # Add rare classes detected by the frame-based fallback.
    events.extend(rare_events)

    # Final cleanup:
    # - remove invalid/very short segments
    # - merge overlapping same-class segments
    # - keep all timestamps inside video duration
    events = clean_and_merge_segments(events,duration)

    return events
