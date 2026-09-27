from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.scene import SceneGeometry
from src.tracker import RoadTracker, Detection


VIDEO_PATH = "test_videos/sample_004.mp4"
OUTPUT_DIR = Path("debug_fire_smoke")

START_SEC = 39.0
END_SEC = 46.0

# Same settings as solution.py
INF_W = 640

MOG_HISTORY = 600
MOG_VAR_THRESHOLD = 32
MOG_LEARNING_RATE = 0.002

MORPH_KERNEL_SIZE = 5

SMOKE_HSV_LOW = np.array([0, 0, 50], dtype=np.uint8)
SMOKE_HSV_HIGH = np.array([180, 40, 180], dtype=np.uint8)

MIN_CONTOUR_AREA = 120
MAX_CONTOUR_AREA = 30000
MIN_BBOX_W = 10
MIN_BBOX_H = 10


def resize_mask(mask, width, height):
    return cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask(frame, mask, alpha=0.35):
    result = frame.copy()

    overlay = np.zeros_like(frame)
    overlay[:, :, 1] = mask  # green

    return cv2.addWeighted(result, 1.0, overlay, alpha, 0)


def draw_polygon_mask(frame, polygon, label):
    result = frame.copy()

    if polygon is None or len(polygon) < 3:
        return result

    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))

    cv2.polylines(
        result,
        [pts],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )

    x, y = pts[0, 0]

    cv2.putText(
        result,
        label,
        (int(x), max(20, int(y) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def scale_polygon(poly, scale_x, scale_y):
    arr = np.asarray(poly, dtype=np.float32).copy()
    arr[:, 0] /= scale_x
    arr[:, 1] /= scale_y
    return arr.astype(np.int32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {frame_count}")
    print(f"Debug interval: {START_SEC:.2f} - {END_SEC:.2f} sec")

    model = YOLO("yolov8n.pt")

    scene = SceneGeometry(width, height)
    tracker = RoadTracker(max_misses=5, iou_thresh=0.25)

    # Same inference scaling as solution.py
    inf_w = INF_W
    inf_h = int(height * (inf_w / width))

    scale_x = width / float(inf_w)
    scale_y = height / float(inf_h)

    print(f"Inference resolution: {inf_w}x{inf_h}")
    print(f"Scale: x={scale_x:.3f}, y={scale_y:.3f}")

    # Same MOG2 configuration
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG_HISTORY,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=True,
    )

    morph_kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        dtype=np.uint8,
    )

    # Build masks in inference resolution
    road_mask = np.zeros((inf_h, inf_w), dtype=np.uint8)
    intersection_mask = np.zeros((inf_h, inf_w), dtype=np.uint8)

    road_polygons = [
        scene.roadway_sb,
        scene.roadway_nb,
        scene.intersection_zone,
    ]

    for polygon in road_polygons:
        if polygon is not None and len(polygon) >= 3:
            scaled = scale_polygon(
                polygon,
                scale_x,
                scale_y,
            )
            cv2.fillPoly(
                road_mask,
                [scaled],
                255,
            )

    if scene.intersection_zone is not None:
        scaled = scale_polygon(
            scene.intersection_zone,
            scale_x,
            scale_y,
        )
        cv2.fillPoly(
            intersection_mask,
            [scaled],
            255,
        )

    start_frame = max(0, int(START_SEC * fps))
    end_frame = min(
        frame_count - 1,
        int(END_SEC * fps),
    )

    # We must run MOG2 from the beginning because its state/history
    # matters. Only frames in 39-46 seconds are saved.
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        t_sec = frame_idx / fps

        # Same sampling as solution.py
        if frame_idx % 4 != 0:
            frame_idx += 1
            continue

        # Same resize
        small = cv2.resize(
            frame,
            (inf_w, inf_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # ---------------------------------------------------------
        # YOLO
        # ---------------------------------------------------------

        results = model(
            small,
            conf=0.25,
            verbose=False,
        )[0]

        detections = []

        yolo_frame = small.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            name = model.names[cls_id]

            if name not in [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "person",
                "bicycle",
            ]:
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            orig_bbox = (
                bx1 * scale_x,
                by1 * scale_y,
                bx2 * scale_x,
                by2 * scale_y,
            )

            detections.append(
                Detection(
                    orig_bbox,
                    name,
                    conf,
                )
            )

            cv2.rectangle(
                yolo_frame,
                (int(bx1), int(by1)),
                (int(bx2), int(by2)),
                (255, 0, 0),
                2,
            )

            label = f"{name} {conf:.2f}"

            cv2.putText(
                yolo_frame,
                label,
                (int(bx1), max(15, int(by1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        active_tracks = tracker.update(
            detections,
            t_sec,
        )

        # ---------------------------------------------------------
        # MOG2
        # ---------------------------------------------------------

        fg_raw = bg_subtractor.apply(
            small,
            learningRate=MOG_LEARNING_RATE,
        )

        # Exactly as solution.py:
        # only 255 = actual foreground.
        fg_mask = np.where(
            fg_raw == 255,
            255,
            0,
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

        # ---------------------------------------------------------
        # Occupied mask from tracked vehicles/people
        # ---------------------------------------------------------

        occupied_mask = np.zeros(
            (inf_h, inf_w),
            dtype=np.uint8,
        )

        for trk in active_tracks:
            x1, y1, x2, y2 = trk.last_bbox

            sx1 = int(x1 / scale_x)
            sy1 = int(y1 / scale_y)
            sx2 = int(x2 / scale_x)
            sy2 = int(y2 / scale_y)

            sx1 = max(0, sx1 - 8)
            sy1 = max(0, sy1 - 8)
            sx2 = min(inf_w - 1, sx2 + 8)
            sy2 = min(inf_h - 1, sy2 + 8)

            cv2.rectangle(
                occupied_mask,
                (sx1, sy1),
                (sx2, sy2),
                255,
                -1,
            )

        # ---------------------------------------------------------
        # Smoke mask
        # ---------------------------------------------------------

        hsv = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2HSV,
        )

        smoke_color_mask = cv2.inRange(
            hsv,
            SMOKE_HSV_LOW,
            SMOKE_HSV_HIGH,
        )

        # Same logic as solution.py:
        # foreground + smoke-like color + intersection.
        smoke_mask = (
            fg_mask
            & smoke_color_mask
            & intersection_mask
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

        # ---------------------------------------------------------
        # Fire/smoke contours
        # ---------------------------------------------------------

        contours, _ = cv2.findContours(
            smoke_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contour_frame = small.copy()

        # Draw intersection boundary
        if scene.intersection_zone is not None:
            scaled_intersection = scale_polygon(
                scene.intersection_zone,
                scale_x,
                scale_y,
            )

            cv2.polylines(
                contour_frame,
                [scaled_intersection],
                True,
                (0, 255, 255),
                2,
            )

        accepted_contours = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < MIN_CONTOUR_AREA:
                continue

            if area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < MIN_BBOX_W or h < MIN_BBOX_H:
                continue

            accepted_contours.append(
                (contour, area, x, y, w, h)
            )

            cv2.drawContours(
                contour_frame,
                [contour],
                -1,
                (0, 0, 255),
                2,
            )

            cv2.rectangle(
                contour_frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2,
            )

            cx = x + w / 2
            cy = y + h / 2

            cv2.circle(
                contour_frame,
                (int(cx), int(cy)),
                4,
                (0, 0, 255),
                -1,
            )

            text_label = (
                f"area={area:.0f} "
                f"bbox={w}x{h}"
            )

            cv2.putText(
                contour_frame,
                text_label,
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # ---------------------------------------------------------
        # Save only requested time interval
        # ---------------------------------------------------------

        if start_frame <= frame_idx <= end_frame:

            timestamp = f"{t_sec:06.2f}s"
            prefix = OUTPUT_DIR / f"frame_{frame_idx:05d}_{timestamp}"

            # 1. Original + YOLO
            cv2.imwrite(
                str(prefix) + "_yolo.jpg",
                yolo_frame,
            )

            # 2. Road mask
            cv2.imwrite(
                str(prefix) + "_road_mask.jpg",
                mask_to_bgr(road_mask),
            )

            # 3. Intersection mask
            cv2.imwrite(
                str(prefix) + "_intersection_mask.jpg",
                mask_to_bgr(intersection_mask),
            )

            # 4. MOG2
            cv2.imwrite(
                str(prefix) + "_mog2.jpg",
                mask_to_bgr(fg_mask),
            )

            # 5. Smoke mask
            cv2.imwrite(
                str(prefix) + "_smoke_mask.jpg",
                mask_to_bgr(smoke_mask),
            )

            # 6. Fire/smoke contours
            cv2.imwrite(
                str(prefix) + "_contours.jpg",
                contour_frame,
            )

            # -----------------------------------------------------
            # Combined diagnostic frame
            # -----------------------------------------------------

            combined = small.copy()

            # Road mask overlay
            combined = cv2.addWeighted(
                combined,
                0.75,
                cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR),
                0.20,
                0,
            )

            # Intersection outline
            if scene.intersection_zone is not None:
                scaled_intersection = scale_polygon(
                    scene.intersection_zone,
                    scale_x,
                    scale_y,
                )

                cv2.polylines(
                    combined,
                    [scaled_intersection],
                    True,
                    (0, 255, 255),
                    2,
                )

            # YOLO boxes
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                name = model.names[cls_id]

                if name not in [
                    "car",
                    "truck",
                    "bus",
                    "motorcycle",
                    "person",
                    "bicycle",
                ]:
                    continue

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())

                cv2.rectangle(
                    combined,
                    (int(bx1), int(by1)),
                    (int(bx2), int(by2)),
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    combined,
                    f"{name} {conf:.2f}",
                    (int(bx1), max(15, int(by1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Smoke contour
            for contour, area, x, y, bw, bh in accepted_contours:

                cv2.drawContours(
                    combined,
                    [contour],
                    -1,
                    (0, 0, 255),
                    2,
                )

                cv2.rectangle(
                    combined,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    combined,
                    f"SMOKE area={area:.0f}",
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Timestamp
            cv2.putText(
                combined,
                f"t = {t_sec:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(prefix) + "_combined.jpg",
                combined,
            )

            print(
                f"[{t_sec:6.2f}s] "
                f"YOLO={len(detections):2d} "
                f"tracks={len(active_tracks):2d} "
                f"FG={np.count_nonzero(fg_mask):6d} "
                f"smoke={np.count_nonzero(smoke_mask):6d} "
                f"contours={len(accepted_contours):2d}"
            )

        frame_idx += 1

        if frame_idx > end_frame:
            break

    cap.release()

    print()
    print(f"Done. Debug images saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()