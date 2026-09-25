"""
scene.py — Fixed CCTV Camera Geometry & Layout for WIUT Intersection.
Calibrated for 3840x2160 resolution (and normalized for arbitrary resolutions).
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Literal

# Reference resolution
REF_W = 3840
REF_H = 2160

# 1. Traffic Light ROI on Median: [ymin, ymax, xmin, xmax]
TL_ROI = (710, 850, 2290, 2360)

# 2. Main Stop Line (for southbound traffic approaching intersection)
STOP_LINE = np.array([[480, 930], [1950, 990]], dtype=np.float32)

# 3. Crosswalk Polygons (Zebra crossings)
# Main crosswalk (southbound lanes across to median)
CROSSWALK_MAIN = np.array([[450, 1010], [2280, 1020], [2280, 1260], [450, 1320]], dtype=np.float32)

# Right crosswalk (median across to right sidewalk)
CROSSWALK_RIGHT = np.array([[2380, 970], [3500, 1000], [3500, 1180], [2380, 1160]], dtype=np.float32)

# Bottom crosswalk (foreground zebra crossing)
CROSSWALK_BOTTOM = np.array([[600, 1400], [1400, 1400], [1750, 2160], [550, 2160]], dtype=np.float32)

ALL_CROSSWALKS = [CROSSWALK_MAIN, CROSSWALK_RIGHT, CROSSWALK_BOTTOM]

# 4. Roadway Carriageway Polygons
# Southbound roadway (normal flow: dy > 0, moving down towards intersection)
ROADWAY_SOUTHBOUND = np.array([
    [300, 100], [1550, 340], [2280, 920], [2280, 1400],
    [1800, 2160], [0, 2160], [0, 1000], [200, 600]
], dtype=np.float32)

# Northbound roadway (normal flow: dy < 0, moving away/upwards)
ROADWAY_NORTHBOUND = np.array([
    [1600, 250], [3840, 600], [3840, 1400], [2350, 950],
    [2300, 750], [1650, 320]
], dtype=np.float32)

# Entire Intersection Polygon
INTERSECTION_ZONE = np.array([
    [450, 950], [2300, 950], [3600, 1050], [3840, 1500],
    [3840, 2160], [1800, 2160], [450, 1350]
], dtype=np.float32)

# Stop-line buffer zone (between stop-line and zebra, for stop_line violations)
STOP_LINE_ZONE = np.array([
    [480, 920], [1950, 980], [1980, 1030], [470, 1010]
], dtype=np.float32)

# Solid lane divider lines (approaching intersection)
SOLID_LINE_1 = np.array([[950, 650], [1050, 940]], dtype=np.float32)
SOLID_LINE_2 = np.array([[1420, 650], [1520, 960]], dtype=np.float32)
SOLID_LINES = [SOLID_LINE_1, SOLID_LINE_2]


class SceneGeometry:
    """Handles coordinate scaling and spatial queries."""
    def __init__(self, width: int = REF_W, height: int = REF_H):
        self.width = width
        self.height = height
        self.sx = width / float(REF_W)
        self.sy = height / float(REF_H)

        # Scale polygons to current video resolution
        self.tl_roi = (
            int(TL_ROI[0] * self.sy), int(TL_ROI[1] * self.sy),
            int(TL_ROI[2] * self.sx), int(TL_ROI[3] * self.sx)
        )
        self.stop_line = self._scale_poly(STOP_LINE)
        self.crosswalk_main = self._scale_poly(CROSSWALK_MAIN)
        self.crosswalk_right = self._scale_poly(CROSSWALK_RIGHT)
        self.crosswalk_bottom = self._scale_poly(CROSSWALK_BOTTOM)
        self.all_crosswalks = [self.crosswalk_main, self.crosswalk_right, self.crosswalk_bottom]
        self.roadway_sb = self._scale_poly(ROADWAY_SOUTHBOUND)
        self.roadway_nb = self._scale_poly(ROADWAY_NORTHBOUND)
        self.intersection_zone = self._scale_poly(INTERSECTION_ZONE)
        self.stop_line_zone = self._scale_poly(STOP_LINE_ZONE)
        self.solid_lines = [self._scale_poly(sl) for sl in SOLID_LINES]

    def _scale_poly(self, poly: np.ndarray) -> np.ndarray:
        scaled = poly.copy()
        scaled[:, 0] *= self.sx
        scaled[:, 1] *= self.sy
        return scaled

    def is_inside_polygon(self, point: tuple[float, float], polygon: np.ndarray) -> bool:
        """Check if point (x, y) is inside polygon."""
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def is_in_crosswalk(self, point: tuple[float, float]) -> bool:
        """Check if point is inside any designated crosswalk."""
        for cw in self.all_crosswalks:
            if self.is_inside_polygon(point, cw):
                return True
        return False

    def is_on_roadway(self, point: tuple[float, float]) -> bool:
        """Check if point is on the carriageway."""
        return self.is_inside_polygon(point, self.roadway_sb) or \
               self.is_inside_polygon(point, self.roadway_nb) or \
               self.is_inside_polygon(point, self.intersection_zone)

    def crosses_stop_line(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Check if trajectory segment from p1 to p2 crosses the stop line (from top to bottom)."""
        # Stop line: approx from left (x1, y1) to right (x2, y2)
        sl1 = self.stop_line[0]
        sl2 = self.stop_line[1]
        
        # Check segment intersection
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        A, B = p1, p2
        C, D = (sl1[0], sl1[1]), (sl2[0], sl2[1])
        intersects = (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        # Ensure moving forward/downward (p2[1] > p1[1])
        return intersects and (p2[1] >= p1[1])

    def crosses_solid_line(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Check if vehicle trajectory crossed any solid lane marking."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        for line in self.solid_lines:
            C, D = (line[0][0], line[0][1]), (line[1][0], line[1][1])
            A, B = p1, p2
            if (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)):
                return True
        return False


class TrafficLightTracker:
    """Robust, temporally smoothed traffic light detector."""
    def __init__(self, scene: SceneGeometry):
        self.scene = scene
        self.history: list[tuple[float, str]] = [] # (t_sec, state)
        self.last_stable_state = "RED"

    def update(self, frame: np.ndarray, t_sec: float) -> str:
        ymin, ymax, xmin, xmax = self.scene.tl_roi
        # Ensure inside bounds
        h, w = frame.shape[:2]
        ymin, ymax = max(0, ymin), min(h, ymax)
        xmin, xmax = max(0, xmin), min(w, xmax)
        
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return self.last_stable_state

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_r = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
        mask_g = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

        r_px = int(np.sum(mask_r > 0))
        g_px = int(np.sum(mask_g > 0))

        raw_state = "UNKNOWN"
        if r_px > 30 and r_px > g_px:
            raw_state = "RED"
        elif g_px > 30 and g_px > r_px:
            raw_state = "GREEN"

        self.history.append((t_sec, raw_state))

        # Temporal filter: look at past 1.0 second
        recent = [s for (t, s) in self.history if t_sec - t <= 1.0 and s != "UNKNOWN"]
        if recent:
            red_count = recent.count("RED")
            green_count = recent.count("GREEN")
            if red_count >= green_count and red_count > 0:
                self.last_stable_state = "RED"
            elif green_count > red_count:
                self.last_stable_state = "GREEN"

        return self.last_stable_state
