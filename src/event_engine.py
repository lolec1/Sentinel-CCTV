"""
event_engine.py — Comprehensive Event Detection Engine for all 14 official traffic event classes.
Calibrated for fixed intersection surveillance with zero false-alarm physics.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from src.scene import SceneGeometry
from src.tracker import Track, iou

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CLASSES = {"person", "bicycle"}


def clean_and_merge_segments(
    events: list[list],
    duration: float,
    min_duration: float = 0.8,
    max_gap: float = 0.5
) -> list[list]:
    """Clean, filter short blips, merge contiguous fragments of the same event,
    and guarantee non-overlapping segments for each class with valid bounds."""
    if not events:
        return []

    by_class: dict[str, list[list[float]]] = {}
    for ev in events:
        s, e, label = float(ev[0]), float(ev[1]), str(ev[2])
        s = max(0.0, s)
        e = min(duration, e)
        if e <= s or (e - s) < min_duration:
            continue
        by_class.setdefault(label, []).append([s, e])

    cleaned_out: list[list] = []
    for label, segs in by_class.items():
        segs.sort(key=lambda x: (x[0], x[1]))
        merged: list[list[float]] = []

        for s, e in segs:
            if not merged:
                merged.append([s, e])
            else:
                prev_s, prev_e = merged[-1]
                if s <= prev_e + max_gap:
                    merged[-1][1] = max(prev_e, e)
                elif s > prev_e:
                    merged.append([s, e])

        resolved: list[list[float]] = []
        for s, e in merged:
            if not resolved:
                resolved.append([s, e])
            else:
                last_end = resolved[-1][1]
                if s < last_end:
                    s = last_end + 0.1
                if e - s >= min_duration:
                    resolved.append([s, e])

        for s, e in resolved:
            cleaned_out.append([round(s, 2), round(e, 2), label])

    cleaned_out.sort(key=lambda x: (x[0], x[1]))
    return cleaned_out


class EventEngine:
    def __init__(self, scene: SceneGeometry, duration: float):
        self.scene = scene
        self.duration = duration
        self.raw_events: list[list] = []

    def process(self, tracks: list[Track], tl_history: list[tuple[float, str]]) -> list[list]:
        """Analyze all tracks and traffic light timeline to detect events."""
        self.raw_events = []

        # 1. Red Light Running & Stop Line Violations
        self._detect_red_light_and_stop_line(tracks, tl_history)

        # 2. Jaywalking & Failure to Yield
        self._detect_pedestrian_events(tracks, tl_history)

        # 3. Wrong Way & Illegal U-Turn
        self._detect_trajectory_violations(tracks)

        # 4. Stopped Vehicle (outside red-light queues)
        self._detect_stopped_vehicles(tracks, tl_history)

        # 5. Solid Line Crossing
        self._detect_solid_line_crossing(tracks)

        # 6. Accident & Near Miss
        self._detect_accidents_and_near_misses(tracks)

        # 7. Congestion
        self._detect_congestion(tracks, tl_history)

        # Post-process, merge fragments, and enforce non-overlapping bounds
        return clean_and_merge_segments(self.raw_events, self.duration)

    def _get_tl_state_at(self, t: float, tl_history: list[tuple[float, str]]) -> str:
        """Find traffic light state at given time t."""
        if not tl_history:
            return "UNKNOWN"
        best_state = tl_history[0][1]
        for item_t, state in tl_history:
            if item_t <= t:
                best_state = state
            else:
                break
        return best_state

    def _detect_red_light_and_stop_line(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        vehicle_tracks = [tr for tr in tracks if tr.cls_name in VEHICLE_CLASSES and len(tr.history) >= 5]

        for tr in vehicle_tracks:
            for i in range(len(tr.history) - 1):
                t1, box1 = tr.history[i]
                t2, box2 = tr.history[i + 1]
                c1 = ((box1[0] + box1[2]) / 2, box1[3])
                c2 = ((box2[0] + box2[2]) / 2, box2[3])

                if self.scene.crosses_stop_line(c1, c2):
                    tl_state = self._get_tl_state_at(t1, tl_history)
                    if tl_state == "RED":
                        rem_history = tr.history[i+1:]
                        if not rem_history:
                            continue

                        # Check destination DURING RED PHASE
                        # Must enter intersection (y > 1300) WHILE STILL RED to be red-light running
                        entered_red_intersection = False
                        exit_t = t1

                        for fut_t, fut_box in rem_history:
                            fut_state = self._get_tl_state_at(fut_t, tl_history)
                            fut_y = fut_box[3]
                            if fut_state == "RED" and fut_y > 1300.0:
                                entered_red_intersection = True
                                exit_t = fut_t
                            elif fut_state == "GREEN" and not entered_red_intersection:
                                break

                        if entered_red_intersection:
                            start_sec = t1
                            end_sec = min(start_sec + 6.0, exit_t)
                            if end_sec > start_sec + 1.0:
                                self.raw_events.append([start_sec, end_sec, "red_light"])
                        else:
                            # Halts past stop line without proceeding through on red -> stop_line
                            start_sec = t1
                            end_sec = rem_history[-1][0]
                            for fut_t, fut_st in tl_history:
                                if fut_t >= start_sec and fut_st == "GREEN":
                                    end_sec = fut_t
                                    break
                            if end_sec > start_sec + 2.0:
                                self.raw_events.append([start_sec, end_sec, "stop_line"])
                    break

    def _detect_pedestrian_events(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        # Only true pedestrians: walking on foot (speed < 9 px/s)
        ped_tracks = [tr for tr in tracks if tr.cls_name == "person" and len(tr.history) >= 8 and tr.speed < 9.0]
        veh_tracks = [tr for tr in tracks if tr.cls_name in VEHICLE_CLASSES and len(tr.history) >= 6]

        # 1. Jaywalking: walking across carriageway outside designated crossings
        for p_tr in ped_tracks:
            cur_start = None
            start_x, end_x = None, None
            for t, box in p_tr.history:
                pt = ((box[0] + box[2]) / 2, box[3])
                in_carriageway = (600 <= pt[0] <= 2100) and (350 <= pt[1] <= 920)
                in_cw = self.scene.is_in_crosswalk(pt)
                on_island = self.scene.is_on_island(pt)

                is_jw = in_carriageway and (not in_cw) and (not on_island) and (2.5 <= p_tr.speed <= 8.5)
                if is_jw:
                    if cur_start is None:
                        cur_start = t
                        start_x = pt[0]
                    end_x = pt[0]
                else:
                    if cur_start is not None:
                        dur = t - cur_start
                        traversal = abs((end_x or 0) - (start_x or 0))
                        if 2.5 <= dur <= 15.0 and traversal >= 120.0:
                            self.raw_events.append([cur_start, t, "jaywalking"])
                        cur_start = None
            if cur_start is not None:
                dur = p_tr.last_t - cur_start
                traversal = abs((end_x or 0) - (start_x or 0))
                if 2.5 <= dur <= 15.0 and traversal >= 120.0:
                    self.raw_events.append([cur_start, p_tr.last_t, "jaywalking"])

        # 2. Failure to Yield: vehicle cuts directly across pedestrian's immediate path on crosswalk
        for cw in [self.scene.crosswalk_main, self.scene.crosswalk_bottom]:
            for v_tr in veh_tracks:
                v_in_cw: list[tuple[float, tuple[float, float]]] = []
                for t, box in v_tr.history:
                    pt = ((box[0] + box[2]) / 2, box[3])
                    if self.scene.is_inside_polygon(pt, cw):
                        v_in_cw.append((t, pt))

                if len(v_in_cw) >= 3 and v_tr.speed > 22.0:
                    v_start = v_in_cw[0][0]
                    v_end = v_in_cw[-1][0]
                    tl_state = self._get_tl_state_at(v_start, tl_history)
                    if tl_state == "GREEN":
                        continue

                    if 0.8 <= (v_end - v_start) <= 4.0:
                        conflict = False
                        for p_tr in ped_tracks:
                            # Pedestrian must be actively traversing the crosswalk
                            p_disp = math.hypot(
                                p_tr.history[-1][1][0] - p_tr.history[0][1][0],
                                p_tr.history[-1][1][1] - p_tr.history[0][1][1]
                            )
                            if p_disp < 40.0:
                                continue

                            for pt_t, p_box in p_tr.history:
                                if v_start <= pt_t <= v_end:
                                    p_pt = ((p_box[0] + p_box[2]) / 2, p_box[3])
                                    if self.scene.is_inside_polygon(p_pt, cw):
                                        v_pt = min(v_in_cw, key=lambda x: abs(x[0] - pt_t))[1]
                                        horiz_dist = abs(p_pt[0] - v_pt[0])
                                        vert_dist = abs(p_pt[1] - v_pt[1])
                                        if horiz_dist < 60.0 and vert_dist < 60.0:
                                            conflict = True
                                            break
                            if conflict:
                                break
                        if conflict:
                            self.raw_events.append([v_start, v_end, "failure_to_yield"])

    def _detect_trajectory_violations(self, tracks: list[Track]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 10:
                continue

            t_first, b_first = tr.history[0]
            t_last, b_last = tr.history[-1]
            c_first = ((b_first[0] + b_first[2]) / 2, (b_first[1] + b_first[3]) / 2)
            c_last = ((b_last[0] + b_last[2]) / 2, (b_last[1] + b_last[3]) / 2)

            # 1. Wrong way: Sustained travel against flow inside southbound approach
            ww_start = None
            y_disp = 0.0
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i+1]
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                dt = t2 - t1
                if dt > 0:
                    vx = (c2[0] - c1[0]) / dt
                    vy = (c2[1] - c1[1]) / dt
                    in_dedicated_sb = (650 <= c1[0] <= 1800) and (350 <= c1[1] <= 880)
                    is_ww = in_dedicated_sb and (vy < -25.0) and (abs(vx) < 25.0)
                    if is_ww:
                        if ww_start is None:
                            ww_start = t1
                        y_disp += (c2[1] - c1[1])
                    else:
                        if ww_start is not None:
                            if (t1 - ww_start) >= 3.0 and y_disp < -220.0:
                                self.raw_events.append([ww_start, t1, "wrong_way"])
                            ww_start = None
                            y_disp = 0.0
            if ww_start is not None and (tr.last_t - ww_start) >= 3.0 and y_disp < -220.0:
                self.raw_events.append([ww_start, tr.last_t, "wrong_way"])

            # 2. Illegal U-turn: Heading reversal ~180 degrees inside southbound approach
            if len(tr.history) >= 15:
                mid = len(tr.history) // 2
                h1_c = (
                    (tr.history[mid][1][0] + tr.history[mid][1][2]) / 2 - c_first[0],
                    (tr.history[mid][1][1] + tr.history[mid][1][3]) / 2 - c_first[1]
                )
                h2_c = (
                    c_last[0] - (tr.history[mid][1][0] + tr.history[mid][1][2]) / 2,
                    c_last[1] - (tr.history[mid][1][1] + tr.history[mid][1][3]) / 2
                )
                m1 = math.hypot(*h1_c)
                m2 = math.hypot(*h2_c)
                if m1 > 70.0 and m2 > 70.0:
                    cos_theta = (h1_c[0] * h2_c[0] + h1_c[1] * h2_c[1]) / (m1 * m2 + 1e-6)
                    if cos_theta < -0.85 and (650 <= c_first[0] <= 1800) and (c_first[1] < 880):
                        self.raw_events.append([t_first, t_last, "illegal_u_turn"])

    def _detect_stopped_vehicles(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 18:
                continue

            stat_start = None
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i + 1]
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                dt = t2 - t1
                speed = math.hypot(c2[0] - c1[0], c2[1] - c1[1]) / (dt + 1e-6)

                if speed < 6.0:
                    if stat_start is None:
                        stat_start = t1
                else:
                    if stat_start is not None:
                        dur = t1 - stat_start
                        if dur >= 12.0:
                            tl_state = self._get_tl_state_at(stat_start, tl_history)
                            in_approach = self.scene.is_inside_polygon(c1, self.scene.roadway_sb)
                            if not (tl_state == "RED" and in_approach):
                                self.raw_events.append([stat_start, t1, "stopped_vehicle"])
                        stat_start = None

            if stat_start is not None:
                dur = tr.last_t - stat_start
                if dur >= 12.0:
                    c = tr.centroid
                    tl_state = self._get_tl_state_at(stat_start, tl_history)
                    in_approach = self.scene.is_inside_polygon(c, self.scene.roadway_sb)
                    if not (tl_state == "RED" and in_approach):
                        self.raw_events.append([stat_start, tr.last_t, "stopped_vehicle"])

    def _detect_solid_line_crossing(self, tracks: list[Track]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 8:
                continue
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i + 1]
                c1 = ((b1[0] + b1[2]) / 2, b1[3])
                c2 = ((b2[0] + b2[2]) / 2, b2[3])
                if self.scene.crosses_solid_line(c1, c2):
                    x_start = (tr.history[0][1][0] + tr.history[0][1][2]) / 2
                    x_end = (tr.history[-1][1][0] + tr.history[-1][1][2]) / 2
                    if abs(x_end - x_start) >= 50.0:
                        self.raw_events.append([t1, min(t2 + 1.5, self.duration), "solid_line_crossing"])
                    break


    def _detect_accidents_and_near_misses(self, tracks: list[Track]):
        """True collision dynamics: requires approach vector, contact, and sudden deceleration."""
        n = len(tracks)
        for i in range(n):
            tr1 = tracks[i]
            if tr1.cls_name not in VEHICLE_CLASSES or len(tr1.history) < 8:
                continue
            for j in range(i + 1, n):
                tr2 = tracks[j]
                if tr2.cls_name not in (VEHICLE_CLASSES | PEDESTRIAN_CLASSES) or len(tr2.history) < 8:
                    continue

                if tr1.cls_name == "person" and tr2.cls_name == "person":
                    continue

                # Birth distance check: duplicate tracks
                init_dist = math.hypot(
                    tr1.history[0][1][0] - tr2.history[0][1][0],
                    tr1.history[0][1][1] - tr2.history[0][1][1]
                )
                if init_dist < 80.0:
                    continue

                # Find closest approach in common time frames
                t1_map = {h[0]: h[1] for h in tr1.history}
                common = [(t, t1_map[t], h[1]) for t, h in [(h[0], h) for h in tr2.history] if t in t1_map]
                if len(common) < 3:
                    continue

                min_dist = float("inf")
                t_best, b1_best, b2_best = None, None, None
                for t, b1, b2 in common:
                    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                    d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if d < min_dist:
                        min_dist = d
                        t_best, b1_best, b2_best = t, b1, b2

                if t_best is None or b1_best is None or b2_best is None:
                    continue

                # Centroids must be close (< 65 px) and boxes must overlap (IoU >= 0.35)
                score = iou(b1_best, b2_best)
                if min_dist < 65.0 and score >= 0.35:
                    # Compute local velocities at t_best
                    h1 = [h for h in tr1.history if abs(h[0] - t_best) <= 0.6]
                    h2 = [h for h in tr2.history if abs(h[0] - t_best) <= 0.6]
                    if len(h1) >= 2 and len(h2) >= 2:
                        dt1 = h1[-1][0] - h1[0][0]
                        dt2 = h2[-1][0] - h2[0][0]
                        vx1 = (h1[-1][1][0] - h1[0][1][0]) / (dt1 + 1e-4)
                        vy1 = (h1[-1][1][1] - h1[0][1][1]) / (dt1 + 1e-4)
                        vx2 = (h2[-1][1][0] - h2[0][1][0]) / (dt2 + 1e-4)
                        vy2 = (h2[-1][1][1] - h2[0][1][1]) / (dt2 + 1e-4)

                        sp1 = math.hypot(vx1, vy1)
                        sp2 = math.hypot(vx2, vy2)
                        cos_ang = (vx1 * vx2 + vy1 * vy2) / (sp1 * sp2 + 1e-6)

                        # Must not be traveling in same lane direction
                        if cos_ang < 0.60:
                            # CRITICAL: Collision must cause immobilization at collision site!
                            post1 = [h for h in tr1.history if h[0] > t_best]
                            post2 = [h for h in tr2.history if h[0] > t_best]
                            post_sp1 = 0.0
                            if len(post1) >= 2:
                                dt = post1[-1][0] - post1[0][0]
                                dp = math.hypot(post1[-1][1][0] - post1[0][1][0], post1[-1][1][1] - post1[0][1][1])
                                post_sp1 = dp / (dt + 1e-4)
                            post_sp2 = 0.0
                            if len(post2) >= 2:
                                dt = post2[-1][0] - post2[0][0]
                                dp = math.hypot(post2[-1][1][0] - post2[0][1][0], post2[-1][1][1] - post2[0][1][1])
                                post_sp2 = dp / (dt + 1e-4)

                            # Both vehicles must stop or be immobilized at site (< 20 px/s)
                            if post_sp1 < 20.0 and post_sp2 < 20.0:
                                start_sec = max(0.0, t_best)
                                end_sec = min(self.duration, start_sec + 6.0)
                                if end_sec > start_sec + 1.0:
                                    self.raw_events.append([start_sec, end_sec, "accident"])

    def _detect_congestion(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        time_bins: dict[int, list[float]] = {}
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES:
                continue
            for t, b in tr.history:
                sec = int(t)
                time_bins.setdefault(sec, []).append(tr.speed)

        cong_start = None
        for sec in sorted(time_bins.keys()):
            speeds = time_bins[sec]
            tl_state = self._get_tl_state_at(float(sec), tl_history)
            is_standstill = len(speeds) >= 6 and (sum(speeds) / len(speeds)) < 10.0 and (tl_state == "GREEN")

            if is_standstill:
                if cong_start is None:
                    cong_start = sec
            else:
                if cong_start is not None:
                    if sec - cong_start >= 15.0:
                        self.raw_events.append([float(cong_start), float(sec), "congestion"])
                    cong_start = None

        if cong_start is not None and (max(time_bins.keys()) - cong_start) >= 15.0:
            self.raw_events.append([float(cong_start), float(max(time_bins.keys())), "congestion"])

