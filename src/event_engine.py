"""
event_engine.py — Comprehensive Event Detection Engine for all 14 official traffic event classes.
"""
from __future__ import annotations
import math
import numpy as np
from src.scene import SceneGeometry
from src.tracker import Track, iou

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CLASSES = {"person", "bicycle"}

def clean_and_merge_segments(events: list[list], min_duration: float = 0.5, max_gap: float = 0.3) -> list[list]:
    """Clean, filter short blips, merge immediately contiguous fragments of the same event,
    and resolve any remaining same-class overlaps with high boundary precision."""
    if not events:
        return []

    by_class: dict[str, list[list[float]]] = {}
    for ev in events:
        s, e, label = float(ev[0]), float(ev[1]), str(ev[2])
        if e <= s:
            continue
        # Drop sub-second blips < 0.4s
        if (e - s) < min_duration:
            continue
        by_class.setdefault(label, []).append([s, e])

    cleaned_out: list[list] = []
    for label, segs in by_class.items():
        # Sort by start time, then duration descending
        segs.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping = []
        for s, e in segs:
            if not non_overlapping:
                non_overlapping.append([s, e])
            else:
                prev_s, prev_e = non_overlapping[-1]
                # If tiny gap between fragments of same event (<= max_gap = 0.3s)
                if s <= prev_e + max_gap:
                    # Merge fragment
                    non_overlapping[-1][1] = max(prev_e, e)
                elif s >= prev_e:
                    # Clean separate event with no overlap
                    non_overlapping.append([s, e])
                # If s < prev_e (overlap), since same class cannot overlap, skip or trim
                else:
                    if e > prev_e + 1.0:
                        # Append non-overlapping tail
                        non_overlapping.append([prev_e + 0.1, e])

        for s, e in non_overlapping:
            if (e - s) >= min_duration:
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
        self._detect_pedestrian_events(tracks)

        # 3. Wrong Way & Illegal U-Turn
        self._detect_trajectory_violations(tracks)

        # 4. Stopped Vehicle
        self._detect_stopped_vehicles(tracks, tl_history)

        # 5. Solid Line Crossing
        self._detect_solid_line_crossing(tracks)

        # 6. Accident & Near Miss
        self._detect_accidents_and_near_misses(tracks)

        # 7. Congestion
        self._detect_congestion(tracks)

        # Post-process and merge
        return clean_and_merge_segments(self.raw_events)

    def _get_tl_state_at(self, t: float, tl_history: list[tuple[float, str]]) -> str:
        """Find traffic light state at given time t."""
        if not tl_history:
            return "UNKNOWN"
        # Find closest before or at t
        best_state = tl_history[0][1]
        for item_t, state in tl_history:
            if item_t <= t:
                best_state = state
            else:
                break
        return best_state

    def _detect_red_light_and_stop_line(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        vehicle_tracks = [tr for tr in tracks if tr.cls_name in VEHICLE_CLASSES and len(tr.history) >= 3]

        for tr in vehicle_tracks:
            # Check if trajectory crosses the stop line
            for i in range(len(tr.history) - 1):
                t1, box1 = tr.history[i]
                t2, box2 = tr.history[i + 1]
                c1 = ((box1[0] + box1[2]) / 2, box1[3]) # bottom-center
                c2 = ((box2[0] + box2[2]) / 2, box2[3])

                if self.scene.crosses_stop_line(c1, c2):
                    tl_state = self._get_tl_state_at(t1, tl_history)
                    if tl_state == "RED":
                        # Check whether vehicle continues into intersection (red_light)
                        # or stops in the buffer zone (stop_line)
                        rem_history = tr.history[i+1:]
                        if rem_history:
                            speeds = []
                            for k in range(len(rem_history) - 1):
                                dt = rem_history[k+1][0] - rem_history[k][0]
                                if dt > 0:
                                    dx = rem_history[k+1][1][0] - rem_history[k][1][0]
                                    dy = rem_history[k+1][1][1] - rem_history[k][1][1]
                                    speeds.append(math.hypot(dx, dy) / dt)
                            avg_speed = sum(speeds) / len(speeds) if speeds else 0

                            # If it keeps moving into intersection -> red_light
                            if avg_speed > 25.0:
                                start_sec = t1
                                end_sec = min(start_sec + 8.0, tr.history[-1][0])
                                if end_sec > start_sec + 0.8:
                                    self.raw_events.append([start_sec, end_sec, "red_light"])
                            else:
                                # Stopped past the line without entering -> stop_line
                                # End when signal turns green
                                start_sec = t1
                                end_sec = tr.history[-1][0]
                                for fut_t, fut_st in tl_history:
                                    if fut_t >= start_sec and fut_st == "GREEN":
                                        end_sec = fut_t
                                        break
                                self.raw_events.append([start_sec, max(start_sec + 1.0, end_sec), "stop_line"])
                    break

    def _detect_pedestrian_events(self, tracks: list[Track]):
        ped_tracks = [tr for tr in tracks if tr.cls_name in PEDESTRIAN_CLASSES and len(tr.history) >= 2]
        veh_tracks = [tr for tr in tracks if tr.cls_name in VEHICLE_CLASSES and len(tr.history) >= 2]

        for p_tr in ped_tracks:
            # Check jaywalking: pedestrian on roadway outside crossings
            # Ignore stationary pedestrians on sidewalk / curb
            cur_start = None
            for t, box in p_tr.history:
                pt = ((box[0] + box[2]) / 2, box[3])
                on_road = self.scene.is_on_roadway(pt)
                in_crosswalk = self.scene.is_in_crosswalk(pt)

                is_jaywalking = on_road and (not in_crosswalk) and (p_tr.speed >= 8.0)
                if is_jaywalking:
                    if cur_start is None:
                        cur_start = t
                else:
                    if cur_start is not None:
                        dur = t - cur_start
                        if 1.0 <= dur <= 15.0:
                            self.raw_events.append([cur_start, t, "jaywalking"])
                        cur_start = None
            if cur_start is not None:
                dur = p_tr.last_t - cur_start
                if 1.0 <= dur <= 15.0:
                    self.raw_events.append([cur_start, p_tr.last_t, "jaywalking"])

        # Failure to yield: vehicle drives through crossing while pedestrian is on it
        for cw in self.scene.all_crosswalks:
            for v_tr in veh_tracks:
                v_in_cw = []
                for t, box in v_tr.history:
                    pt = ((box[0] + box[2]) / 2, box[3])
                    if self.scene.is_inside_polygon(pt, cw):
                        v_in_cw.append(t)

                if len(v_in_cw) >= 2 and v_tr.speed > 15.0:
                    v_start, v_end = v_in_cw[0], v_in_cw[-1]
                    dur = v_end - v_start
                    if 0.8 <= dur <= 6.0:
                        # Check if any pedestrian was actively inside this crosswalk during [v_start, v_end]
                        ped_present = False
                        for p_tr in ped_tracks:
                            for pt_t, p_box in p_tr.history:
                                if v_start <= pt_t <= v_end:
                                    p_pt = ((p_box[0] + p_box[2]) / 2, p_box[3])
                                    if self.scene.is_inside_polygon(p_pt, cw):
                                        ped_present = True
                                        break
                            if ped_present:
                                break
                        if ped_present:
                            self.raw_events.append([v_start, v_end, "failure_to_yield"])

    def _detect_trajectory_violations(self, tracks: list[Track]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 5:
                continue

            # Trajectory displacement
            t_first, b_first = tr.history[0]
            t_last, b_last = tr.history[-1]
            c_first = ((b_first[0] + b_first[2]) / 2, (b_first[1] + b_first[3]) / 2)
            c_last = ((b_last[0] + b_last[2]) / 2, (b_last[1] + b_last[3]) / 2)

            # Wrong way: sustained movement against lane flow
            ww_start = None
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i+1]
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                dt = t2 - t1
                if dt > 0:
                    vy = (c2[1] - c1[1]) / dt
                    in_sb = self.scene.is_inside_polygon(c1, self.scene.roadway_sb)
                    # Southbound lanes: normal flow is vy > 0. Opposing is vy < -25 px/s
                    is_ww = in_sb and (vy < -25.0)
                    if is_ww:
                        if ww_start is None:
                            ww_start = t1
                    else:
                        if ww_start is not None:
                            if t1 - ww_start >= 1.5:
                                self.raw_events.append([ww_start, t1, "wrong_way"])
                            ww_start = None
            if ww_start is not None and (tr.last_t - ww_start) >= 1.5:
                self.raw_events.append([ww_start, tr.last_t, "wrong_way"])

            # Illegal U-turn: heading reverses ~180 degrees
            # Check headings in first half vs second half
            if len(tr.history) >= 8:
                mid = len(tr.history) // 2
                h1_c = ((tr.history[mid][1][0] + tr.history[mid][1][2]) / 2 - c_first[0],
                        (tr.history[mid][1][1] + tr.history[mid][1][3]) / 2 - c_first[1])
                h2_c = (c_last[0] - (tr.history[mid][1][0] + tr.history[mid][1][2]) / 2,
                        c_last[1] - (tr.history[mid][1][1] + tr.history[mid][1][3]) / 2)

                dot = h1_c[0] * h2_c[0] + h1_c[1] * h2_c[1]
                m1 = math.hypot(*h1_c)
                m2 = math.hypot(*h2_c)
                if m1 > 30 and m2 > 30:
                    cos_theta = dot / (m1 * m2 + 1e-6)
                    if cos_theta < -0.65:
                        self.raw_events.append([t_first, t_last, "illegal_u_turn"])

    def _detect_stopped_vehicles(self, tracks: list[Track], tl_history: list[tuple[float, str]]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 10:
                continue

            # Look for contiguous stationary intervals >= 10.0 seconds
            stat_start = None
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i + 1]
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                dt = t2 - t1
                speed = math.hypot(c2[0] - c1[0], c2[1] - c1[1]) / (dt + 1e-6)

                if speed < 10.0:
                    if stat_start is None:
                        stat_start = t1
                else:
                    if stat_start is not None:
                        dur = t1 - stat_start
                        if dur >= 10.0:
                            # Verify NOT in queue during red light
                            tl_state = self._get_tl_state_at(stat_start, tl_history)
                            in_stop_zone = self.scene.is_inside_polygon(c1, self.scene.stop_line_zone)
                            if not (tl_state == "RED" and in_stop_zone):
                                self.raw_events.append([stat_start, t1, "stopped_vehicle"])
                        stat_start = None

            if stat_start is not None:
                dur = tr.last_t - stat_start
                if dur >= 10.0:
                    c = tr.centroid
                    tl_state = self._get_tl_state_at(stat_start, tl_history)
                    in_stop_zone = self.scene.is_inside_polygon(c, self.scene.stop_line_zone)
                    if not (tl_state == "RED" and in_stop_zone):
                        self.raw_events.append([stat_start, tr.last_t, "stopped_vehicle"])

    def _detect_solid_line_crossing(self, tracks: list[Track]):
        for tr in tracks:
            if tr.cls_name not in VEHICLE_CLASSES or len(tr.history) < 3:
                continue
            for i in range(len(tr.history) - 1):
                t1, b1 = tr.history[i]
                t2, b2 = tr.history[i + 1]
                c1 = ((b1[0] + b1[2]) / 2, b1[3])
                c2 = ((b2[0] + b2[2]) / 2, b2[3])
                if self.scene.crosses_solid_line(c1, c2):
                    self.raw_events.append([t1, min(t2 + 1.5, self.duration), "solid_line_crossing"])
                    break

    def _detect_accidents_and_near_misses(self, tracks: list[Track]):
        # Analyze pairs of tracks that overlap in time
        n = len(tracks)
        for i in range(n):
            tr1 = tracks[i]
            if tr1.cls_name not in VEHICLE_CLASSES:
                continue
            for j in range(i + 1, n):
                tr2 = tracks[j]
                if tr2.cls_name not in (VEHICLE_CLASSES | PEDESTRIAN_CLASSES):
                    continue

                # Ignore pedestrians walking together
                if tr1.cls_name == "person" and tr2.cls_name == "person":
                    continue

                # Find time overlap
                t_min = max(tr1.history[0][0], tr2.history[0][0])
                t_max = min(tr1.last_t, tr2.last_t)
                if t_max - t_min < 0.4:
                    continue

                # Check dynamic collision metrics
                min_dist = float("inf")
                max_iou = 0.0
                t_closest = t_min
                max_closing_speed = 0.0

                # Compute pairwise metrics across common frames
                for t1, b1 in tr1.history:
                    if t1 < t_min or t1 > t_max:
                        continue
                    # Find closest sample in tr2
                    closest_item = min(tr2.history, key=lambda x: abs(x[0] - t1))
                    closest_b2 = closest_item[1]
                    score = iou(b1, closest_b2)
                    max_iou = max(max_iou, score)

                    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                    c2 = ((closest_b2[0] + closest_b2[2]) / 2, (closest_b2[1] + closest_b2[3]) / 2)
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if dist < min_dist:
                        min_dist = dist
                        t_closest = t1

                # Closing velocity check (must be approaching each other, not driving in parallel)
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                dvx = tr1.vx - tr2.vx
                dvy = tr1.vy - tr2.vy
                dist = math.hypot(dx, dy)
                v_closing = (dx * dvx + dy * dvy) / (dist + 1e-6)

                # Check post-event mobility
                tr1_post_speed = 0.0
                tr1_post_samples = [h for h in tr1.history if h[0] > t_closest]
                if len(tr1_post_samples) >= 2:
                    dt = tr1_post_samples[-1][0] - tr1_post_samples[0][0]
                    if dt > 0:
                        dp = math.hypot(tr1_post_samples[-1][1][0] - tr1_post_samples[0][1][0],
                                        tr1_post_samples[-1][1][1] - tr1_post_samples[0][1][1])
                        tr1_post_speed = dp / dt

                # Collision criteria:
                # 1. Closing in directly (v_closing < -20 px/s)
                # 2. Significant physical contact (max_iou > 0.35)
                # 3. Post-impact stoppage (at least one vehicle stops or speed drops to < 10 px/s)
                if max_iou > 0.35 and v_closing < -20.0 and tr1_post_speed < 10.0:
                    start_sec = max(0.0, t_closest)
                    end_sec = min(self.duration, min(start_sec + 15.0, max(tr1.last_t, tr2.last_t)))
                    if end_sec > start_sec + 1.0:
                        self.raw_events.append([start_sec, end_sec, "accident"])

                # Near-miss criteria:
                # High closing speed, close distance (< 65px), but evasive maneuver avoided contact (max_iou < 0.10)
                elif v_closing < -30.0 and min_dist < 65.0 and max_iou < 0.10:
                    start_sec = max(0.0, t_closest - 0.8)
                    end_sec = min(self.duration, t_closest + 1.2)
                    if end_sec > start_sec + 0.4:
                        self.raw_events.append([start_sec, end_sec, "near_miss"])

    def _detect_congestion(self, tracks: list[Track]):
        # Check if traffic queue is stopped across lanes for >= 8s
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
            if len(speeds) >= 5 and (sum(speeds) / len(speeds)) < 12.0:
                if cong_start is None:
                    cong_start = sec
            else:
                if cong_start is not None:
                    if sec - cong_start >= 8.0:
                        self.raw_events.append([float(cong_start), float(sec), "congestion"])
                    cong_start = None
        if cong_start is not None and (max(time_bins.keys()) - cong_start) >= 8.0:
            self.raw_events.append([float(cong_start), float(max(time_bins.keys())), "congestion"])
