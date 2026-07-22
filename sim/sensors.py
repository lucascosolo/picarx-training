#!/usr/bin/env python3
"""
Synthetic sensor snapshot builder.

Produces picarx/state/world payloads in EXACTLY the shape world_state.py
publishes on the real robot (see its module docstring), so field_agent,
the coach and every other consumer read the virtual world through the
same fields, staleness flags and derived signals they see in reality.

Replaced pipeline: distance_sensor.py + vision_basic.py + world_state.py.
That means this module must also reproduce world_state's DERIVED data:
  - per-object approach_rate / approaching (bbox area growth per second
    while roughly centered, threshold 0.12)
  - overhead approach_rate / approaching (same trick on the overhead mass)
and vision_basic's synthesis choices:
  - area_ratio from distance: area = (size_k / distance_cm)^2, the
    exact inverse of steering_controller's distance estimate
    (distance ~= area_distance_k / sqrt(area)), so the robot's
    perception math round-trips faithfully
  - center_offset in pixels, positive = right of frame center
  - close_object: class-agnostic "something fills most of the frame"
  - scene_motion: mean-abs-thumbnail-diff stand-in; ~6+ while actually
    moving, near zero while static or pushing against an obstacle
    (which is what field_agent's stuck detection keys on)
"""
import math
import time

FRAME_W = 640
FRAME_H = 480
CAM_FOV_DEG = 62.0

APPROACH_RATE_THRESHOLD = 0.12       # world_state.py
APPROACH_CENTER_FRACTION = 0.5
CLOSE_OBJECT_AREA = 0.45             # frame-filler bar
MAX_AREA_RATIO = 0.95

# scene_motion calibration: real values sit ~6-10 while driving,
# < 1 while parked (field_agent's STUCK_MOTION_THRESHOLD is 3.0).
MOTION_PER_CM_S = 0.38
MOTION_PER_DEG_S = 0.10
MOTION_FLOOR = 0.3                   # sensor noise while static


class SensorSynthesizer:
    def __init__(self, world, clock=None):
        self.world = world
        self._clock = clock or time.time   # sim-time under --speedf
        self._prev_areas = {}        # obstacle index -> (area, t)
        self._prev_overhead = None   # (area, t)
        self._first_seen = {}        # obstacle index -> ts
        self.last_heard = {"text": None, "updated_at": None}
        self.last_action = {"source": None, "action": None,
                            "result": None, "updated_at": None}
        self._prev_collision_events = 0   # to edge-detect a new collision (impact)

    # Wire these to bus subscriptions so the snapshot mirrors
    # last_heard / last_action exactly like world_state does.
    def on_heard(self, payload):
        self.last_heard = {"text": payload.get("text"),
                           "updated_at": self._clock()}

    def on_action_result(self, payload):
        self.last_action = {"source": payload.get("source"),
                            "action": payload.get("action"),
                            "result": payload.get("result"),
                            "updated_at": self._clock()}

    # ---------- snapshot ----------

    def build_snapshot(self):
        now = self._clock()
        w = self.world
        robot = w.robot

        items = []
        overhead = None
        close_object = False
        seen_indices = set()

        for vis in w.visible_objects(fov_deg=CAM_FOV_DEG):
            o = vis["obstacle"]
            d = vis["distance_cm"]
            area = min(MAX_AREA_RATIO, (o.size_k / max(d, 1.0)) ** 2)
            # positive center_offset = right of center; rel_bearing is
            # positive-left in the CCW world frame.
            half_fov = math.radians(CAM_FOV_DEG) / 2.0
            frac = math.tan(-vis["rel_bearing_rad"]) / math.tan(half_fov)
            center_offset = max(-FRAME_W / 2.0,
                                min(FRAME_W / 2.0, frac * (FRAME_W / 2.0)))

            if vis["kind"] == "overhead":
                overhead = self._overhead_payload(area, now)
                continue

            idx = vis["index"]
            seen_indices.add(idx)
            rate = self._approach_rate(idx, area, now)
            approaching = (rate > APPROACH_RATE_THRESHOLD
                           and abs(center_offset) <
                           (FRAME_W / 2.0) * APPROACH_CENTER_FRACTION)
            if idx not in self._first_seen:
                self._first_seen[idx] = now
            box = math.sqrt(area * FRAME_W * FRAME_H)
            items.append({
                "id": idx,
                "label": o.label,
                "confidence": 0.82,
                "x": int(FRAME_W / 2.0 + center_offset - box / 2.0),
                "y": int(FRAME_H / 2.0 - box / 2.0),
                "w": int(box),
                "h": int(box),
                "frame_width": FRAME_W,
                "frame_height": FRAME_H,
                "area_ratio": round(area, 4),
                "center_offset": round(center_offset, 1),
                "first_seen": self._first_seen[idx],
                "last_seen": now,
                "approach_rate": round(rate, 4),
                "approaching": approaching,
            })
            if area >= CLOSE_OBJECT_AREA and abs(center_offset) < FRAME_W * 0.3:
                close_object = True

        # Drop growth history for objects that left the view, exactly so
        # a re-sighting doesn't compute a bogus rate across the gap
        # (world_state rebuilds its record dict every message too).
        self._prev_areas = {i: v for i, v in self._prev_areas.items()
                            if i in seen_indices}
        if overhead is None:
            self._prev_overhead = None

        distance = w.ultrasonic_read()
        scene_motion = self._scene_motion(robot)
        battery = w.battery_state()

        return {
            "timestamp": now,
            "face": {"detected": False, "updated_at": None, "stale": True},
            "person": {"name": None, "confidence": None,
                       "updated_at": None, "stale": True},
            "distance_cm": round(distance, 1),
            "distance_stale": False,
            "objects": {
                "items": items,
                "close_object": close_object,
                "overhead": overhead,
                "scene_motion": round(scene_motion, 2),
                "stale": False,
            },
            "battery": {**battery, "updated_at": now, "stale": False},
            "imu": self._imu_payload(now),
            "last_heard": {**self.last_heard,
                           "stale": self._stale(self.last_heard["updated_at"], 15.0)},
            "last_action": dict(self.last_action),
        }

    def _imu_payload(self, now):
        """Synthetic head-frame IMU block, matching the real world_state.py imu
        shape so field_agent's IMU consumer works in the sim too. moving/rotation
        come from the true body motion; a NEW collision this window is the impact
        (the jolt a real accelerometer feels when it hits something the ranged
        sensors missed). Head-mounted, so head_pose rides along. Flat sim floor
        -> never tilted."""
        r = self.world.robot
        events = getattr(self.world, "collision_events", 0)
        impact = events > self._prev_collision_events
        self._prev_collision_events = events
        yaw_dps = math.degrees(r.actual_yaw_rate)
        moving = abs(r.actual_speed_cm_s) > 1.0 or abs(yaw_dps) > 3.0
        return {
            "moving": bool(moving),
            "impact": bool(impact),
            "tilted": False,
            "rotation_rate_dps": round(abs(yaw_dps), 2),
            "tilt_from_rest_deg": 0.0,
            "body_tilt_deg": 0.0,
            "accel": {"x": 0.0, "y": 0.0, "z": 9.807},
            "gyro": {"x": 0.0, "y": 0.0, "z": round(yaw_dps, 2)},
            "temperature_c": 30.0,
            "head_pose": {"pan": r.cam_pan_deg, "tilt": r.cam_tilt_deg},
            "calibrated": True,
            "updated_at": now,
            "stale": False,
        }

    # ---------- derived signals ----------

    def _approach_rate(self, idx, area, now):
        prev = self._prev_areas.get(idx)
        self._prev_areas[idx] = (area, now)
        if prev is None:
            return 0.0
        dt = now - prev[1]
        if dt <= 0:
            return 0.0
        return (area - prev[0]) / dt

    def _overhead_payload(self, area, now):
        rate = 0.0
        if self._prev_overhead is not None:
            dt = now - self._prev_overhead[1]
            if dt > 0:
                rate = (area - self._prev_overhead[0]) / dt
        self._prev_overhead = (area, now)
        return {
            "area_ratio": round(area, 4),
            "y_center_frac": 0.22,      # upper-frame mass, as vision reports
            "approach_rate": round(rate, 4),
            "approaching": rate > APPROACH_RATE_THRESHOLD,
        }

    def _scene_motion(self, robot):
        v = abs(robot.actual_speed_cm_s)
        yaw_deg_s = abs(math.degrees(robot.actual_yaw_rate))
        return MOTION_FLOOR + v * MOTION_PER_CM_S + yaw_deg_s * MOTION_PER_DEG_S

    def _stale(self, updated_at, threshold):
        return updated_at is None or (self._clock() - updated_at) > threshold
