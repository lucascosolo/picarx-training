#!/usr/bin/env python3
"""
Virtual safety daemon - drop-in replacement for safety/safety_daemon.py
serving the same Unix socket protocol against the virtual world.

The real arbiter.py (and anything else that queries the socket) runs
UNMODIFIED against this: it listens on /tmp/picarx_safety.sock, accepts
one JSON action/query per connection, and answers with the same reply
shapes and - critically - the same STABLE reason_code values the
learning layer keys on: obstacle | cliff | reverse_limit | unknown.

Faithfully reproduced from the real daemon:
  - is_safe(): only FORWARD needs sensor checks; forward is vetoed
    inside SAFE_DISTANCE_CM (15) or over a cliff; backward is bounded
    by MAX_CONTINUOUS_REVERSE_SEC (2.0, no rear sensor); stop/turn/look
    are never vetoed; any non-reverse resets the reverse timer.
  - a veto triggers emergency stop (snap everything to zero).
  - MotionSmoother: 50Hz ramp of speed (step 2.0) and steering angle
    (step 5.0) toward their targets, with direction reversals snapping
    through zero instead of decelerating first.
  - camera pan/tilt clamped to CAM_PAN_RANGE / CAM_TILT_RANGE.
  - queries: battery_status -> {"voltage","low","critical"};
    distance -> {"distance_cm"}.

The 50Hz smoother tick doubles as the physics tick: each cycle the
smoothed (speed, angle) pair is fed to World.step().
"""
import json
import os
import socket
import threading
import time

from .world import ULTRA_MAX_RANGE_CM

SOCKET_PATH = os.environ.get("SIM_SAFETY_SOCKET", "/tmp/picarx_safety.sock")

SAFE_DISTANCE_CM = 15
MAX_CONTINUOUS_REVERSE_SEC = 2.0
CAM_PAN_RANGE = (-80, 80)
CAM_TILT_RANGE = (-30, 60)

SMOOTHER_HZ = 50
SPEED_STEP = 2.0
ANGLE_STEP = 5.0


class VirtualSafetyDaemon:
    def __init__(self, world, time_scale=1.0, socket_path=SOCKET_PATH):
        self.world = world
        self.time_scale = float(time_scale)
        self.socket_path = socket_path
        self.lock = threading.Lock()      # guards smoother state + world

        self.target_speed = 0.0
        self.current_speed = 0.0
        self.target_angle = 0.0
        self.current_angle = 0.0

        self._reverse_since = None
        self._running = False
        self._server = None
        self.veto_count = 0
        self.veto_reasons = {}            # reason_code -> count

    # ---------- lifecycle ----------

    def start(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o666)
        self._server.listen(8)
        self._server.settimeout(0.5)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True,
                         name="vsafety-accept").start()
        threading.Thread(target=self._physics_loop, daemon=True,
                         name="vsafety-physics").start()

    def stop(self):
        self._running = False
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            os.remove(self.socket_path)
        except OSError:
            pass

    # ---------- smoother + physics (50Hz, replaces MotionSmoother) ----------

    def _physics_loop(self):
        period = 1.0 / SMOOTHER_HZ
        dt = period * self.time_scale
        while self._running:
            with self.lock:
                if self.target_speed * self.current_speed < 0:
                    self.current_speed = 0.0     # reversal snaps through zero
                if self.current_speed < self.target_speed:
                    self.current_speed = min(self.current_speed + SPEED_STEP,
                                             self.target_speed)
                elif self.current_speed > self.target_speed:
                    self.current_speed = max(self.current_speed - SPEED_STEP,
                                             self.target_speed)
                if self.current_angle < self.target_angle:
                    self.current_angle = min(self.current_angle + ANGLE_STEP,
                                             self.target_angle)
                elif self.current_angle > self.target_angle:
                    self.current_angle = max(self.current_angle - ANGLE_STEP,
                                             self.target_angle)
                self.world.step(self.current_speed, self.current_angle, dt)
            time.sleep(period)

    def _emergency_stop(self):
        # Caller holds no lock; mirrors motion.emergency_stop().
        with self.lock:
            self.target_speed = 0.0
            self.current_speed = 0.0

    # ---------- safety rules (mirrors is_safe) ----------

    def _is_safe(self, action):
        direction = action.get("direction")

        if direction == "backward":
            now = time.monotonic()
            if self._reverse_since is None:
                self._reverse_since = now
            if now - self._reverse_since > MAX_CONTINUOUS_REVERSE_SEC / self.time_scale:
                return False, "reverse time limit (no rear sensor)"
            return True, "ok"

        self._reverse_since = None

        if direction in ("stop", "turn", "look"):
            return True, "ok"

        with self.lock:
            distance = self.world.ultrasonic_read()
            cliff = self.world.cliff_ahead()

        if direction == "forward" and 0 < distance < SAFE_DISTANCE_CM:
            return False, f"obstacle at {round(distance)}cm"
        if direction == "forward" and cliff:
            return False, "cliff detected"
        return True, "ok"

    # ---------- action execution (mirrors execute) ----------

    def _execute(self, action):
        d = action.get("direction")
        speed = action.get("speed", 30)
        with self.lock:
            if d == "forward":
                self.target_speed = speed
            elif d == "backward":
                self.target_speed = -speed
            elif d == "stop":
                self.target_speed = 0
            elif d == "turn":
                self.target_angle = action.get("angle", 0)
            elif d == "look":
                pan = max(CAM_PAN_RANGE[0],
                          min(CAM_PAN_RANGE[1], int(action.get("pan", 0))))
                tilt = max(CAM_TILT_RANGE[0],
                           min(CAM_TILT_RANGE[1], int(action.get("tilt", 0))))
                self.world.robot.cam_pan_deg = pan
                self.world.robot.cam_tilt_deg = tilt

    # ---------- socket handling (mirrors main) ----------

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle(conn)
            except Exception as e:
                try:
                    conn.sendall(json.dumps({"status": "error",
                                             "detail": str(e)}).encode())
                except OSError:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn):
        data = conn.recv(1024)
        if not data:
            return
        action = json.loads(data.decode())

        if action.get("query") == "battery_status":
            with self.lock:
                state = self.world.battery_state()
            conn.sendall(json.dumps(state).encode())
            return
        if action.get("query") == "distance":
            with self.lock:
                d = self.world.ultrasonic_read()
            conn.sendall(json.dumps({"distance_cm": round(d, 1)}).encode())
            return

        safe, reason = self._is_safe(action)
        if safe:
            self._execute(action)
            conn.sendall(json.dumps({"status": "executed"}).encode())
        else:
            self._emergency_stop()
            code = ("obstacle" if reason.startswith("obstacle")
                    else "cliff" if "cliff" in reason
                    else "reverse_limit" if "reverse" in reason
                    else "unknown")
            self.veto_count += 1
            self.veto_reasons[code] = self.veto_reasons.get(code, 0) + 1
            conn.sendall(json.dumps({"status": "vetoed", "reason": reason,
                                     "reason_code": code}).encode())
