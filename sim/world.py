#!/usr/bin/env python3
"""
Virtual 2D world for PiCar-X training simulation.

Everything here is plain geometry + kinematics with no I/O, so it is
directly unit-testable. Units and conventions:

  - distances in centimeters, angles in radians internally
  - x grows right, y grows up, heading is CCW from the +x axis
  - a POSITIVE steering angle (as commanded by the robot's modules,
    e.g. {"direction": "turn", "angle": 30}) means turning RIGHT,
    which in this CCW world decreases the heading
  - camera pan follows the same convention as the robot's code:
    negative pan = look left (SCAN_PAN_ANGLES comment in field_agent)

The robot is modelled as a circle of BODY_RADIUS_CM for collision, an
Ackermann bicycle model (wheelbase from the real config's 95mm) for
motion, a front-bumper ultrasonic raycast for distance, and cliff
regions checked at a point just ahead of the front bumper.

Obstacle sensor model (mirrors the real robot's failure modes, which
is what field_agent's layered defenses exist to handle):

  height="normal"    ultrasonic sees it; vision tracks it iff labelled
  height="overhead"  ultrasonic MISSES it (beam passes under - the
                     counter-lip case); vision reports it on the
                     class-agnostic "overhead" channel
  ultrasonic=False   explicit override, e.g. a glass wall or a black
                     sound-absorbing object the sensor can't echo off;
                     with no label either, only physical collision +
                     stuck detection can find it

Walls (label None) collide and echo, but the SSD detector has no
"wall" class so vision never lists them - same as reality.
"""
import math

BODY_RADIUS_CM = 11.0        # collision circle around the chassis center
WHEELBASE_CM = 9.5           # matches config.json kinematics.wheelbase_mm = 95
ULTRA_MAX_RANGE_CM = 300.0   # beyond this the sensor reports max range
ULTRA_CONE_HALF_DEG = 10.0   # beam half-width; min distance across the cone wins
ULTRA_RAYS = 5               # rays cast across the cone
CLIFF_LOOKAHEAD_CM = 8.0     # grayscale sensors sit just ahead of the bumper
MAX_STEER_DEG = 35.0         # servo hard limit


def _wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------

class Obstacle:
    """A circle or axis-aligned rect with sensor-visibility attributes."""

    def __init__(self, shape, label=None, height="normal", ultrasonic=None,
                 size_k=35.0, **geom):
        assert shape in ("circle", "rect"), f"unknown shape {shape}"
        self.shape = shape
        self.label = label            # SSD class label, or None (untracked)
        self.height = height          # "normal" | "overhead"
        # Default: overhead things are invisible to the low bumper beam.
        self.ultrasonic = (height != "overhead") if ultrasonic is None else ultrasonic
        # Bounding-box-area calibration constant for synthetic vision:
        # area_ratio = (size_k / distance_cm)^2. 35.0 matches the real
        # config's steering.area_distance_k, so the robot's area->distance
        # estimate is exact for a default-sized object.
        self.size_k = float(size_k)
        if shape == "circle":
            self.x, self.y, self.r = float(geom["x"]), float(geom["y"]), float(geom["r"])
        else:
            self.x, self.y = float(geom["x"]), float(geom["y"])   # lower-left corner
            self.w, self.h = float(geom["w"]), float(geom["h"])

    # ---- geometry ----

    def center(self):
        if self.shape == "circle":
            return (self.x, self.y)
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def distance_to_point(self, px, py):
        """Distance from a point to this shape's boundary (0 inside)."""
        if self.shape == "circle":
            return max(0.0, math.hypot(px - self.x, py - self.y) - self.r)
        dx = max(self.x - px, 0.0, px - (self.x + self.w))
        dy = max(self.y - py, 0.0, py - (self.y + self.h))
        return math.hypot(dx, dy)

    def collides_circle(self, cx, cy, radius):
        return self.distance_to_point(cx, cy) < radius

    def ray_intersect(self, ox, oy, dx, dy):
        """Smallest t >= 0 with (ox,oy) + t*(dx,dy) on the shape, or None."""
        if self.shape == "circle":
            return _ray_circle(ox, oy, dx, dy, self.x, self.y, self.r)
        return _ray_rect(ox, oy, dx, dy, self.x, self.y, self.w, self.h)

    def occludes_segment(self, ax, ay, bx, by):
        """Does this shape block the open sight line from A to B?
        (Used for vision occlusion; B is the target's center so an 'own
        boundary' hit at the far end must not count.)"""
        ddx, ddy = bx - ax, by - ay
        seg_len = math.hypot(ddx, ddy)
        if seg_len <= 1e-9:
            return False
        t = self.ray_intersect(ax, ay, ddx / seg_len, ddy / seg_len)
        return t is not None and t < seg_len - 1e-6


def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    fx, fy = ox - cx, oy - cy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    for t in ((-b - sq) / 2.0, (-b + sq) / 2.0):
        if t >= 0:
            return t
    return None


def _ray_rect(ox, oy, dx, dy, rx, ry, rw, rh):
    tmin, tmax = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, rx, rx + rw), (oy, dy, ry, ry + rh)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin


class CliffRegion:
    """A rectangular drop-off (stairwell, table edge past this line)."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


# ---------------------------------------------------------------------
# Robot + world state
# ---------------------------------------------------------------------

class Robot:
    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)     # radians CCW
        self.cam_pan_deg = 0.0            # negative = left (robot convention)
        self.cam_tilt_deg = 0.0
        # Actual achieved motion last step (post-collision), for
        # synthetic scene_motion and odometry-style metrics.
        self.actual_speed_cm_s = 0.0
        self.actual_yaw_rate = 0.0        # rad/s

    def front_point(self, ahead_cm=BODY_RADIUS_CM):
        return (self.x + ahead_cm * math.cos(self.heading),
                self.y + ahead_cm * math.sin(self.heading))

    def camera_axis(self):
        """World-frame bearing of the camera's optical axis (radians)."""
        return self.heading - math.radians(self.cam_pan_deg)


class World:
    """Owns the physical truth: robot pose, obstacles, cliffs, battery."""

    def __init__(self, bounds, robot, obstacles=None, cliffs=None, physics=None):
        self.bounds = bounds                       # (x_min, y_min, x_max, y_max)
        self.robot = robot
        self.obstacles = list(obstacles or [])
        self.cliffs = list(cliffs or [])
        p = physics or {}
        # speed command units (0-100) -> cm/s. Default puts cruise
        # speed 25 at ~21 cm/s, a realistic PiCar-X pace.
        self.speed_to_cm_s = float(p.get("speed_to_cm_s", 0.85))
        self.battery_v = float(p.get("battery_start_v", 7.6))
        self.battery_idle_drain = float(p.get("battery_idle_drain_v_s", 0.0002))
        self.battery_drive_drain = float(p.get("battery_drive_drain_v_s", 0.0010))

        # Episode-level physical truth accumulated by step():
        self.collision_contact = False    # currently pushing against something
        self.collision_events = 0         # distinct contact episodes
        self.fell_off_cliff = False
        self.distance_travelled_cm = 0.0
        self.visited_cells = set()        # 20cm grid coverage
        self.sim_time = 0.0
        self._add_boundary_walls()

    def _add_boundary_walls(self):
        x0, y0, x1, y1 = self.bounds
        t = 4.0  # wall thickness
        for geom in ((x0 - t, y0 - t, (x1 - x0) + 2 * t, t),      # bottom
                     (x0 - t, y1, (x1 - x0) + 2 * t, t),          # top
                     (x0 - t, y0 - t, t, (y1 - y0) + 2 * t),      # left
                     (x1, y0 - t, t, (y1 - y0) + 2 * t)):         # right
            self.obstacles.append(Obstacle(
                "rect", label=None, height="normal",
                x=geom[0], y=geom[1], w=geom[2], h=geom[3]))

    # ---------- physics ----------

    def step(self, speed_units, steer_deg, dt):
        """Advance the robot dt seconds with the given (smoothed) motor
        speed and steering angle. Handles Ackermann motion, collision
        blocking, cliff falls, coverage, battery drain."""
        self.sim_time += dt
        r = self.robot
        v = speed_units * self.speed_to_cm_s              # signed cm/s
        steer = math.radians(max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_deg)))

        # Battery drains faster under load.
        load = abs(speed_units) / 25.0
        self.battery_v -= (self.battery_idle_drain
                           + self.battery_drive_drain * load) * dt
        self.battery_v = max(0.0, self.battery_v)

        if abs(v) < 1e-9:
            r.actual_speed_cm_s = 0.0
            r.actual_yaw_rate = 0.0
            self.collision_contact = False
            return

        # Ackermann bicycle model; positive steer = right turn = CW.
        yaw_rate = -(v / WHEELBASE_CM) * math.tan(steer)
        new_heading = _wrap_angle(r.heading + yaw_rate * dt)
        mid_heading = r.heading + yaw_rate * dt / 2.0     # midpoint integration
        nx = r.x + v * dt * math.cos(mid_heading)
        ny = r.y + v * dt * math.sin(mid_heading)

        if self._pose_collides(nx, ny):
            # Wheels push against the object: no displacement, no yaw.
            # (A blocked wheel can't swing the chassis either.)
            if not self.collision_contact:
                self.collision_events += 1
            self.collision_contact = True
            r.actual_speed_cm_s = 0.0
            r.actual_yaw_rate = 0.0
            return

        self.collision_contact = False
        moved = math.hypot(nx - r.x, ny - r.y)
        self.distance_travelled_cm += moved
        r.x, r.y, r.heading = nx, ny, new_heading
        r.actual_speed_cm_s = v
        r.actual_yaw_rate = yaw_rate
        self.visited_cells.add((int(r.x // 20), int(r.y // 20)))

        if any(c.contains(r.x, r.y) for c in self.cliffs):
            self.fell_off_cliff = True

    def _pose_collides(self, x, y):
        return any(o.collides_circle(x, y, BODY_RADIUS_CM) for o in self.obstacles)

    # ---------- sensors (physical truth queries) ----------

    def ultrasonic_read(self):
        """Distance (cm) from the front bumper along the robot heading,
        min across a narrow cone, over ultrasonic-visible obstacles
        only. Returns ULTRA_MAX_RANGE_CM when nothing echoes."""
        r = self.robot
        ox, oy = r.front_point(BODY_RADIUS_CM * 0.9)
        best = ULTRA_MAX_RANGE_CM
        half = math.radians(ULTRA_CONE_HALF_DEG)
        for i in range(ULTRA_RAYS):
            ang = r.heading - half + (2 * half) * i / (ULTRA_RAYS - 1)
            dx, dy = math.cos(ang), math.sin(ang)
            for o in self.obstacles:
                if not o.ultrasonic:
                    continue
                t = o.ray_intersect(ox, oy, dx, dy)
                if t is not None and t < best:
                    best = t
        return best

    def cliff_ahead(self):
        """Would the front grayscale sensors read 'cliff' right now?"""
        px, py = self.robot.front_point(BODY_RADIUS_CM + CLIFF_LOOKAHEAD_CM)
        return any(c.contains(px, py) for c in self.cliffs)

    def visible_objects(self, fov_deg=62.0, max_range_cm=350.0):
        """Objects the camera can currently see: labelled (SSD-tracked)
        and overhead masses, with occlusion by any physical obstacle.
        Returns list of dicts: {obstacle, index, distance_cm,
        rel_bearing_rad (positive = LEFT of camera axis), kind}."""
        r = self.robot
        cam = r.camera_axis()
        half_fov = math.radians(fov_deg) / 2.0
        out = []
        for idx, o in enumerate(self.obstacles):
            kind = ("overhead" if o.height == "overhead"
                    else ("tracked" if o.label else None))
            if kind is None:
                continue
            cx, cy = o.center()
            d = math.hypot(cx - r.x, cy - r.y)
            if d < 1e-6 or d > max_range_cm:
                continue
            bearing = math.atan2(cy - r.y, cx - r.x)
            rel = _wrap_angle(bearing - cam)
            if abs(rel) > half_fov:
                continue
            if self._occluded(r.x, r.y, cx, cy, ignore=o):
                continue
            out.append({"obstacle": o, "index": idx,
                        "distance_cm": d, "rel_bearing_rad": rel,
                        "kind": kind})
        return out

    def _occluded(self, ax, ay, bx, by, ignore):
        for o in self.obstacles:
            if o is ignore:
                continue
            if o.occludes_segment(ax, ay, bx, by):
                return True
        return False

    def battery_state(self):
        # Same thresholds as safety_daemon.py.
        return {"voltage": round(self.battery_v, 2),
                "low": self.battery_v < 6.7,
                "critical": self.battery_v < 6.4}
