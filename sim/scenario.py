#!/usr/bin/env python3
"""
Scenario files: a JSON description of one training environment.

A scenario fully determines the virtual world - obstacle geometry,
robot start pose, cliffs, physics constants, duration and an optional
goal - so a run is reproducible and comparable across code changes.

Schema (all lengths in cm, angles in degrees):

{
  "name": "corridor",
  "description": "human-readable purpose of this scenario",
  "bounds": {"x_min": 0, "y_min": 0, "x_max": 400, "y_max": 300},
  "robot": {"x": 50, "y": 150, "heading_deg": 0},
  "obstacles": [
    {"shape": "circle", "x": 200, "y": 150, "r": 15,
     "label": "chair",              # SSD label; omit/null = untracked (wall-like)
     "height": "normal",            # or "overhead" (ultrasonic-invisible lip)
     "ultrasonic": true,            # optional override, e.g. false = glass
     "size_k": 35.0},               # vision bbox calibration (bigger = looms larger)
    {"shape": "rect", "x": 100, "y": 0, "w": 10, "h": 120, "label": null}
  ],
  "cliffs": [ {"x": 300, "y": 0, "w": 100, "h": 60} ],
  "physics": {"speed_to_cm_s": 0.85, "battery_start_v": 7.6},
  "duration_sec": 90,
  "goal": {"x": 350, "y": 150, "radius": 30},   # optional reach-goal
  "auto_explore": true                          # say "explore" at start
}

Boundary walls are added automatically around "bounds".
"""
import json
import math
import os

from .world import CliffRegion, Obstacle, Robot, World


class Scenario:
    def __init__(self, spec, path=None):
        self.spec = spec
        self.path = path
        self.name = spec.get("name") or (os.path.basename(path or "scenario")
                                         .rsplit(".", 1)[0])
        self.description = spec.get("description", "")
        self.duration_sec = float(spec.get("duration_sec", 60.0))
        self.goal = spec.get("goal")
        self.auto_explore = bool(spec.get("auto_explore", True))
        self._validate()

    @classmethod
    def load(cls, path):
        with open(path) as f:
            spec = json.load(f)
        return cls(spec, path=path)

    def _validate(self):
        b = self.spec.get("bounds")
        if not b or not all(k in b for k in ("x_min", "y_min", "x_max", "y_max")):
            raise ValueError(f"{self.name}: 'bounds' with x_min/y_min/x_max/y_max required")
        if b["x_max"] - b["x_min"] < 60 or b["y_max"] - b["y_min"] < 60:
            raise ValueError(f"{self.name}: bounds must be at least 60x60 cm")
        r = self.spec.get("robot")
        if not r or "x" not in r or "y" not in r:
            raise ValueError(f"{self.name}: 'robot' start pose with x/y required")
        for i, o in enumerate(self.spec.get("obstacles", [])):
            if o.get("shape") not in ("circle", "rect"):
                raise ValueError(f"{self.name}: obstacle {i} needs shape circle|rect")

    def build_world(self):
        b = self.spec["bounds"]
        r = self.spec["robot"]
        robot = Robot(x=r["x"], y=r["y"],
                      heading=math.radians(r.get("heading_deg", 0.0)))
        obstacles = []
        for o in self.spec.get("obstacles", []):
            kwargs = {k: o[k] for k in ("label", "height", "ultrasonic", "size_k")
                      if k in o}
            geom_keys = ("x", "y", "r") if o["shape"] == "circle" else ("x", "y", "w", "h")
            geom = {k: o[k] for k in geom_keys}
            obstacles.append(Obstacle(o["shape"], **kwargs, **geom))
        cliffs = [CliffRegion(c["x"], c["y"], c["w"], c["h"])
                  for c in self.spec.get("cliffs", [])]
        return World(bounds=(b["x_min"], b["y_min"], b["x_max"], b["y_max"]),
                     robot=robot, obstacles=obstacles, cliffs=cliffs,
                     physics=self.spec.get("physics"))

    def goal_reached(self, world):
        if not self.goal:
            return False
        dx = world.robot.x - self.goal["x"]
        dy = world.robot.y - self.goal["y"]
        return math.hypot(dx, dy) <= self.goal.get("radius", 30.0)
