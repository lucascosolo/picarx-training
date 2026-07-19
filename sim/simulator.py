#!/usr/bin/env python3
"""
Simulator core: wires the virtual world to the bus.

One Simulator instance owns, for a single episode:
  - the World built from a Scenario
  - the VirtualSafetyDaemon (Unix socket the real arbiter talks to,
    50Hz smoother+physics loop)
  - the SensorSynthesizer publishing picarx/state/world at the same
    2Hz the real world_state.py uses
  - a MetricsCollector on the bus

It does NOT launch the robot's modules - that's sim/launcher.py - so
tests can drive a Simulator directly with a hand-rolled bus client.
"""
import threading
import time

from .metrics import MetricsCollector
from .sensors import SensorSynthesizer
from .virtual_safety import VirtualSafetyDaemon

WORLD_PUBLISH_HZ = 2         # matches world_state.py PUBLISH_HZ


class Simulator:
    def __init__(self, scenario, bus, socket_path=None):
        self.scenario = scenario
        self.bus = bus
        self.world = scenario.build_world()
        kwargs = {"socket_path": socket_path} if socket_path else {}
        self.safety = VirtualSafetyDaemon(self.world, **kwargs)
        self.sensors = SensorSynthesizer(self.world)
        self.metrics = MetricsCollector(bus)
        self._running = False
        self.end_reason = None

        bus.subscribe("picarx/audio/heard", self.sensors.on_heard)
        bus.subscribe("picarx/action/result", self.sensors.on_action_result)

    # ---------- lifecycle ----------

    def start(self):
        self._running = True
        self.safety.start()
        threading.Thread(target=self._publish_loop, daemon=True,
                         name="sim-world-pub").start()

    def stop(self):
        self._running = False
        self.safety.stop()

    def say(self, text):
        """Inject a spoken command exactly as the STT pipeline would."""
        self.bus.publish("picarx/audio/heard", {"text": text, "ts": time.time()})

    # ---------- episode ----------

    def run_episode(self):
        """Block until the scenario ends. Returns the metrics summary."""
        self.start()
        if self.scenario.auto_explore:
            # Let the modules connect and settle before the start cue.
            time.sleep(2.0)
            self.say("explore")

        deadline = time.time() + self.scenario.duration_sec
        while time.time() < deadline:
            if self.scenario.goal_reached(self.world):
                self.metrics.mark_goal_reached()
                self.end_reason = "goal_reached"
                break
            if self.world.fell_off_cliff:
                self.end_reason = "fell_off_cliff"
                break
            if self.world.battery_state()["critical"]:
                self.end_reason = "battery_critical"
                break
            time.sleep(0.2)
        else:
            self.end_reason = "duration_elapsed"

        self.say("stop")
        time.sleep(1.0)              # let the stop land before teardown
        self.stop()
        summary = self.metrics.snapshot(self.world, self.scenario.name)
        summary["end_reason"] = self.end_reason
        return summary

    # ---------- world-state publishing ----------

    def _publish_loop(self):
        period = 1.0 / WORLD_PUBLISH_HZ
        while self._running:
            snapshot = self.sensors.build_snapshot()
            self.bus.publish("picarx/state/world", snapshot)
            time.sleep(period)
