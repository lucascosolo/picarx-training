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
    def __init__(self, scenario, bus, socket_path=None, verbose=False,
                 progress_interval=5.0, speedf=1.0):
        self.scenario = scenario
        self.bus = bus
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.speedf = max(1.0, float(speedf))
        # Sim-time clock shared with the module subprocesses (same epoch,
        # same scale), so launcher-side timestamps line up with theirs.
        self._t0 = time.time()
        self._clock = (time.time if self.speedf == 1.0
                       else lambda: self._t0 + (time.time() - self._t0) * self.speedf)
        self.world = scenario.build_world()
        kwargs = {"socket_path": socket_path} if socket_path else {}
        # Physics advances speedf x per real second, matching the modules'
        # dilated tick rate so the whole system stays in sim-time lockstep.
        self.safety = VirtualSafetyDaemon(self.world, time_scale=self.speedf,
                                          **kwargs)
        self.sensors = SensorSynthesizer(self.world, clock=self._clock)
        self.metrics = MetricsCollector(bus, live=verbose, clock=self._clock)
        self._running = False
        self.end_reason = None

        bus.subscribe("picarx/audio/heard", self.sensors.on_heard)
        bus.subscribe("picarx/action/result", self.sensors.on_action_result)

    # ---------- lifecycle ----------

    def start(self):
        if self._running:
            return                   # idempotent: launcher starts us early
        self._running = True
        self.safety.start()
        threading.Thread(target=self._publish_loop, daemon=True,
                         name="sim-world-pub").start()

    def stop(self):
        self._running = False
        self.safety.stop()

    def say(self, text):
        """Inject a spoken command exactly as the STT pipeline would."""
        self.bus.publish("picarx/audio/heard", {"text": text, "ts": self._clock()})

    # ---------- episode ----------

    def run_episode(self):
        """Block until the scenario ends. Returns the metrics summary."""
        self.start()
        self._log("modules connecting; settling before the start cue...")
        if self.scenario.auto_explore:
            # Let the modules connect and settle before the start cue.
            time.sleep(2.0)
            self.say("explore")
            self._log('sent start cue: "explore"')

        # The loop watches sim-time (self._clock), so a `total`-second
        # scenario ends after total/speedf real seconds. Poll cadence and
        # the "let stop land" wait stay real so they don't starve the CPU.
        total = self.scenario.duration_sec
        deadline = self._clock() + total
        next_progress = self._clock() + self.progress_interval
        while self._clock() < deadline:
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
            if self.verbose and self._clock() >= next_progress:
                print("  " + self.metrics.progress_line(self.world, total),
                      flush=True)
                next_progress += self.progress_interval
            time.sleep(0.2 / self.speedf)
        else:
            self.end_reason = "duration_elapsed"

        self._log(f"episode ending: {self.end_reason}")
        self.say("stop")
        time.sleep(1.0 / self.speedf)   # let the stop land before teardown
        self.stop()
        summary = self.metrics.snapshot(self.world, self.scenario.name)
        summary["end_reason"] = self.end_reason
        return summary

    def _log(self, msg):
        if self.verbose:
            print(f"  [{self._clock() - self.metrics.started_at:6.1f}s] {msg}",
                  flush=True)

    # ---------- world-state publishing ----------

    def _publish_loop(self):
        # Publish at WORLD_PUBLISH_HZ in SIM-time: real period shrinks by
        # speedf so consumers see 2Hz of sim-time regardless of speed.
        period = 1.0 / (WORLD_PUBLISH_HZ * self.speedf)
        while self._running:
            snapshot = self.sensors.build_snapshot()
            self.bus.publish("picarx/state/world", snapshot)
            time.sleep(period)
