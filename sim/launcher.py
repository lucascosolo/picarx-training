#!/usr/bin/env python3
"""
Launcher: stands up one full training episode.

Order of operations:
  1. start the minibus server (the "MQTT broker")
  2. build the Simulator (virtual world + safety socket + sensor
     publisher + metrics), from the scenario
  3. spawn the requested robot modules as subprocesses via
     sim.run_module, each logging to <run_dir>/logs/<name>.log
  4. run the episode (auto-says "explore", watches for goal/cliff/
     battery end conditions, enforces duration)
  5. tear everything down, write metrics.json + the scenario copy
     into the run directory

The run directory is the complete record of an episode: metrics,
module stdout, and the sandboxed events.db / spatial.db the modules
wrote - reflection or pattern mining can chew on those later exactly
like on-robot data.
"""
import json
import os
import shutil
import socket as socketlib
import subprocess
import sys
import time

from .minibus import BusServer, BusClient
from .metrics import print_summary, save_summary
from .simulator import Simulator

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODULES = ("arbiter", "field_agent", "event_logger", "coach")
MODULE_START_ORDER = ("arbiter", "event_logger", "coach", "explorer",
                      "location_graph", "goal_manager", "reflection",
                      "field_agent")   # field_agent last: everything ready first


def _port_free(port, host="127.0.0.1"):
    with socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _pick_bus_port():
    for port in (1883, 18830, 18831):
        if _port_free(port):
            return port
    raise SystemExit("No free port for the bus (tried 1883, 18830, 18831)")


class EpisodeRunner:
    def __init__(self, scenario, run_dir, modules=DEFAULT_MODULES,
                 seed=None, policy_dir=None, verbose=True,
                 progress_interval=5.0):
        self.scenario = scenario
        self.run_dir = os.path.abspath(run_dir)
        self.modules = [m for m in MODULE_START_ORDER if m in modules]
        unknown = set(modules) - set(MODULE_START_ORDER)
        if unknown:
            raise SystemExit(f"Unknown modules: {sorted(unknown)}")
        self.seed = seed
        self.policy_dir = policy_dir
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.procs = {}
        self.bus_server = None

    def run(self):
        data_dir = os.path.join(self.run_dir, "data")
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        port = _pick_bus_port()
        self.bus_server = BusServer(port=port)
        self.bus_server.start()

        # A training-private safety socket (keyed by bus port so parallel
        # runs don't clash), NOT the real robot's /tmp/picarx_safety.sock.
        # This keeps the sim from touching a running safety_daemon and,
        # crucially, stops the training arbiter from driving real hardware.
        sock_path = f"/tmp/picarx_train_{port}.sock"

        bus = BusClient(port=port)
        sim = Simulator(self.scenario, bus, socket_path=sock_path,
                        verbose=self.verbose,
                        progress_interval=self.progress_interval)

        env = dict(os.environ)
        env["SIM_BUS_PORT"] = str(port)
        env["SIM_SAFETY_SOCKET"] = sock_path
        # An explicit PICARX_REPO (already in the env copy) is honored and
        # validated by run_module; otherwise run_module locates the repo
        # itself, handling both the dev sibling-checkout and Pi layouts.
        if self.seed is not None:
            env["SIM_SEED"] = str(self.seed)
        if self.policy_dir:
            env["SIM_POLICY_DIR"] = os.path.abspath(self.policy_dir)

        summary = None
        try:
            sim.start()              # bind the safety socket before any
                                     # module tries to connect to it
            self._spawn_modules(data_dir, log_dir, env)
            summary = sim.run_episode()
            summary["seed"] = self.seed
            summary["modules"] = self.modules
        finally:
            self._teardown()
            sim.stop()
            bus.close()
            self.bus_server.stop()

        if summary is not None:
            save_summary(summary, os.path.join(self.run_dir, "metrics.json"))
            if self.scenario.path:
                shutil.copy(self.scenario.path,
                            os.path.join(self.run_dir, "scenario.json"))
        return summary

    # ---------- subprocess management ----------

    def _spawn_modules(self, data_dir, log_dir, env):
        if self.verbose:
            print(f"  spawning modules: {', '.join(self.modules)}", flush=True)
        for name in self.modules:
            log = open(os.path.join(log_dir, f"{name}.log"), "w")
            self.procs[name] = (subprocess.Popen(
                [sys.executable, "-u", "-m", "sim.run_module", name,
                 "--data-dir", data_dir],
                cwd=TRAINING_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT), log)
            time.sleep(0.3)          # stagger connections

        time.sleep(0.7)
        dead = [n for n, (p, _) in self.procs.items() if p.poll() is not None]
        if dead:
            self._teardown()
            for n in dead:
                log_path = os.path.join(log_dir, f"{n}.log")
                print(f"--- {n} failed to start; last log lines: ---")
                with open(log_path) as f:
                    print("".join(f.readlines()[-15:]))
            raise SystemExit(f"Modules failed to start: {dead}")

    def _teardown(self):
        for name, (proc, log) in self.procs.items():
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 3.0
        for name, (proc, log) in self.procs.items():
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        self.procs.clear()


def run_scenario(scenario, out_root="runs", **kwargs):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_root, f"{scenario.name}-{stamp}")
    runner = EpisodeRunner(scenario, run_dir, **kwargs)
    summary = runner.run()
    if summary:
        print_summary(summary)
        print(f"  run dir: {run_dir}")
    return summary
