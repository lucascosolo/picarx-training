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
import sqlite3
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

# The robot's live learning we can seed a training run from, so the sim
# refines REAL experience instead of a blank slate. coach_policy.json is the
# one that matters (the coach reads it via SIM_POLICY_DIR and refines it in
# place); events.db seeds the end-of-suite pattern-mining corpus with real
# history; semantic.db is carried for completeness. spatial.db is deliberately
# NOT seeded - place memories don't transfer (the sim's rooms aren't the house).
SEED_FILES = ("coach_policy.json", "events.db", "semantic.db")


def _copy_seed_file(src, dst):
    """Copy one seed file into place. A SQLite db (.db) is snapshotted through
    the backup API - a consistent read even while Layer B is mid-write on the
    robot, landing a standalone db with no -wal/-shm sidecar to carry along.
    JSON is copied verbatim (coach.py writes coach_policy.json atomically, so a
    plain copy can't tear). Fail-soft: if the snapshot can't run, fall back to a
    byte copy rather than abort the whole seed."""
    if dst.endswith(".db"):
        try:
            src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            try:
                dst_conn = sqlite3.connect(dst)
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
            finally:
                src_conn.close()
            return
        except sqlite3.Error:
            pass  # not a usable sqlite db (or locked) - fall back to a raw copy
    shutil.copy2(src, dst)


def seed_knowledge_dir(seed_from, knowledge_dir, force=False, verbose=True):
    """Seed a training knowledge dir from the robot's live data BEFORE a run.

    Copies each of SEED_FILES that exists in `seed_from` into `knowledge_dir`.
    Idempotent by default: a file already present in `knowledge_dir` is KEPT
    (so a resumed/repeated suite builds on its accumulated progress rather than
    being reset to the seed each time); pass force=True to overwrite. Returns
    the list of basenames actually (re)written.

    Pure file-ops - no bus, no hardware, no subprocesses - so it's unit-testable
    and safe to call on the robot with Layer B still running."""
    seed_from = os.path.abspath(seed_from)
    knowledge_dir = os.path.abspath(knowledge_dir)
    if not os.path.isdir(seed_from):
        raise SystemExit(f"--seed-from dir not found: {seed_from}")
    if seed_from == knowledge_dir:
        raise SystemExit("--seed-from and the knowledge dir are the same path")
    os.makedirs(knowledge_dir, exist_ok=True)

    copied, kept = [], []
    for name in SEED_FILES:
        src = os.path.join(seed_from, name)
        dst = os.path.join(knowledge_dir, name)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and not force:
            kept.append(name)
            continue
        if force:
            # drop any stale sidecars so an overwritten db can't inherit a WAL
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(dst + suffix)
                except FileNotFoundError:
                    pass
        _copy_seed_file(src, dst)
        copied.append(name)

    if verbose:
        if copied:
            print(f"  seed: copied {', '.join(copied)} from {seed_from}")
        if kept:
            print(f"  seed: kept existing {', '.join(kept)} in knowledge dir "
                  "(use --seed-force to overwrite)")
        if not copied and not kept:
            print(f"  seed: nothing to copy from {seed_from} "
                  f"(none of {', '.join(SEED_FILES)} present)")
    return copied




class EpisodeRunner:
    def __init__(self, scenario, run_dir, modules=DEFAULT_MODULES,
                 seed=None, knowledge_dir=None, verbose=True,
                 progress_interval=5.0, speedf=1.0):
        self.scenario = scenario
        self.run_dir = os.path.abspath(run_dir)
        self.modules = [m for m in MODULE_START_ORDER if m in modules]
        unknown = set(modules) - set(MODULE_START_ORDER)
        if unknown:
            raise SystemExit(f"Unknown modules: {sorted(unknown)}")
        self.seed = seed
        # The persistent knowledge dir accumulates the deployable learning
        # ACROSS episodes: the coach writes coach_policy.json here, and after
        # each run we fold this episode's event log into a shared events.db
        # here so a suite-wide pattern mine has real volume to work with.
        self.knowledge_dir = os.path.abspath(knowledge_dir) if knowledge_dir else None
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.speedf = max(1.0, float(speedf))
        self.procs = {}
        self.bus_server = None

    def run(self):
        data_dir = os.path.join(self.run_dir, "data")
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # port 0 -> OS picks a free ephemeral port; read the real one back.
        # Parallel runs each get a unique port with no collision or race.
        self.bus_server = BusServer(port=0)
        self.bus_server.start()
        port = self.bus_server.port

        # A training-private safety socket (keyed by bus port so parallel
        # runs don't clash), NOT the real robot's /tmp/picarx_safety.sock.
        # This keeps the sim from touching a running safety_daemon and,
        # crucially, stops the training arbiter from driving real hardware.
        sock_path = f"/tmp/picarx_train_{port}.sock"

        bus = BusClient(port=port)
        sim = Simulator(self.scenario, bus, socket_path=sock_path,
                        verbose=self.verbose,
                        progress_interval=self.progress_interval,
                        speedf=self.speedf)

        env = dict(os.environ)
        env["SIM_BUS_PORT"] = str(port)
        env["SIM_SAFETY_SOCKET"] = sock_path
        if self.speedf != 1.0:
            # Dilate the modules' clocks to match the sim, sharing the
            # Simulator's exact epoch so timestamps stay comparable.
            env["SIM_SPEEDF"] = repr(self.speedf)
            env["SIM_CLOCK_T0"] = repr(sim._t0)
            env["SIM_CLOCK_M0"] = repr(time.monotonic())
        # An explicit PICARX_REPO (already in the env copy) is honored and
        # validated by run_module; otherwise run_module locates the repo
        # itself, handling both the dev sibling-checkout and Pi layouts.
        if self.seed is not None:
            env["SIM_SEED"] = str(self.seed)
        if self.knowledge_dir:
            # coach.py reads SIM_POLICY_DIR (see run_module) - point it at the
            # knowledge dir so the policy accumulates there alongside events.db.
            env["SIM_POLICY_DIR"] = self.knowledge_dir

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
            # Fold this episode's events into the suite-wide corpus so the
            # end-of-suite pattern mine sees every episode, not just the last.
            if self.knowledge_dir:
                self._accumulate_events(os.path.join(data_dir, "events.db"))
        return summary

    def _accumulate_events(self, run_events_db):
        """Append this run's event rows into the knowledge dir's shared
        events.db. Runs AFTER teardown, so event_logger's connection is closed
        and its WAL is applied when we attach the file read-through. Fail-soft:
        a lock (parallel invocations sharing a dir) or a missing db just skips
        this episode's contribution rather than aborting the run."""
        if not os.path.exists(run_events_db):
            return
        dest = os.path.join(self.knowledge_dir, "events.db")
        try:
            conn = sqlite3.connect(dest)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS events ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
                    "topic TEXT NOT NULL, payload_json TEXT NOT NULL)")
                conn.execute("ATTACH DATABASE ? AS src", (run_events_db,))
                conn.execute(
                    "INSERT INTO events (ts, topic, payload_json) "
                    "SELECT ts, topic, payload_json FROM src.events")
                conn.commit()
                conn.execute("DETACH DATABASE src")
            finally:
                conn.close()
        except sqlite3.Error as e:
            if self.verbose:
                print(f"  (could not accumulate events into knowledge dir: {e})")

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
    # Include the pid so parallel runs started in the same second (e.g.
    # `for i in ...; do run_training.py ... & done`) get distinct dirs.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_root, f"{scenario.name}-{stamp}-{os.getpid()}")
    runner = EpisodeRunner(scenario, run_dir, **kwargs)
    summary = runner.run()
    if summary:
        print_summary(summary)
        print(f"  run dir: {run_dir}")
    return summary
