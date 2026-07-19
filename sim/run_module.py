#!/usr/bin/env python3
"""
Bootstrap for running one UNMODIFIED robot module inside the training
environment.

The robot's modules hardcode Pi paths (/home/picarx/layer_b/...) and
import paho-mqtt. This bootstrap keeps the promise that the modules'
CODE is untouched by doing the adaptation from the outside:

  1. puts sim/paho_shim first on sys.path, so `import paho.mqtt.client`
     resolves to the minibus-backed shim (harmless if a real broker
     setup is ever used instead - just drop the shim from the path)
  2. puts <picarx repo>/layer_b and .../layer_b/modules on sys.path,
     exactly like the modules' own sys.path.insert lines expect
  3. imports the module, then rebinds its path CONSTANTS (db files,
     policy cache, map json) into this run's sandbox data directory -
     every one of those globals is read at call time, never baked in
     at import, so attribute patching is sufficient
  4. seeds `random` when SIM_SEED is set, for reproducible wander
  5. instantiates and runs the module's main class, same as its
     `if __name__ == "__main__"` block

Usage:  python -m sim.run_module field_agent --data-dir runs/x/data
"""
import argparse
import os
import random
import sys

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAHO_SHIM = os.path.join(TRAINING_ROOT, "sim", "paho_shim")


def _has_modules(path):
    """A valid picarx repo root has layer_b/modules under it."""
    return os.path.isdir(os.path.join(path, "layer_b", "modules"))


def _resolve_picarx_repo():
    """Locate the picarx repo root (the dir containing layer_b/modules).

    Works both on a dev box, where picarx is a sibling checkout
    (<parent>/picarx), and on the Pi, where layer_b lives directly under
    the home dir (repo root == /home/picarx, beside picarx-training).
    An explicit PICARX_REPO always wins but is still validated so a stale
    value fails loudly instead of causing opaque import errors."""
    parent = os.path.dirname(TRAINING_ROOT)
    explicit = os.environ.get("PICARX_REPO")
    candidates = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    candidates += [
        os.path.join(parent, "picarx"),   # sibling checkout (dev machine)
        parent,                           # layer_b beside training (Pi home)
        os.path.expanduser("~/picarx"),
        os.path.expanduser("~"),
    ]
    for c in candidates:
        if _has_modules(c):
            return c
    if explicit:
        raise SystemExit(
            f"PICARX_REPO={explicit} has no layer_b/modules under it")
    raise SystemExit(
        "Cannot find the picarx repo (a dir containing layer_b/modules).\n"
        "Set PICARX_REPO to the picarx repo root. Tried: "
        + ", ".join(dict.fromkeys(candidates)))


def _setup_paths(repo):
    layer_b = os.path.join(repo, "layer_b")
    modules = os.path.join(layer_b, "modules")
    for p in (modules, layer_b, PAHO_SHIM, TRAINING_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _patch_paths(name, mod, data_dir):
    """Rebind hardcoded Pi paths into the sandbox data dir."""
    os.makedirs(data_dir, exist_ok=True)

    import spatial_store
    spatial_store.DB_DIR = data_dir
    spatial_store.DB_PATH = os.path.join(data_dir, "spatial.db")
    try:
        import semantic_store
        semantic_store.DB_DIR = data_dir
        semantic_store.DB_PATH = os.path.join(data_dir, "semantic.db")
    except ImportError:
        pass

    if name == "field_agent":
        mod.DB_PATH = os.path.join(data_dir, "events.db")
    elif name == "event_logger":
        mod.DB_DIR = data_dir
        mod.DB_PATH = os.path.join(data_dir, "events.db")
    elif name == "coach":
        # The policy cache is the coach's accumulated learning; it may
        # live OUTSIDE the per-run sandbox (SIM_POLICY_DIR) so training
        # runs build on each other and the file can later be deployed
        # to the real robot.
        policy_dir = os.environ.get("SIM_POLICY_DIR", data_dir)
        os.makedirs(policy_dir, exist_ok=True)
        mod.DATA_DIR = policy_dir
        mod.COACH_POLICY_PATH = os.path.join(policy_dir, "coach_policy.json")
    elif name == "explorer":
        mod.MAP_JSON_PATH = os.path.join(data_dir, "uncertainty_map.json")
    elif name == "reflection":
        mod.EVENTS_DB_PATH = os.path.join(data_dir, "events.db")


ENTRIES = {
    "arbiter": lambda m: m.Arbiter().run(),
    "field_agent": lambda m: m.FieldAgent().run(),
    "event_logger": lambda m: m.EventLogger().run(),
    "coach": lambda m: m.Coach().run(),
    "explorer": lambda m: m.Explorer().run(),
    "reflection": lambda m: m.Reflection().run(),
    "location_graph": lambda m: m.LocationGraph().run(),
    "goal_manager": lambda m: m.GoalManager().run(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", choices=sorted(ENTRIES))
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    seed = os.environ.get("SIM_SEED")
    if seed:
        random.seed(f"{seed}:{args.module}")

    repo = _resolve_picarx_repo()
    _setup_paths(repo)

    import importlib
    mod = importlib.import_module(args.module)
    _patch_paths(args.module, mod, os.path.abspath(args.data_dir))

    print(f"[run_module] {args.module} up (repo={repo}, data={args.data_dir})",
          flush=True)
    ENTRIES[args.module](mod)


if __name__ == "__main__":
    main()
