#!/usr/bin/env python3
"""
PiCar-X training simulation runner.

Runs the robot's UNMODIFIED decision modules (field_agent, arbiter,
coach, ...) against a virtual world, so the robot's mind can practice
obstacle avoidance without hardware. See TRAINING_SIMULATION_DESIGN.md.

Examples:

  # one scenario, default modules (arbiter, field_agent, event_logger, coach)
  python3 run_training.py scenarios/open_room.json

  # a whole suite, accumulating coach learning across scenarios
  python3 run_training.py scenarios/*.json --policy-dir training_data

  # baseline without the coach (canned evasion only), for comparison
  python3 run_training.py scenarios/box_corner.json --no-coach

  # 3x faster than real time, bulk-training a whole suite quietly
  python3 run_training.py scenarios/*.json --speedf 3 --quiet \
      --policy-dir training_data

  # reproducible run with more modules
  python3 run_training.py scenarios/corridor.json --seed 7 \
      --modules arbiter,field_agent,event_logger,coach,explorer

Each episode writes runs/<scenario>-<timestamp>/ with metrics.json,
per-module logs, and the sandboxed events.db the modules produced.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.launcher import DEFAULT_MODULES, run_scenario   # noqa: E402
from sim.scenario import Scenario                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Run PiCar-X decision modules in a virtual world.")
    ap.add_argument("scenarios", nargs="+", help="scenario JSON file(s)")
    ap.add_argument("--modules", default=",".join(DEFAULT_MODULES),
                    help="comma-separated module list "
                         f"(default: {','.join(DEFAULT_MODULES)})")
    ap.add_argument("--duration", type=float, default=None,
                    help="override scenario duration_sec")
    ap.add_argument("--seed", default=None,
                    help="seed the modules' random for reproducible wander")
    ap.add_argument("--out", default="runs", help="output root (default: runs)")
    coach_group = ap.add_mutually_exclusive_group()
    coach_group.add_argument("--coach", dest="coach", action="store_true",
                             default=None,
                             help="force the coach on (add it to --modules)")
    coach_group.add_argument("--no-coach", dest="coach", action="store_false",
                             help="run without the coach; field_agent falls "
                                  "back to its canned evasion behavior")
    ap.add_argument("--policy-dir", default=None,
                    help="persistent dir for coach_policy.json so learning "
                         "accumulates across runs (default: per-run sandbox)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the start header and end summary "
                         "(default: live decision/veto/coach trace + heartbeat)")
    ap.add_argument("--progress-interval", type=float, default=5.0,
                    help="seconds between status heartbeat lines (default: 5)")
    ap.add_argument("--speedf", type=float, default=1.0, metavar="N",
                    help="run faster than real time by factor N (e.g. 2 = 2x). "
                         "Dilates the whole system's clock in lockstep so "
                         "behavior stays faithful; 2-6x is safe, higher starts "
                         "to starve the modules of CPU. Combine with --quiet "
                         "and parallel invocations for bulk training.")
    args = ap.parse_args()
    if args.speedf < 1.0:
        ap.error("--speedf must be >= 1.0")

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    if args.coach is True and "coach" not in modules:
        modules.append("coach")
    elif args.coach is False:
        modules = [m for m in modules if m != "coach"]
    print(f"modules: {', '.join(modules)}"
          + ("" if "coach" in modules else "   (coach disabled)")
          + (f"   [{args.speedf:g}x speed]" if args.speedf != 1.0 else ""))

    results = []
    for path in args.scenarios:
        scenario = Scenario.load(path)
        if args.duration:
            scenario.duration_sec = args.duration
        print(f"\n>>> scenario: {scenario.name}"
              + (f" - {scenario.description}" if scenario.description else ""))
        summary = run_scenario(scenario, out_root=args.out, modules=modules,
                               seed=args.seed, policy_dir=args.policy_dir,
                               verbose=not args.quiet,
                               progress_interval=args.progress_interval,
                               speedf=args.speedf)
        results.append(summary)

    if len(results) > 1:
        ok = [r for r in results if r]
        print("\n=== suite summary ===")
        print(f"  episodes: {len(ok)}")
        print(f"  total collisions: {sum(r['collisions'] for r in ok)}")
        print(f"  total vetoes: {sum(r['vetoes'] for r in ok)}")
        print(f"  total distance: {sum(r['distance_travelled_cm'] for r in ok):.0f}cm")
        goals = [r["goal_reached_at_sec"] for r in ok
                 if r["goal_reached_at_sec"] is not None]
        if goals:
            print(f"  goals reached: {len(goals)}")


if __name__ == "__main__":
    main()
