#!/usr/bin/env python3
"""
PiCar-X training simulation runner.

Runs the robot's UNMODIFIED decision modules (field_agent, arbiter,
coach, ...) against a virtual world, so the robot's mind can practice
obstacle avoidance without hardware. See TRAINING_SIMULATION_DESIGN.md.

Examples:

  # one scenario, default modules (arbiter, field_agent, event_logger, coach)
  python3 run_training.py scenarios/open_room.json

  # a whole suite, accumulating learning into a deployable knowledge pack
  python3 run_training.py scenarios/*.json --knowledge-dir training_data

  # baseline without the coach (canned evasion only), for comparison
  python3 run_training.py scenarios/box_corner.json --no-coach

  # 3x faster than real time, bulk-training a whole suite quietly
  python3 run_training.py scenarios/*.json --speedf 3 --quiet \
      --policy-dir training_data

  # reproducible run with more modules
  python3 run_training.py scenarios/corridor.json --seed 7 \
      --modules arbiter,field_agent,event_logger,coach,explorer

  # on-robot idle self-training: refine the robot's OWN live learning, then
  # import it back with --adopt (SIGTERM the run to stop it the instant the
  # robot has real work to do)
  python3 run_training.py scenarios/*.json --seed-from layer_b/data \
      --knowledge-dir /tmp/selftrain --speedf 4 --quiet

Each episode writes runs/<scenario>-<timestamp>/ with metrics.json,
per-module logs, and the sandboxed events.db the modules produced.

With --knowledge-dir, the suite ALSO accumulates a deployable knowledge
pack there (coach_policy.json + navigation_facts.json + knowledge_pack.json)
that the real robot imports with `layer_b/import_training.py`.
"""
import argparse
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.launcher import (DEFAULT_MODULES, run_scenario,   # noqa: E402
                          seed_knowledge_dir)
from sim.scenario import Scenario                        # noqa: E402
from sim import knowledge                                # noqa: E402


def _sigterm_to_keyboardinterrupt(signum, frame):
    raise KeyboardInterrupt()


def _install_sigterm_handler():
    """Make a SIGTERM behave like Ctrl-C so the caller can kill an idle
    self-training run instantly and cleanly. Default SIGTERM would terminate
    this process WITHOUT unwinding, orphaning the module subprocesses and the
    bus thread; raising KeyboardInterrupt instead runs EpisodeRunner.run()'s
    finally (terminate subprocesses, stop the sim, close the bus) on the way
    out. SIGINT already raises KeyboardInterrupt, so both paths converge."""
    signal.signal(signal.SIGTERM, _sigterm_to_keyboardinterrupt)


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
    ap.add_argument("--knowledge-dir", default=None,
                    help="persistent dir the suite accumulates a deployable "
                         "knowledge pack into (coach_policy.json + events.db "
                         "-> navigation_facts.json + knowledge_pack.json). "
                         "Learning builds across runs; the real robot imports "
                         "it with layer_b/import_training.py.")
    ap.add_argument("--policy-dir", default=None,
                    help="deprecated alias for --knowledge-dir (kept working).")
    ap.add_argument("--seed-from", default=None, metavar="DIR",
                    help="before the run, seed the knowledge dir from the "
                         "robot's live data DIR (e.g. layer_b/data): copies "
                         "coach_policy.json + events.db/semantic.db if present, "
                         "so the sim REFINES real learning. Files already in the "
                         "knowledge dir are kept (resumes build on progress). "
                         "The pack's lineage fingerprints this seed, so the "
                         "robot imports it back with --adopt. Requires "
                         "--knowledge-dir.")
    ap.add_argument("--seed-force", action="store_true",
                    help="with --seed-from, overwrite files already present in "
                         "the knowledge dir instead of keeping them.")
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
    # --policy-dir is the old name for the same persistent dir.
    knowledge_dir = args.knowledge_dir or args.policy_dir

    if args.seed_from and not knowledge_dir:
        ap.error("--seed-from needs a destination: pass --knowledge-dir too")

    # A SIGTERM must tear the run down as cleanly as Ctrl-C, so an idle
    # self-training run can be killed the instant the robot wakes up.
    _install_sigterm_handler()

    # Seed the knowledge dir from the robot's live data BEFORE anything reads
    # it, so the coach refines real learning and the seed lineage below is the
    # robot's, not a blank slate.
    if args.seed_from:
        seed_knowledge_dir(args.seed_from, knowledge_dir, force=args.seed_force)

    # Capture the lineage of the seed policy NOW, before the coach refines
    # coach_policy.json in place, so the pack records which robot policy it
    # descends from ('cold' if the dir wasn't seeded). See sim.knowledge.
    seed_lineage = knowledge.seed_lineage(knowledge_dir) if knowledge_dir else None

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    if args.coach is True and "coach" not in modules:
        modules.append("coach")
    elif args.coach is False:
        modules = [m for m in modules if m != "coach"]
    print(f"modules: {', '.join(modules)}"
          + ("" if "coach" in modules else "   (coach disabled)")
          + (f"   [{args.speedf:g}x speed]" if args.speedf != 1.0 else ""))

    results = []
    interrupted = False
    try:
        for path in args.scenarios:
            scenario = Scenario.load(path)
            if args.duration:
                scenario.duration_sec = args.duration
            print(f"\n>>> scenario: {scenario.name}"
                  + (f" - {scenario.description}" if scenario.description else ""))
            summary = run_scenario(scenario, out_root=args.out, modules=modules,
                                   seed=args.seed, knowledge_dir=knowledge_dir,
                                   verbose=not args.quiet,
                                   progress_interval=args.progress_interval,
                                   speedf=args.speedf)
            results.append(summary)
    except KeyboardInterrupt:
        # SIGTERM/Ctrl-C: the in-flight episode already tore itself down in
        # EpisodeRunner.run()'s finally (subprocesses + bus). Skip the rest of
        # the suite and still distill whatever completed below.
        interrupted = True
        print("\nInterrupted - stopped cleanly after the current episode.")

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

    # Distill everything that accumulated into a deployable knowledge pack.
    # Runs for a single scenario too - and even after an interrupt - so
    # `--knowledge-dir` always leaves the robot something to import.
    if knowledge_dir:
        knowledge.consolidate(knowledge_dir, summaries=results,
                              lineage=seed_lineage)
        if interrupted:
            print("Distilled the partial run into the knowledge pack.")


if __name__ == "__main__":
    main()
