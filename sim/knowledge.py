#!/usr/bin/env python3
"""
Distill a training suite's raw artifacts into a deployable KNOWLEDGE PACK
that the real robot can import (see picarx/layer_b/import_training.py).

This is the piece that makes training *useful to the robot*: on its own a
run leaves behind a coach policy and a pile of per-episode event logs, but
nothing that packages the transferable lessons for deployment. This module
reads what accumulated in the knowledge dir over a suite and writes two
portable files next to it:

    navigation_facts.json   transferable facts + mined behavioural patterns
    knowledge_pack.json     a manifest: what's inside and how it was trained

Inputs it reads from the knowledge dir:
    coach_policy.json   the coach's learned escape maneuvers (bandit arms),
                        already accumulated across the suite by coach.py
    events.db           every episode's event log, accumulated here by the
                        launcher after each run (see launcher._accumulate_events)

WHAT TRANSFERS, AND WHAT DELIBERATELY DOESN'T
---------------------------------------------
Only knowledge about the *robot itself* is exported, because that is what a
virtual world and the real house genuinely share:

  * escape maneuvers (coach_policy.json)  - how Ackermann steering gets the
    car unstuck. Carried verbatim; the robot's importer merges the win/loss
    records into whatever it has already learned.
  * behavioural patterns (pattern_miner) - "obstacle vetoes come in bursts,
    a single small retreat won't clear it"; "escapes here that start by
    reversing usually work". Keyed by situation-type and veto-code, never by
    a specific place.
  * a couple of first-person escape-tactic tendencies aggregated from the
    coach's own win/loss records ("backing away first works more reliably
    than pushing forward").

Place-specific and spatial memories are NOT exported: the sim's rooms are
not the real house, so "the corner with the sofa vetoes a lot" would be a
lie on the physical robot. The robot rebuilds its map from real sensors.

Pure-Python and API-key-free: pattern mining reuses the robot's own
pattern_miner unchanged, and the escape-tactic aggregation mirrors
reflection.py's self-model logic. Nothing here needs the network or a model.
"""
import json
import os
import sys
import time

# The escape-tactic thresholds below mirror reflection.py's self-model
# constants (SELF_MIN_PULLS_PER_DIRECTION / _RATE_GAP / _TOTAL_PULLS) so a
# tendency has to clear the same evidence bar the robot would apply itself.
MIN_PULLS_PER_DIRECTION = 3   # min tries before comparing two escape directions
MIN_RATE_GAP = 0.15           # min success-rate gap to call one direction better
MIN_TOTAL_PULLS = 5           # min coaching attempts before an overall claim

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# locating the robot repo (so we can reuse its unmodified pattern_miner)
# --------------------------------------------------------------------------

def _resolve_picarx_repo():
    """Reuse run_module's resolver so we agree on where the robot code lives
    (sibling checkout on a dev box, ~ layout on the Pi, or $PICARX_REPO).
    Works both as a package import and when run as a bare script."""
    try:
        from .run_module import _resolve_picarx_repo as resolve
    except ImportError:  # invoked as a plain script, not as sim.knowledge
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_module import _resolve_picarx_repo as resolve
    return resolve()


def _import_pattern_miner(picarx_repo):
    """Import the robot's OWN pattern miner so mined patterns are identical to
    what the robot would compute. pattern_miner only needs stdlib at import
    time, so putting layer_b on the path is enough - no paho/robot_config."""
    layer_b = os.path.join(picarx_repo, "layer_b")
    if layer_b not in sys.path:
        sys.path.insert(0, layer_b)
    import pattern_miner  # noqa: E402
    return pattern_miner


# --------------------------------------------------------------------------
# small fail-soft helpers
# --------------------------------------------------------------------------

def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# coach policy -> escape-tactic facts + inventory
# --------------------------------------------------------------------------

def _aggregate_escape_directions(policy):
    """Sum coaching win/loss across every arm, keyed by the FIRST step's
    movement direction. Mirrors reflection.Reflection._aggregate_escape_directions.
    Returns {direction: [successes, failures]}."""
    agg = {}
    for key, entry in policy.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        for arm in (entry.get("arms") or {}).values():
            steps = arm.get("steps") or []
            if not steps:
                continue
            direction = (steps[0].get("action") or {}).get("direction")
            if not direction:
                continue
            bucket = agg.setdefault(direction, [0, 0])
            bucket[0] += int(arm.get("successes", 0))
            bucket[1] += int(arm.get("failures", 0))
    return agg


def escape_facts(policy):
    """A few transferable, first-person escape-tactic tendencies distilled
    from the coach's own win/loss records. Subject is 'escape tactics' (NOT
    'self') on purpose: the robot's reflection.py fully recomputes the 'self'
    subject each pass and would otherwise wipe an imported self-fact, whereas
    'escape tactics' persists and flows into future coach prompts.

    Returns a list of {subject, fact, confidence, source} dicts; may be empty
    on a robot that simply hasn't been coached much yet - we state only what
    the counts support, never filler."""
    agg = _aggregate_escape_directions(policy)
    facts = []

    def rate(direction):
        s, f = agg.get(direction, [0, 0])
        total = s + f
        return (s / total if total else None), total

    b_rate, b_n = rate("backward")
    f_rate, f_n = rate("forward")
    if (b_n >= MIN_PULLS_PER_DIRECTION and f_n >= MIN_PULLS_PER_DIRECTION
            and b_rate is not None and f_rate is not None
            and abs(b_rate - f_rate) >= MIN_RATE_GAP):
        if b_rate > f_rate:
            facts.append("backing away first before turning gets me unstuck more "
                         "reliably than pushing forward does")
        else:
            facts.append("easing forward gets me unstuck more reliably than "
                         "reversing does")

    total_s = sum(v[0] for v in agg.values())
    total_f = sum(v[1] for v in agg.values())
    total_pulls = total_s + total_f
    if total_pulls >= MIN_TOTAL_PULLS:
        overall = total_s / total_pulls
        if overall >= 0.6:
            facts.append("I can usually work my own way out of a tight spot once "
                         "I try a maneuver I've learned")
        elif overall <= 0.4:
            facts.append("I still get stuck fairly often even when I try what I've "
                         "learned, so I'm cautious in tight spaces")

    return [{"subject": "escape tactics", "fact": text, "confidence": 0.7,
             "source": "training"} for text in facts]


def policy_inventory(policy):
    """Count what's in a coach policy, for the manifest."""
    situations = [k for k in policy if not k.startswith("_")]
    arms = sum(len(policy[k].get("arms") or {}) for k in situations
               if isinstance(policy[k], dict))
    demos = policy.get("_demonstrations")
    return {
        "present": bool(policy),
        "situations": len(situations),
        "arms": arms,
        "demonstrations": len(demos) if isinstance(demos, list) else 0,
    }


# --------------------------------------------------------------------------
# suite summaries -> manifest totals
# --------------------------------------------------------------------------

def _suite_totals(summaries):
    ok = [s for s in summaries if s]
    coach = [s.get("coach") or {} for s in ok]
    goals = [s.get("goal_reached_at_sec") for s in ok
             if s.get("goal_reached_at_sec") is not None]
    return {
        "episodes": len(ok),
        "scenarios": sorted({s.get("scenario") for s in ok if s.get("scenario")}),
        "collisions": sum(s.get("collisions", 0) for s in ok),
        "vetoes": sum(s.get("vetoes", 0) for s in ok),
        "coach_queries": sum(c.get("queries", 0) for c in coach),
        "coach_episodes": sum(c.get("episodes", 0) for c in coach),
        "coach_wins": sum(c.get("wins", 0) for c in coach),
        "distance_cm": round(sum(s.get("distance_travelled_cm", 0) for s in ok), 1),
        "goals_reached": len(goals),
    }


# --------------------------------------------------------------------------
# the one entry point the launcher calls
# --------------------------------------------------------------------------

def consolidate(knowledge_dir, summaries=None, picarx_repo=None, verbose=True):
    """Distill the knowledge dir into navigation_facts.json + knowledge_pack.json.

    knowledge_dir : the accumulating dir holding coach_policy.json + events.db
    summaries     : the episode metrics dicts from the suite (for the manifest)
    picarx_repo   : robot repo root; auto-resolved if None

    Returns the manifest dict. Fail-soft: a missing input just yields an empty
    section rather than raising, so a keyless run (no coach arms) still emits a
    valid pack carrying whatever patterns the veto stream produced."""
    knowledge_dir = os.path.abspath(knowledge_dir)
    os.makedirs(knowledge_dir, exist_ok=True)
    summaries = summaries or []

    policy = _load_json(os.path.join(knowledge_dir, "coach_policy.json"), {})
    if not isinstance(policy, dict):
        policy = {}

    events_db = os.path.join(knowledge_dir, "events.db")
    patterns = []
    if os.path.exists(events_db):
        try:
            miner = _import_pattern_miner(picarx_repo or _resolve_picarx_repo())
            patterns = miner.mine_patterns(events_db)
        except Exception as e:   # pragma: no cover - repo-resolution edge
            if verbose:
                print(f"  (pattern mining skipped: {e})")
    for p in patterns:
        p["source"] = "training"

    facts = escape_facts(policy)

    navigation = {"schema": SCHEMA_VERSION, "facts": facts, "patterns": patterns}
    _write_json(os.path.join(knowledge_dir, "navigation_facts.json"), navigation)

    manifest = {
        "schema": SCHEMA_VERSION,
        "kind": "picarx-training-knowledge-pack",
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training": _suite_totals(summaries),
        "contents": {
            "coach_policy": policy_inventory(policy),
            "navigation_facts": {"facts": len(facts), "patterns": len(patterns)},
        },
    }
    _write_json(os.path.join(knowledge_dir, "knowledge_pack.json"), manifest)

    if verbose:
        inv = manifest["contents"]["coach_policy"]
        print(f"\n=== knowledge pack written to {knowledge_dir} ===")
        print(f"  coach policy: {inv['situations']} situations, {inv['arms']} arms, "
              f"{inv['demonstrations']} demonstrations")
        print(f"  navigation:   {len(facts)} facts, {len(patterns)} patterns")
        print("  deploy it:    python3 layer_b/import_training.py "
              f"{knowledge_dir}   (on the robot)")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Distill a training knowledge dir into a deployable pack.")
    ap.add_argument("knowledge_dir",
                    help="dir holding coach_policy.json + events.db")
    ap.add_argument("--picarx-repo", default=None,
                    help="robot repo root (auto-resolved if omitted)")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    consolidate(args.knowledge_dir, picarx_repo=args.picarx_repo)
