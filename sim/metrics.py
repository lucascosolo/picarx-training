#!/usr/bin/env python3
"""
Episode metrics: what happened during a training run, summarized.

Physical truth (collisions, distance, coverage, falls) comes from the
World object; behavioral truth (vetoes, evasions, coach activity,
decisions) is collected off the same bus topics the robot itself uses,
so the numbers describe exactly what the modules experienced.
"""
import json
import threading
import time


class MetricsCollector:
    """Subscribe to the bus; call snapshot(world) at episode end."""

    def __init__(self, bus):
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.action_results = 0
        self.vetoes = 0
        self.veto_reasons = {}
        self.decisions = {}           # kind -> count
        self.evade_triggers = {}      # trigger -> count
        self.coach_queries = 0
        self.coach_suggestions = 0
        self.coach_episodes = []      # {situation_key, success}
        self.announcements = []       # spoken lines, a readable behavior trace
        self.goal_reached_at = None

        bus.subscribe("picarx/action/result", self._on_action_result)
        bus.subscribe("picarx/decision", self._on_decision)
        bus.subscribe("picarx/coach/query", self._on_coach_query)
        bus.subscribe("picarx/coach/suggestion", self._on_coach_suggestion)
        bus.subscribe("picarx/coach/episode", self._on_coach_episode)
        bus.subscribe("picarx/audio/speak", self._on_speak)

    # ---------- callbacks ----------

    def _on_action_result(self, payload):
        with self.lock:
            self.action_results += 1
            result = payload.get("result") or {}
            if result.get("status") == "vetoed":
                self.vetoes += 1
                code = result.get("reason_code", "unknown")
                self.veto_reasons[code] = self.veto_reasons.get(code, 0) + 1

    def _on_decision(self, payload):
        with self.lock:
            kind = payload.get("kind", "unknown")
            self.decisions[kind] = self.decisions.get(kind, 0) + 1
            if kind == "evade":
                trig = (payload.get("choice") or {}).get("trigger", "unknown")
                self.evade_triggers[trig] = self.evade_triggers.get(trig, 0) + 1

    def _on_coach_query(self, payload):
        with self.lock:
            self.coach_queries += 1

    def _on_coach_suggestion(self, payload):
        with self.lock:
            self.coach_suggestions += 1

    def _on_coach_episode(self, payload):
        with self.lock:
            self.coach_episodes.append({
                "situation_key": payload.get("situation_key"),
                "success": payload.get("success"),
            })

    def _on_speak(self, payload):
        with self.lock:
            text = payload.get("text")
            if text:
                self.announcements.append(
                    {"t": round(time.time() - self.started_at, 1), "text": text})

    def mark_goal_reached(self):
        with self.lock:
            if self.goal_reached_at is None:
                self.goal_reached_at = time.time() - self.started_at

    # ---------- summary ----------

    def snapshot(self, world, scenario_name=""):
        with self.lock:
            wall = time.time() - self.started_at
            coach_wins = sum(1 for e in self.coach_episodes if e["success"])
            return {
                "scenario": scenario_name,
                "wall_time_sec": round(wall, 1),
                "sim_time_sec": round(world.sim_time, 1),
                "distance_travelled_cm": round(world.distance_travelled_cm, 1),
                "coverage_cells_20cm": len(world.visited_cells),
                "collisions": world.collision_events,
                "fell_off_cliff": world.fell_off_cliff,
                "battery_v_end": round(world.battery_v, 2),
                "action_results": self.action_results,
                "vetoes": self.vetoes,
                "veto_reasons": dict(self.veto_reasons),
                "decisions": dict(self.decisions),
                "evade_triggers": dict(self.evade_triggers),
                "coach": {
                    "queries": self.coach_queries,
                    "suggestions": self.coach_suggestions,
                    "episodes": len(self.coach_episodes),
                    "wins": coach_wins,
                },
                "goal_reached_at_sec": (round(self.goal_reached_at, 1)
                                        if self.goal_reached_at is not None else None),
                "announcements": self.announcements[-40:],
            }


def print_summary(summary):
    c = summary
    print(f"\n=== {c['scenario']} - episode summary ===")
    print(f"  time: {c['wall_time_sec']}s   distance: {c['distance_travelled_cm']}cm   "
          f"coverage: {c['coverage_cells_20cm']} cells")
    print(f"  collisions: {c['collisions']}   vetoes: {c['vetoes']} {c['veto_reasons']}"
          + ("   FELL OFF CLIFF" if c["fell_off_cliff"] else ""))
    print(f"  decisions: {c['decisions']}")
    if c["evade_triggers"]:
        print(f"  evade triggers: {c['evade_triggers']}")
    coach = c["coach"]
    if coach["queries"]:
        print(f"  coach: {coach['queries']} queries, {coach['episodes']} episodes, "
              f"{coach['wins']} wins")
    if c["goal_reached_at_sec"] is not None:
        print(f"  goal reached at {c['goal_reached_at_sec']}s")


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
