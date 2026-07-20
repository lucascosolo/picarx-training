"""The knowledge consolidator must distill a training suite into a pack that
carries ONLY transferable robot-dynamics knowledge: escape-tactic tendencies
from the coach's win/loss records, and patterns mined by the robot's own
pattern_miner. Place-specific memory is deliberately never exported."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)

from sim import knowledge  # noqa: E402


def _arm(direction, s, f):
    return (json.dumps([{"action": {"direction": direction}, "duration": 1.0}]),
            {"steps": [{"action": {"direction": direction, "speed": 25},
                        "duration": 1.0}], "successes": s, "failures": f})


def _policy(*situations):
    return {name: {"arms": dict(arms)} for name, arms in situations}


class EscapeFactsTest(unittest.TestCase):
    def test_backward_better_than_forward(self):
        policy = _policy(("k", [_arm("backward", 8, 1), _arm("forward", 1, 6)]))
        facts = knowledge.escape_facts(policy)
        texts = " ".join(f["fact"] for f in facts)
        self.assertIn("backing away first", texts)
        self.assertTrue(all(f["subject"] == "escape tactics" for f in facts))
        self.assertTrue(all(f["source"] == "training" for f in facts))

    def test_forward_better_than_backward(self):
        policy = _policy(("k", [_arm("backward", 1, 6), _arm("forward", 8, 1)]))
        texts = " ".join(f["fact"] for f in knowledge.escape_facts(policy))
        self.assertIn("easing forward", texts)

    def test_low_evidence_yields_nothing(self):
        # only two total pulls per direction, below the thresholds
        policy = _policy(("k", [_arm("backward", 1, 0), _arm("forward", 0, 1)]))
        self.assertEqual(knowledge.escape_facts(policy), [])

    def test_overall_competence_stated_when_mostly_winning(self):
        policy = _policy(("k", [_arm("turn", 8, 1)]))
        texts = " ".join(f["fact"] for f in knowledge.escape_facts(policy))
        self.assertIn("work my own way out", texts)

    def test_cautious_when_mostly_failing(self):
        policy = _policy(("k", [_arm("turn", 1, 8)]))
        texts = " ".join(f["fact"] for f in knowledge.escape_facts(policy))
        self.assertIn("still get stuck", texts)

    def test_reserved_sections_ignored(self):
        policy = {"_demonstrations": [{"steps": []}],
                  "k": {"arms": dict([_arm("backward", 5, 0), _arm("forward", 0, 5)])}}
        # must not crash on the reserved list; still reads the real situation
        self.assertIsInstance(knowledge.escape_facts(policy), list)


class InventoryAndTotalsTest(unittest.TestCase):
    def test_policy_inventory_counts(self):
        policy = _policy(("a", [_arm("backward", 1, 0), _arm("forward", 0, 1)]),
                         ("b", [_arm("turn", 2, 0)]))
        policy["_demonstrations"] = [{}, {}]
        inv = knowledge.policy_inventory(policy)
        self.assertEqual((inv["situations"], inv["arms"], inv["demonstrations"]),
                         (2, 3, 2))

    def test_suite_totals_aggregate(self):
        summaries = [
            {"scenario": "a", "collisions": 2, "vetoes": 5,
             "distance_travelled_cm": 100, "goal_reached_at_sec": 3.0,
             "coach": {"queries": 4, "episodes": 3, "wins": 2}},
            {"scenario": "b", "collisions": 1, "vetoes": 0,
             "distance_travelled_cm": 50, "goal_reached_at_sec": None,
             "coach": {"queries": 1, "episodes": 1, "wins": 1}},
            None,   # a failed episode - must be tolerated
        ]
        t = knowledge._suite_totals(summaries)
        self.assertEqual(t["episodes"], 2)
        self.assertEqual(t["scenarios"], ["a", "b"])
        self.assertEqual(t["collisions"], 3)
        self.assertEqual(t["coach_episodes"], 4)
        self.assertEqual(t["goals_reached"], 1)


class ConsolidateEndToEndTest(unittest.TestCase):
    """The full path: a knowledge dir with a coach policy + accumulated
    events.db in, a navigation pack + manifest out."""

    def setUp(self):
        self.kd = tempfile.mkdtemp()
        # a policy that clearly favours reversing
        policy = _policy(("collision_loop:evasion_loop:obstacle",
                          [_arm("backward", 7, 1), _arm("forward", 1, 5)]))
        with open(os.path.join(self.kd, "coach_policy.json"), "w") as f:
            json.dump(policy, f)
        # events.db with coach episodes (backward first-move, mostly winning)
        db = os.path.join(self.kd, "events.db")
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "ts REAL, topic TEXT, payload_json TEXT)")
        t = 1000.0
        for i in range(6):
            c.execute("INSERT INTO events (ts,topic,payload_json) VALUES (?,?,?)",
                      (t, "picarx/coach/episode", json.dumps({
                          "situation_key": "collision_loop:evasion_loop:obstacle",
                          "steps": [{"action": {"direction": "backward"},
                                     "duration": 1.0}], "success": i != 0})))
            t += 1
        c.commit()
        c.close()

    def test_pack_written_with_transferable_knowledge(self):
        manifest = knowledge.consolidate(
            self.kd, summaries=[{"scenario": "box_corner", "collisions": 1,
                                 "vetoes": 3, "distance_travelled_cm": 500,
                                 "goal_reached_at_sec": None,
                                 "coach": {"queries": 6, "episodes": 6, "wins": 5}}],
            picarx_repo=os.path.join(os.path.dirname(TRAINING_ROOT), "picarx"),
            verbose=False)

        with open(os.path.join(self.kd, "navigation_facts.json")) as fh:
            nav = json.load(fh)
        self.assertEqual(nav["schema"], knowledge.SCHEMA_VERSION)
        self.assertTrue(any("backing away first" in f["fact"] for f in nav["facts"]))
        # the robot's pattern_miner should surface the backward-first-move win
        conds = {p["condition"] for p in nav["patterns"]}
        self.assertIn("stuck:collision_loop:evasion_loop:obstacle", conds)
        self.assertTrue(all(p["source"] == "training" for p in nav["patterns"]))

        # manifest provenance
        self.assertEqual(manifest["kind"], "picarx-training-knowledge-pack")
        self.assertEqual(manifest["contents"]["coach_policy"]["arms"], 2)
        self.assertEqual(manifest["training"]["scenarios"], ["box_corner"])
        self.assertTrue(os.path.exists(os.path.join(self.kd, "knowledge_pack.json")))

    def test_no_place_specific_facts_leak(self):
        knowledge.consolidate(self.kd, verbose=False,
                              picarx_repo=os.path.join(
                                  os.path.dirname(TRAINING_ROOT), "picarx"))
        with open(os.path.join(self.kd, "navigation_facts.json")) as fh:
            nav = json.load(fh)
        subjects = {f["subject"] for f in nav["facts"]}
        # every exported fact is generic robot-dynamics, subject 'escape tactics'
        self.assertTrue(subjects <= {"escape tactics"})

    def test_empty_dir_still_writes_valid_pack(self):
        empty = tempfile.mkdtemp()
        manifest = knowledge.consolidate(empty, verbose=False)
        self.assertTrue(os.path.exists(os.path.join(empty, "navigation_facts.json")))
        with open(os.path.join(empty, "navigation_facts.json")) as fh:
            nav = json.load(fh)
        self.assertEqual(nav["facts"], [])
        self.assertEqual(manifest["contents"]["coach_policy"]["present"], False)


if __name__ == "__main__":
    unittest.main()
