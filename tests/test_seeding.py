"""Seeding a training knowledge dir from the robot's live data.

--seed-from copies coach_policy.json + events.db/semantic.db into a FRESH
knowledge dir so the sim refines REAL learning, but leaves an already-populated
dir alone unless --seed-force is given (so a resumed suite keeps its
accumulated progress). The seeded coach policy is also what the pack's lineage
fingerprints, which is how the robot later knows to import it back with --adopt.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)

from sim import knowledge, launcher  # noqa: E402


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _write_db(path, notes):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT)")
    c.executemany("INSERT INTO events (note) VALUES (?)", [(n,) for n in notes])
    c.commit()
    c.close()


def _db_notes(path):
    c = sqlite3.connect(path)
    try:
        return [r[0] for r in c.execute("SELECT note FROM events ORDER BY id")]
    finally:
        c.close()


class SeedKnowledgeDirTest(unittest.TestCase):
    def setUp(self):
        self.seed = tempfile.mkdtemp()   # the robot's live data dir
        self.kd = tempfile.mkdtemp()     # the training knowledge dir
        _write_json(os.path.join(self.seed, "coach_policy.json"),
                    {"k": {"arms": {"a": {"successes": 7, "failures": 1}}}})
        _write_db(os.path.join(self.seed, "events.db"), ["real-a", "real-b"])
        _write_db(os.path.join(self.seed, "semantic.db"), ["fact-1"])

    def test_fresh_dir_gets_the_whole_seed(self):
        copied = launcher.seed_knowledge_dir(self.seed, self.kd, verbose=False)
        self.assertEqual(set(copied),
                         {"coach_policy.json", "events.db", "semantic.db"})
        with open(os.path.join(self.kd, "coach_policy.json")) as f:
            self.assertEqual(json.load(f)["k"]["arms"]["a"]["successes"], 7)
        # the sqlite snapshot carried the real rows, as a standalone db (no WAL)
        self.assertEqual(_db_notes(os.path.join(self.kd, "events.db")),
                         ["real-a", "real-b"])
        self.assertFalse(os.path.exists(os.path.join(self.kd, "events.db-wal")))

    def test_populated_dir_left_alone_without_force(self):
        # knowledge dir already holds accumulated progress from a prior suite
        _write_json(os.path.join(self.kd, "coach_policy.json"),
                    {"k": {"arms": {"a": {"successes": 99, "failures": 0}}}})
        copied = launcher.seed_knowledge_dir(self.seed, self.kd, verbose=False)
        self.assertNotIn("coach_policy.json", copied)          # kept, not clobbered
        with open(os.path.join(self.kd, "coach_policy.json")) as f:
            self.assertEqual(json.load(f)["k"]["arms"]["a"]["successes"], 99)
        # files the dir DIDN'T already have are still adopted
        self.assertIn("events.db", copied)
        self.assertIn("semantic.db", copied)

    def test_seed_force_overwrites(self):
        _write_json(os.path.join(self.kd, "coach_policy.json"),
                    {"k": {"arms": {"a": {"successes": 99, "failures": 0}}}})
        copied = launcher.seed_knowledge_dir(self.seed, self.kd, force=True,
                                             verbose=False)
        self.assertIn("coach_policy.json", copied)
        with open(os.path.join(self.kd, "coach_policy.json")) as f:
            self.assertEqual(json.load(f)["k"]["arms"]["a"]["successes"], 7)

    def test_absent_seed_files_are_skipped(self):
        bare = tempfile.mkdtemp()
        _write_json(os.path.join(bare, "coach_policy.json"), {"k": {"arms": {}}})
        copied = launcher.seed_knowledge_dir(bare, self.kd, verbose=False)
        self.assertEqual(copied, ["coach_policy.json"])        # only what existed
        self.assertFalse(os.path.exists(os.path.join(self.kd, "events.db")))

    def test_missing_seed_dir_raises(self):
        with self.assertRaises(SystemExit):
            launcher.seed_knowledge_dir("/no/such/seed/dir", self.kd, verbose=False)

    def test_seed_makes_lineage_track_the_robot(self):
        # the whole point: after seeding, the pack's lineage == the robot's own
        # policy fingerprint, so Step A's importer flags it for --adopt.
        launcher.seed_knowledge_dir(self.seed, self.kd, verbose=False)
        with open(os.path.join(self.seed, "coach_policy.json")) as f:
            seed_policy = json.load(f)
        self.assertNotEqual(knowledge.seed_lineage(self.kd), "cold")
        self.assertEqual(knowledge.seed_lineage(self.kd),
                         knowledge.policy_lineage(seed_policy))


if __name__ == "__main__":
    unittest.main()
