# PiCar-X Training Simulation System

## Overview

The PiCar-X's decision-making layer (field_agent, coach, explorer, reflection) is already abstracted from hardware through MQTT topic communication. A training simulation system leverages this decoupling to allow the robot's "mind" to practice obstacle avoidance and decision-making in virtual environments without requiring physical hardware execution or Pi CPU resources.

The simulator runs on a development machine and replaces the real hardware's sensor stream and motor actuation with a lightweight virtual environment. Field_agent, coach, and other decision modules run unmodified on the same MQTT bus, making decisions based on synthetic sensor data. To field_agent's perspective, the training environment is indistinguishable from the real world—it sees distance readings, object detections, and battery status arriving on the same topics as always.

## System Architecture

### Information Flow

```
┌─────────────────────────────────────────┐
│   Development Machine (MQTT Broker)     │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────┐               │
│  │  Virtual Environment │               │
│  │  (Physics + Sensors) │               │
│  └──────────────────────┘               │
│    ↓ publishes      ↑ subscribes        │
│  picarx/state/world                     │
│  picarx/action/result                   │
│                ↕                        │
│  ┌──────────────────────────────────┐   │
│  │  field_agent.py   (unmodified)   │   │
│  │  coach.py         (unmodified)   │   │
│  │  reflection.py    (unmodified)   │   │
│  │  arbiter.py       (unmodified)   │   │
│  │  explorer.py      (unmodified)   │   │
│  └──────────────────────────────────┘   │
│    ↓ publishes      ↑ subscribes        │
│  picarx/intent/move                     │
│  picarx/coach/query                     │
│  (etc.)                                 │
│                                         │
└─────────────────────────────────────────┘
```

The simulator:
1. Maintains a virtual 2D or 3D space with obstacles, the robot's pose, and physics state
2. Receives movement intents from field_agent via `picarx/intent/move`
3. Updates robot position/heading based on those intents
4. Detects collisions and other physical constraints
5. Generates synthetic sensor readings (distance, object detection, battery, etc.) matching the format of real sensors
6. Publishes sensor data to `picarx/state/world` and action outcomes to `picarx/action/result`
7. Repeats at a realistic tick rate (e.g., 10 Hz)

### No Hardware Changes Required

The real robot's hardware pipeline (safety daemon, motor drivers, sensors) is **never involved**. The simulator completely replaces:
- The safety daemon's execution layer (acting as the safety verifier for the virtual world)
- Sensor publishers (distance_sensor.py, vision_basic.py, world_state.py)
- The arbiter's socket connection to real hardware

All decision-making code runs exactly as it does on the robot, but fed synthetic input and never producing real motor commands.

## Key Advantages

**Training Scale**: The bot can practice thousands of scenarios without physical wear or time constraints. Scenarios can run at 10× or 100× real-time speed.

**Deterministic Replay**: A specific sequence of sensor readings can be captured and replayed infinitely for debugging or analyzing field_agent's behavior on edge cases.

**Safe Experimentation**: Coach can suggest aggressive maneuvers; if they fail in simulation, no hardware is damaged. Failures feed directly into reflection and pattern mining.

**Bulk Data Collection**: Run many scenarios in parallel or sequentially to populate the semantic database with diverse experiences, accelerating pattern discovery and fact extraction.

**Hyperparameter Tuning**: Constants in field_agent (evasion thresholds, wander intervals) and coach (situation similarity, arm retirement) can be optimized against simulated performance metrics without touching the real robot.

**Reproducibility**: A scenario file fully describes an environment and initial conditions. Future runs use identical physics, allowing consistent comparison of behavioral changes or coaching improvements.

## Interaction Model

### From Field_Agent's Perspective

Field_agent remains unchanged. It:
1. Reads sensor data from `picarx/state/world` (distance, objects, battery, etc.)
2. Makes decisions (explore, evade, wander, etc.)
3. Publishes movement intents to `picarx/intent/move`
4. Listens for action outcomes on `picarx/action/result`
5. Updates its state machine and decision journal

In simulation, each of these operations is identical in behavior to the real robot. The only difference is timing—the simulator controls when sensor updates arrive and how collisions are resolved.

### From Coach's Perspective

Coach runs the same learning loop:
1. Receives failure-state queries from field_agent (`picarx/coach/query`)
2. Queries its cached policy and learned facts
3. Asks the LLM for a recovery maneuver if needed
4. Publishes suggested steps to `picarx/coach/suggestion`
5. Receives outcomes on `picarx/coach/outcome` when field_agent finishes executing the suggestion

In simulation, failures are deterministic (e.g., always collide with a particular obstacle at a specific angle). This makes the coach's learning more reproducible: the same situation produces consistent outcomes, helping the bandit algorithm converge on reliable tactics.

## Use Cases

**Continuous Development**: Before deploying changes to field_agent or coach, run the new version through a suite of training scenarios to verify behavior improvement or regression.

**Scenario Capture**: Record problematic sequences from real robot operation (e.g., a specific collision pattern), export them as training scenarios, and have the robot practice them in simulation until the issue is resolved.

**Ablation Studies**: Run the bot with and without specific modules (e.g., explorer.py, reflection.py) to measure their contribution to decision quality.

**Curriculum Learning**: Start with simple scenes (few obstacles, open spaces), gradually increase complexity (mazes, tight corners, moving obstacles) as the coach's policy improves.

**Coaching Validation**: Verify that the coach's learned tactics actually generalize to new situations by testing on held-out scenarios the coach has never seen.

## Implementation Approach

The simulator is a standalone module that:
1. Loads a scenario description (obstacle positions, robot start pose, bounds)
2. Connects to the same MQTT broker as field_agent and coach
3. Runs a physics/collision loop at a fixed tick rate
4. Translates physical state into sensor messages matching the real robot's message format
5. Handles arbiter queries (distance, battery) if needed
6. Logs all MQTT traffic to events.db for later analysis

Scenario files (JSON, YAML, or similar) describe:
- Obstacle geometry and positions
- Robot start position and heading
- Optional goal location or target condition
- Physics parameters (friction, speed limits, etc.)

Multiple scenarios can be composed into a test suite that runs in sequence or parallel, collecting statistics on field_agent's performance (collision rate, exploration efficiency, time to goal, etc.).

## Integration with Existing Systems

**Event Logger**: All MQTT traffic during simulation is logged to events.db just like real robot operation. Reflection.py can analyze simulated events identically to real ones, extracting patterns and facts.

**Reflection & Pattern Mining**: Facts and patterns learned in simulation carry over to real deployment via the knowledge pack (below). The robot arrives at the physical world with pre-trained intuitions.

**Decision Journal**: Every decision field_agent makes in simulation is journaled, building a corpus of "practice decisions" that can be reviewed or used to identify systematic biases.

**Coach Policy Cache**: Arms and outcomes accumulated during simulation are persisted to the policy file. Successful tactics discovered in the training environment remain available on the real robot.

## The knowledge pack — turning a session into something the robot can use

A raw run leaves behind a coach policy and a pile of per-episode event logs;
neither is, on its own, something you can hand a robot. Passing
`--knowledge-dir <dir>` (older alias: `--policy-dir`) turns that directory into
an **accumulating, deployable knowledge pack**:

```bash
python3 run_training.py scenarios/*.json --knowledge-dir training_data
```

Across the suite the directory gathers:

- **`coach_policy.json`** — the coach's learned escape maneuvers, accumulated by
  `coach.py` exactly as on the robot.
- **`events.db`** — every episode's event log, folded in after each run
  (`launcher._accumulate_events`) so a suite-wide analysis has real volume.

When the suite finishes, `sim/knowledge.py` distils those into two portable
files next to them:

- **`navigation_facts.json`** — transferable facts + behavioural patterns. The
  patterns come from the robot's **own unmodified `pattern_miner`** ("obstacle
  vetoes come in bursts, a single small retreat won't clear it"; "escapes here
  that start by reversing usually work"); the facts are first-person
  escape-tactic tendencies aggregated from the coach's win/loss records, mirroring
  `reflection.py`'s self-model thresholds. No API key and no network required.
- **`knowledge_pack.json`** — a manifest: scenarios trained, episode/collision/
  coach totals, what the pack contains, and a **lineage** id (see below) marking
  which robot policy the pack descends from.

**What is deliberately *not* exported:** place-specific and spatial memories.
The sim's rooms are not the real house, so "the corner with the sofa vetoes a
lot" would be a lie on the physical robot. Only knowledge about the *robot
itself* — how Ackermann steering escapes a trap, which failure modes loop —
transfers. The robot rebuilds its map from real sensors.

The robot ingests a pack with its own [`layer_b/import_training.py`](../picarx/layer_b/import_training.py),
which **combines** (never overwrites), in one of two modes for an escape
maneuver the robot already knows:

- **merge** (default) — the pack's win/loss counts are *added* to the robot's,
  so two **independent** learners reinforce the same UCB1 statistics. Right for
  a pack trained on a dev machine or from a cold-started sim.
- **adopt** (`--adopt`) — the pack's refined counts **replace** the robot's,
  because the pack was seeded from *this robot's own* policy (see
  **Self-training from live data** below) and summing would double-count the
  shared seed.

Either way, unseen situations/arms and transferable facts are always taken on.

## Self-training from live data (the idle round-trip)

The workflows above build a pack from scratch on a dev box. The robot can also
**refine its own accumulated learning** while it sits idle: seed a run from the
robot's live data dir, let the sim sharpen the coach policy against harder
virtual traps than the carpet offers, then import the result **back** with
`--adopt`.

`--seed-from <dir>` copies the robot's live files into the knowledge dir
**before** the run, so the coach refines real experience instead of a blank
slate:

```bash
# on the robot, while idle — layer_b/data is the robot's live data dir
python3 run_training.py scenarios/*.json \
    --knowledge-dir /tmp/selftrain --seed-from layer_b/data --speedf 4 --quiet
# then, with Layer B stopped:
python3 layer_b/import_training.py /tmp/selftrain --adopt
```

- **What gets seeded:** `coach_policy.json` (the one that matters — the coach
  reads and refines it in place), plus `events.db` and `semantic.db` if present.
  `spatial.db` is **never** seeded — place memories don't transfer. Files already
  in the knowledge dir are **kept** so a resumed run builds on its progress;
  `--seed-force` overwrites them. SQLite dbs are snapshotted through the backup
  API, so seeding is safe even while Layer B is still writing.
- **Lineage → `--adopt`:** the pack's manifest is stamped with a **lineage** id —
  a fingerprint of the seed `coach_policy.json` (or `cold` when unseeded), stable
  across re-runs of the same dir. On import, `import_training.py` compares it to
  the robot's own policy; a match means "this is your own learning, refined,"
  and it recommends `--adopt` so the shared seed isn't counted twice (an arm the
  robot left at 7/1 and the sim sharpened to 10/2 imports back as 10/2, not 17/3).
- **Killable instantly:** a `SIGTERM` to `run_training.py` tears the run down as
  cleanly as `Ctrl-C` — module subprocesses terminated, sim and bus stopped —
  and still distils whatever completed. So an idle-time trainer can be stopped
  the moment the robot has real work to do.
- **Isolation is unchanged:** training always talks to a private,
  per-run `/tmp/picarx_train_<port>.sock` and an ephemeral bus port — never the
  real `/tmp/picarx_safety.sock` or `localhost:1883`, so it can never drive the
  physical robot or disturb a running safety daemon.

## Deployment Workflow

1. **Develop & test** locally using the simulator.
2. **Run a scenario suite** with `--knowledge-dir training_data` to accumulate a
   knowledge pack across diverse situations.
3. **Inspect the pack** — `knowledge_pack.json` for provenance, `navigation_facts.json`
   for the distilled intuitions, `coach_policy.json` for the learned arms.
4. **Deploy to the robot**: copy `training_data/` over and, with Layer B stopped,
   `python3 layer_b/import_training.py training_data` (add `--dry-run` first to
   preview; add `--adopt` if the pack was seeded from this robot's own data —
   the importer flags this when the lineage matches). Restart the orchestrator;
   the bot begins physical operation with experience.
5. **Collect real-world data** from physical exploration.
6. **Merge learnings** — real operation keeps updating the same `coach_policy.json`
   and `semantic.db` the pack seeded.
7. **Iterate**: capture real-world edge cases as new scenarios and retrain.

Over time, the robot learns from both simulated practice and real experience, converging on robust behavior.

## Example Benefits

- **Day 1**: Train the bot through 100 maze scenarios overnight, discovering which collision-recovery tactics work best
- **Day 2**: Deploy to real robot; it already knows how to escape tight corners from simulation
- **Day 3**: Real robot encounters a new failure mode; capture the sequence, replay it 50 times in simulation with tweaked coach parameters, then re-deploy
- **Week 1**: Reflection has mined 20+ patterns from simulated + real data; coach uses them to avoid known bad situations proactively

## No Timing or Implementation Details

The simulator's internal representation (2D physics, raycast distance sensing, bounding-box object detection) is an implementation choice. The key insight is that the simulator acts as a plug-in replacement for the hardware layer, speaking the same MQTT protocol as the real world. The specifics of how it models physics, renders obstacles, or computes sensor readings are secondary to the architectural abstraction: field_agent never knows (and never needs to know) whether `picarx/state/world` comes from real sensors or a virtual environment.

This design is scalable: as capabilities improve, the simulator can become more sophisticated (physics engines, realistic sensor noise, dynamics), but field_agent's code remains unchanged.
