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

**Reflection & Pattern Mining**: Facts and patterns learned in simulation carry over to real deployment. The robot arrives at the physical world with pre-trained intuitions.

**Decision Journal**: Every decision field_agent makes in simulation is journaled, building a corpus of "practice decisions" that can be reviewed or used to identify systematic biases.

**Coach Policy Cache**: Arms and outcomes accumulated during simulation are persisted to the policy file. Successful tactics discovered in the training environment remain available on the real robot.

## Deployment Workflow

1. **Develop & test** locally using the simulator
2. **Run scenario suite** to verify decision quality across diverse situations
3. **Inspect learned policy** (coach.py's cached arms and tactics)
4. **Deploy to robot** with the trained policy file—the bot begins physical operation with experience
5. **Collect real-world data** from physical exploration
6. **Merge learnings** from real operation back into the policy cache
7. **Iterate**: use real-world edge cases as new training scenarios

Over time, the robot learns from both simulated practice and real experience, converging on robust behavior.

## Example Benefits

- **Day 1**: Train the bot through 100 maze scenarios overnight, discovering which collision-recovery tactics work best
- **Day 2**: Deploy to real robot; it already knows how to escape tight corners from simulation
- **Day 3**: Real robot encounters a new failure mode; capture the sequence, replay it 50 times in simulation with tweaked coach parameters, then re-deploy
- **Week 1**: Reflection has mined 20+ patterns from simulated + real data; coach uses them to avoid known bad situations proactively

## No Timing or Implementation Details

The simulator's internal representation (2D physics, raycast distance sensing, bounding-box object detection) is an implementation choice. The key insight is that the simulator acts as a plug-in replacement for the hardware layer, speaking the same MQTT protocol as the real world. The specifics of how it models physics, renders obstacles, or computes sensor readings are secondary to the architectural abstraction: field_agent never knows (and never needs to know) whether `picarx/state/world` comes from real sensors or a virtual environment.

This design is scalable: as capabilities improve, the simulator can become more sophisticated (physics engines, realistic sensor noise, dynamics), but field_agent's code remains unchanged.
