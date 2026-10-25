# EIDOLON

An experimental artificial-life sandbox for Windows desktop. **Live Learning**
is the default desktop mode: watch real neural muscle candidates practice in
repeatable episodes, automatically retry, and evolve through the existing
sequential mutation/selection algorithm. The laboratory shows the simulation,
actual neural activity, and live or saved training results.
Eight controllable material regions deform existing bonds and can move the body
through floor contact with compatibility propulsion disabled. Normal Survival
mode remains available, with rules as its default controller; the seeded NumPy
steering controller is retained.
The biomass appearance, grabbing, tearing/reconnection, energy, growth,
starvation, fixed world, and model format remain.

Muscle weights can be optimized through mutation and selection. There is no
reinforcement learning algorithm, cognitive memory, body-structure evolution,
reproduction, or second organism. The original steering network remains untrained.
No ML frameworks, externally pretrained models, LLMs, cloud
APIs, credentials, or network connection are required after installation.
Reaching/gripping and food-seeking training are outside this phase.

## Windows setup (PowerShell)

Install **64-bit Python 3.14** from [python.org](https://www.python.org/downloads/)
with the Python launcher enabled, and install Git. Python 3.11 or newer is required;
3.14 is the target version for Windows 11.

```powershell
git clone https://github.com/Schwzz/Eidolon.git
cd Eidolon
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m eidolon
```

The explicit interpreter paths work without activation or changing PowerShell
execution policy. For an existing checkout, skip cloning and install from its
root. Dependencies remain **Pygame-CE**, **NumPy**, and **pytest** in the `dev`
extra. Install `pygame-ce`, not the separate `pygame` distribution; both use the
same import name and should not share an environment.

## Run and controls

```powershell
.\.venv\Scripts\python.exe -m eidolon --seed 17 --max-fps 60 --width 1280 --height 720
.\.venv\Scripts\eidolon.exe --seed 42
.\.venv\Scripts\python.exe -m eidolon --body simple --muscle-test manual --show-nodes
```

| Control | Action |
| --- | --- |
| Left mouse button, hold and move | Grab a nearby node; pull gently to stretch or firmly to tear local bonds; release to allow recovery |
| Space | Pause/resume the current mode, including learning progress or survival systems |
| R | Live Learning: retry the current attempt, retaining the population and completed evaluations. Survival: reset the seeded world and biology |
| S | Stop/continue Live Learning, retaining progress |
| N | Archive the current learning run and start a fresh population with the same settings |
| L / top-right mode buttons | Switch Live Learning / Survival; retain each mode's state while inactive |
| F1 / F2 / F3 / F4 | Simulation / Neural Network / Training / Split View |
| − / + (or =), including numeric keypad | Change simulation speed in 0.25× steps, from 0.25× to 2× |
| Escape / window close | Exit and release Pygame resources |
| `--seed INTEGER` | Reproduce colony layout, masses, palette, food, and behavior |

The default window is 1280 × 720; the minimum is 480 × 320. Ctrl+C also exits.
Live Learning always uses the existing simple body. Survival retains the
detailed body by default. `--body simple` selects exactly
**10 nodes and 17 initial bonds**: two five-node rows with horizontal rails, five
vertical links, and alternating diagonal braces. At the default radius, spacing
is 19.2 world units. Every muscle channel affects existing bonds, including four
upper and four lower rail segments. The same seed determines small shape/mass
variations, and assignments stay attached to material during deformation.
`--body detailed` retains the existing 85–98-node mesh. Both bodies share the same
physics, growth, survival, grabbing, tearing, reconnection, and flat membrane
renderer. `--show-nodes` adds optional small outline markers without changing
physics. Training defaults to `simple`.
The 60 FPS limit is configurable; `--max-fps 0` removes it and may increase CPU use.
Grabbing uses the current fit camera, including letterboxing. Empty clicks are
ignored. Mouse release, leaving the window, loss of focus, resizing, pause,
restart, death, and shutdown cancel the grab; press again to resume interaction.
Dragging beyond the chamber clamps the target to the world boundary.
Grabbing is available in Survival; learning episodes exclude external pointer
forces so fitness reflects only the candidate's muscles. Eight muscle readouts
remain visible in Simulation and Split View, including during neural control.
They show actual physical activation percentages: `+` contracting, `-` relaxing,
and `=` holding, compared with each muscle's requested activation.

The same seed, settings, and input/update sequence reproduce the simulation.
Colony generation, appearance colors, and food spawning use separate private
random generators. Rendering consumes no random numbers and changes no physics
state. Floating-point/raster results may differ across Python or platform versions.

## Live Learning

```powershell
.\.venv\Scripts\python.exe -m eidolon
.\.venv\Scripts\python.exe -m eidolon --mode live --live-run runs\my-live-run --seed 17 --view split
.\.venv\Scripts\python.exe -m eidolon --mode survival
```

The visible organism is the candidate being evaluated, controlled by its actual
**30→16→8** muscle network. Each new run defaults to six candidates, two elites,
and two world seeds (`--seed` and the next integer). Each attempt passively
settles for three simulation seconds, then acts for up to six seconds toward a
target 20 world units right. The unchanged episode reward measures physical net
progress, stability, and muscle-energy efficiency. There is no propulsion,
automatic gait, fragment homing, food-seeking, or starvation in learning episodes.
The fixed ten-node body and existing physics are unchanged.

After each attempt, its real result is retained and the next candidate/seed
starts automatically. After all twelve attempts, mean multi-seed scores drive
the existing elite selection and weight mutation. A full default generation is
about 108 simulation seconds plus transitions, unless targets finish early.
No heavy population evaluation runs in the draw loop: each frame advances at
most 24 fixed physics ticks within a default 4 ms work budget. If the machine
falls behind, simulation time slows rather than skipping reward-bearing ticks.
The only background worker writes checkpoints; it never evaluates candidates.
Changing weights is real optimization, but **effective crawling is not
guaranteed or demonstrated by this usability change**.

The Simulation HUD shows generation, attempt, current reward, and best completed
score. Neural Network shows the current candidate's actual inference, with
settling/paused/stopped activity labeled explicitly. Training shows live state,
completed-generation reward curves, candidate scores, and save status. Switching
tabs never interrupts the attempt. Switching modes retains each session in
memory and advances only the selected mode.

Progress saves automatically after attempts, when stopping, and on clean exit.
The default directory is `runs/live`, ignored by Git. Atomic `live.json` is the
authoritative population/RNG/cursor snapshot; standard `checkpoint.json`,
`metrics.json`, and champion files are also written for the existing CLI tools.
One active save plus one newest pending save bounds disk work. A single-writer
OS lock prevents the GUI and headless trainer from overwriting the same run.
Save failures visibly stop learning instead of claiming success.

Launching again with the same directory restores completed evaluations, neural
weights, RNG state, and pause/stop flags. **An unfinished attempt restarts from
its seeded beginning**; within-attempt node positions are not checkpointed.
`R` retries without discarding learning. `N` first archives the previous snapshot
and artifacts under `archive-001`, etc., then starts anew. The GUI can also open a
standard simple-body `checkpoint.json` directory from headless training; saved
settings take precedence over launch defaults. Use a fresh directory for a
different experiment. Close the GUI before headless continuation of its run.
Headless continuation repeats an unfinished generation from its saved
population, then updates both checkpoint formats; subsequent GUI launches use
the newer completed progress.

Normal Survival is continuous, with feeding, growth, metabolism, and starvation;
it never automatically resets. Existing explicit `--brain`, `--controller`,
`--body`, or `--muscle-test` launches select Survival unless a mode is specified.
Saved steering and muscle brains retain their existing formats and behavior.

## Controllers and neural parameters

Compare both controllers in the same initial world:

```powershell
.\.venv\Scripts\python.exe -m eidolon --controller rules --seed 17
.\.venv\Scripts\python.exe -m eidolon --controller neural --seed 17
.\.venv\Scripts\python.exe -m eidolon --controller neural --seed 17 --network-seed 123 --hidden-layers 16 8
```

The existing HUD phase label shows `RULES` or `NEURAL`. `--seed` still determines
the world, colony, and appearance; the network uses that seed by default.
`--network-seed` changes only initial weights, so different networks can share
an identical initial world. Seeds are normalized modulo 2⁶⁴ for NumPy initialization.
Private random generators do not change world/resource randomness. Trajectories
can diverge when the controllers make different decisions.

The default network has **12 inputs → 16 hidden units → 2 outputs**, with **242
float64 parameters**. Each layer computes `tanh(W @ x + b)`. Xavier normal weights
and zero biases are initialized once. Inference is stateless: no weight updates,
memory, or fresh random draws occur. `--hidden-layers 16 8` selects two hidden
layers; the Python API also supports arbitrary input/output widths and no hidden
layer, within the documented size limits below.

Inputs have this fixed order:

| Indices | Sensory input | Normalization / unavailable value |
| --- | --- | --- |
| 0 | Energy | Energy / maximum, clipped to [0, 1] |
| 1–2 | Direction toward sensed food | World-space unit x/y vector in [−1, 1]; (0, 0) without food or at zero distance |
| 3 | Food distance | Distance / sensing radius, clipped to [0, 1]; 1 without food |
| 4–5 | Current heading | Cosine and sine of movement direction |
| 6 | Actual movement speed | Speed / configured movement speed, clipped to [0, 1]; 0 when movement is disabled |
| 7–10 | Left, right, top, bottom proximity | `clip(1 - (distance_to_wall - body_clearance) / sensing_radius, 0, 1)` |
| 11 | Food detected | 1 if the sensory snapshot contains food, otherwise 0 |

Output 0 requests steering relative to current heading (`output × π` radians).
Output 1 sets effort (`(output + 1) / 2`). Existing turn-rate limits determine
heading; the horizontal component of requested movement drives floor traction.
The controller cannot apply upward force or propel airborne tissue. Energy
costs, particle constraints, gravity, and contacts determine the response. Rules
seek food horizontally and wander between the side walls, ignoring floor/ceiling
repulsion that previously steered them vertically. Saved networks keep the same
12 inputs, two outputs, and version-1 JSON format.
Neural mode can circle, ignore food, or starve; it receives no rule-based rescue.

The controller is evaluated **once per live simulation integration step** (normally
60 times/second at 1×, more at increased simulation speed). Rendering and internal
1/120-second physics substeps do not evaluate it. Pause and death stop decisions.
Restart recreates the seeded world and retains the selected network unchanged.

Save initialized parameters or load an existing snapshot:

```powershell
.\.venv\Scripts\python.exe -m eidolon --controller neural --seed 17 --save-network models\seed17.json
.\.venv\Scripts\python.exe -m eidolon --controller neural --seed 17 --load-network models\seed17.json
```

Saving happens before the window opens and overwrites the requested file. Parent
directories are created. Local `models/` snapshots are Git-ignored; model files
remain separate from application configuration. Network options require neural
mode. Loading cannot be combined with initialization seed/hidden-layer options.

Snapshots use **UTF-8 JSON**, with exactly these fields: `format` =
`"eidolon.feedforward"`, `version` = `1`, `activation` = `"tanh"`, `layers` (integer
widths), `weights` (one matrix per transition, shape **output × input**), and
`biases` (one output-width vector per transition). Loading validates metadata,
numeric types, shapes, and finite values; it never executes serialized code.
Controller models must have 12 inputs and 2 outputs. Limits are 2–8 total layers,
1–256 units/layer, 65,536 parameters, and 8 MiB per input file. Invalid files are
reported before window startup; numerical overflow during inference stops cleanly
with an error rather than injecting invalid movement into the colony.

The math can be used without the application or graphics:

```python
from eidolon.neural_network import FeedForwardNetwork

network = FeedForwardNetwork((12, 16, 8, 2), seed=17)
outputs = network.forward([0.0] * 12)
network.save("models/example.json")
restored = FeedForwardNetwork.load("models/example.json")
parameters = restored.to_parameters()
copied = FeedForwardNetwork.from_parameters(parameters)
```

## Grounded physics and passive episodes

World **+Y points downward**. The default floor is the bottom boundary at
`world.height`; the default world remains 1200 × 800 units. Seeded tissue starts
two units above floor contact and settles, rather than spawning in the middle
of the chamber. The fit camera and resizing affect only the view.

`PhysicsConfig` is shared by the world and colony. Defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `gravity` | 180 | Downward acceleration in world units/second², independent of mass |
| `floor_friction` | 0.55 | Floor friction coefficient |
| `linear_damping` | 0.8 | Exponential damping per second |
| `max_speed` | 220 | Food speed cap; node cap is this value × √growth scale |
| `contact_slop` | 0.5 | Small floor-support tolerance in world units |
| `drive_acceleration` | 240 | Maximum compatibility-drive acceleration |
| `propulsion_enabled` | `True` | Enable compatibility traction, fragment homing, and automatic rest-length waves |

Gravity, contacts, friction, and elastic constraints run at substeps of at most
1/120 second. Floor projection prevents penetration; downward contact velocity
is removed with no collision restitution. Particle friction uses the normal
contact correction to limit tangential movement. Mass-weighted springs and
relative damping let the tissue compress and relax instead of holding a rigid
pose. Soft tissue may rebound elastically, but the controller has no flight force.

`locomotion.py` isolates compatibility propulsion: horizontal force is applied
only through supporting nodes and is limited and distributed over floor contacts.
Neighboring tissue follows through elastic bonds. Recovery similarly targets
floor-supporting tissue instead of trying to levitate toward elevated nodes.
Grabbing remains an external force and can lift tissue while held.

```powershell
.\.venv\Scripts\python.exe -m eidolon --gravity 180 --floor-friction 0.55
.\.venv\Scripts\python.exe -m eidolon --no-propulsion
.\.venv\Scripts\python.exe -m eidolon --controller neural --no-propulsion
```

`--no-propulsion` disables traction, assisted fragment travel, and automatic
breathing/traveling bond waves completely. Gravity, damping, elastic recovery,
contacts, grabbing/tearing, nearby material reconnection, and commanded muscles
remain. Without muscle commands this is a passive body.
Public node IDs, positions, velocities, masses, adjacency, and bond rest lengths/
compliance remain available for future local control.

The headless API needs no graphics or training framework:

```python
from eidolon.config import PhysicsConfig
from eidolon.simulation import Simulation

simulation = Simulation(seed=17, physics_config=PhysicsConfig(
    gravity=180, floor_friction=0.55, propulsion_enabled=False,
))
for _ in range(600):
    simulation.update(1 / 60)
simulation.reset()  # Same initial tissue, food positions/velocities, clocks, and RNG state.
```

These are simplified world-unit dynamics, not a calibrated material model.
There is currently one flat floor and no obstacles. Food collides with world
walls/floor; food–food and food–tissue pushing contacts are not implemented.
Consumption still uses the existing feeding interaction range.

## Artificial muscles

```powershell
.\.venv\Scripts\python.exe -m eidolon --muscle-test manual --no-propulsion
.\.venv\Scripts\python.exe -m eidolon --muscle-test wave --seed 17
```

Both muscle-test modes **automatically disable compatibility propulsion**, assisted
fragment travel, and the old automatic gait. Steering controllers are bypassed in
test mode. All displacement comes from internal bond constraints interacting with
gravity, ground contact, and friction; no translation, horizontal drive force,
or fragment homing is substituted. Mouse grabbing still applies an external force.
Normal rules/neural modes and saved networks are preserved; muscles start relaxed.

In `manual` mode, **hold number-row keys 1–8** to contract individual regions,
release to relax, or press **0** to relax all. Regions are arranged in the original
material layout: upper left→right **1–4**, lower left→right **5–8**. Membership
stays attached to node IDs when the tissue moves, folds, grows, tears, or reconnects;
it does not mean current screen-space top/bottom after deformation. New bonds use
the midpoint of their endpoints' original material coordinates. The HUD shows
eight activation bars/numbers, **dx** (all-material horizontal center-of-mass
displacement from reset, including fragments, in world units) and **E** (cumulative
muscle energy spent). The organism renderer is unchanged.

`wave` mode uses a predefined four-second traveling contraction with upper/lower
regions in opposite phases, after three seconds of passive settling. It is a
demonstration pattern, not a learned or food-seeking gait. Bare `--muscle-test`
selects `wave`. Pause freezes activation, physics, and pattern time; reset clears
muscle state/costs and restarts the seeded episode. Pause, resize, lost focus,
window leave, shutdown, and death clear manual commands; press again to resume.
The automatic wave resumes on the next live update, except while paused/dead.
Space, R, +/−, Escape, and mouse interaction keep their existing meanings.

`muscles.py` is independent of graphics, neural math, biology, and external forces.
Eight normalized actions in **[0, 1]** control existing XPBD rest-length targets.
`MuscleConfig` defaults: maximum shortening **28%** (hard configuration ceiling
35%), activation increase **1.5/second**, relaxation **2/second**, holding cost
**1.2 energy/second** at full-body activation, and **0.8 energy** per full-body
activation increase. Costs are weighted by the original material share, so new
healed bonds do not multiply consumption. The simulation deducts costs from the
same biological energy supply as metabolism and movement. Insufficient fuel or
death triggers rate-limited passive relaxation; it cannot power new contractions.
Original bond rest lengths remain unchanged, so release restores relaxed targets.
Existing compliance, collision, damping, and velocity/extension limits still apply.

Future controllers can use this headless action interface without changing the
current **12→16→2** steering network or version-1 neural JSON:

```python
from eidolon.config import PhysicsConfig
from eidolon.simulation import Simulation

simulation = Simulation(seed=17, physics_config=PhysicsConfig(propulsion_enabled=False))
simulation.set_muscle_activations([0.8, 0.4, 0, 0, 0, 0, 0.4, 0.8])
for _ in range(120):
    simulation.update(1 / 60)
activations = simulation.colony.muscles.activations
regions = simulation.colony.muscles.node_regions
simulation.relax_muscles()
simulation.reset()  # Same mesh, assignments, relaxed actions, food, and RNG state.
```

The original steering network is not trained, replaced, or mapped to eight outputs;
the evolutionary muscle policy described below is separate.
For isolated muscle movement use `propulsion_enabled=False` or a muscle-test mode;
normal compatibility mode can intentionally combine its old drive with muscle actions.
Detached fragments retain their muscle regions and shared biology. Nearby contact
can rebuild multiple bonds; distant fragments have no assisted return when
propulsion is disabled and may remain separated until physical contact is restored.
The graph is a simplified elastic material, not an anatomical muscle model.

## Headless locomotion laboratory

Run these from the repository root after installation. No Pygame window opens;
the episode modules import neither Pygame nor NumPy. These commands evaluate
fixed patterns, without neural training or weight changes:

```powershell
.\.venv\Scripts\python.exe -m eidolon.train --body simple --seed 17 --duration 20 --pattern idle
.\.venv\Scripts\python.exe -m eidolon.train --body simple --seed 17 --duration 20 --pattern random
.\.venv\Scripts\python.exe -m eidolon.train --body simple --seed 17 --duration 20 --pattern scripted --target-distance 100
.\.venv\Scripts\python.exe -m eidolon.train --body detailed --seed 17 --duration 20 --pattern scripted
.\.venv\Scripts\python.exe -m eidolon.train --body simple --pattern random --duration 60 --episodes 3 --benchmark
```

The installed `eidolon-train` command is equivalent. Output is JSON containing
distance (signed world X), target progress (positive toward the goal), total
reward, actuator energy, stability, contacts/components, and completion state.
`--benchmark` adds setup time, stepping time, ticks/second, and simulated seconds
per wall second. Episodes use seeds `seed`, `seed+1`, …; no policy changes between
them. Run separate patterns with the same seed/settings for comparisons. Random
actions use a private seed and change only at action intervals. `scripted` uses
the existing traveling contraction; it is not target-aware or a learned gait.

Defaults: 120 Hz fixed ticks, 0.1-second action intervals, 20-second episodes,
target 100 world units right, and three seconds of passive settling before scoring.
`--hz`, `--action-interval`, `--duration`, `--settle-seconds`, `--gravity`, and
`--floor-friction` configure experiments. Durations/intervals must contain whole
fixed ticks; the final action is shortened to the episode end. Negative
`--target-distance` selects a leftward goal. Targets must stay inside conservative
side-wall clearance. Settling drift/energy is excluded from episode measurements.

`training.LocomotionEnvironment` owns the existing `Colony` directly. Propulsion,
fragment homing, automatic gait, and steering are disabled. It runs no survival,
food, metabolism, starvation, or growth systems. Actuation has unlimited experimental
fuel, with real muscle costs accumulated separately and penalized. Normal desktop
biology still limits muscle use by energy and can starve. Colony positions are
set only at reset; subsequent motion comes from muscle constraints and contacts.

```python
from eidolon.training import EpisodeConfig, LocomotionEnvironment

env = LocomotionEnvironment(EpisodeConfig(body="simple", seed=17, duration=20))
observation = env.reset()  # Repeats the same settled body, muscles, and clock.
while not env.done:
    result = env.step([0.8, 0.4, 0, 0, 0, 0, 0.4, 0.8])
    observation = result.observation
    # result.reward, result.terminated, result.truncated, result.info
metrics = env.metrics()
```

Actions are eight finite values in [0, 1]. Invalid actions fail before advancing
time or physics. Thirty normalized observations have this stable order:

| Indices | Values | Normalization |
| --- | --- | --- |
| 0–1 | All-material center X/Y | World coordinate → [−1, 1] |
| 2–3 | All-material velocity X/Y | Divide by velocity cap, clip to [−1, 1] |
| 4 | Remaining target X offset | Divide by absolute target distance, clip to [−1, 1] |
| 5 | Ground contact | Supporting nodes / node count |
| 6–7 | Width/height deformation | Relative change from settled extent, clip to [−1, 1] |
| 8–9 | Mean signed / maximum absolute bond strain | Relative to relaxed length, clipped |
| 10–11 | Fragmentation / time remaining | [0, 1] |
| 12–19 | Actual muscle activations | [0, 1] |
| 20–27 | Requested muscle actions | [0, 1] |
| 28–29 | Latest-interval / total muscle energy | Divide by theoretical rate bound × interval/duration |

`OBSERVATION_NAMES` provides the names. This 30-value laboratory interface is
separate from the existing 12→16→2 steering network and version-1 JSON models.

For target direction `s`, settled starting mass center `x0`, and target magnitude
`D`, define `p = clip(s * (xCOM - x0), -D, D)`. Each stable physics tick earns:

```text
reward = (p_now - p_before) / D
         - 0.02 * muscle_energy_used
         - 0.01 * dt if forward progress <= 0.001 * dt
         - 0.10 * dt * (component_count - 1)
         + 5.0 on the first cohesive target crossing
```

Progress differences telescope, so cycling backward and forward cannot accumulate
positive progress reward. Measurement includes every fragment's mass, avoiding
centroid jumps when the largest component changes. Reaching the target terminates
the episode once; duration expiry truncates it. A numerical/boundary/conservation
fault gives **−5 for the failing tick**, earns no tick progress/bonus, and terminates.
Side-wall contact is excluded as a locomotion anchor. Invalid state returns the
last finite observation. `RewardConfig` exposes penalty/bonus coefficients.

This is an experimental reward design, not an exploitation-proof or calibrated
task. Locomotion is slow and seed/pattern-dependent; fixed demonstrations can have
negative reward or move away from the target. Distant fragments lack assisted
recovery with propulsion disabled. Training episodes have no mouse input, while
the normal application retains grabbing and recovery. This episode API itself
does not optimize weights; the separate evolutionary trainer below drives it.

## Evolutionary muscle training

`NeuralMuscleController` uses the same 30 normalized observations and eight
independent [0, 1] muscle actions. Its default network is **30→16→8**, with **632
float64 parameters** and tanh in every layer. Outputs map through `(tanh + 1)/2`.
One or two hidden layers of 1–32 units are configurable. The existing **12→16→2**
steering controller and its version-1 network JSON files are unchanged.

Training evaluates each candidate sequentially across every configured seed,
ranks by mean episode reward, keeps the best elites unchanged, and fills the
population by choosing elite parents and adding masked Gaussian mutations to
weights/biases. Defaults: population 12, three elites, mutation probability 0.2,
standard deviation 0.15, and parameters clipped to [−5, 5]. Stable index order
breaks ties. Only neural parameters evolve; body, physics, actions, and rewards
stay fixed. No food or starvation systems run in these episodes.

PowerShell examples (the first command reproduces the reference experiment):

```powershell
.\.venv\Scripts\python.exe -m eidolon.evolve train --output runs\demo --generations 20 --population 8 --elites 2 --seed 123 --seeds 0 17 42 --duration 12 --target-distance 20 --hidden-layers 16
.\.venv\Scripts\python.exe -m eidolon.evolve evaluate --brain runs\demo\best.json --seeds 101 102 103 104 105
.\.venv\Scripts\python.exe -m eidolon.evolve evaluate --brain runs\demo\best.json --baseline-brain runs\demo\initial-best.json --seeds 101 102 103 104 105 --compare --output runs\demo\heldout.json
.\.venv\Scripts\python.exe -m eidolon.evolve resume --checkpoint runs\demo\checkpoint.json --generations 5
.\.venv\Scripts\python.exe -m eidolon --brain runs\demo\best.json --training-run runs\demo --view split --seed 101
```

`eidolon-evolve` is the equivalent installed command. New training refuses to
overwrite another run's artifact files. Resume uses all saved settings and takes
an **additional** generation count; `--output runs\continued` can save continuation
to a fresh directory. Candidate evaluation is CPU-only and sequential; larger
populations, detailed bodies, longer episodes, and more seeds multiply runtime.
Start with a small run. `--body detailed`, `--mutation-std`, and `--mutation-rate`
are also available. Headless commands still evaluate full episodes sequentially;
Live Learning advances the same evaluation incrementally in the window.

After each completed generation the run directory contains:

- `best.json`: muscle-brain bundle, network parameters, exact episode settings,
  ordered observation schema, and training-seed provenance.
- `initial-best.json`: the first unmutated population's champion, selected on the
  same training seeds, for an initialization baseline.
- `metrics.json`: configuration and completed generation/candidate metrics.
- `checkpoint.json`: next population, champions, completed history, configuration,
  and full PCG64 RNG state, allowing exact continuation in the same environment.

Files use bounded, validated UTF-8 JSON with no executable serialization. Each
file is replaced atomically. The checkpoint is authoritative if interruption
occurs between artifact writes; resume from its last completed generation.
Ordinary `runs/` and `models/` output is Git-ignored. Reproducibility assumes the
same Python/NumPy/platform and fixed settings; cross-platform bitwise equivalence
has not been verified. Changing evaluation seeds never changes saved weights.

`--compare` evaluates the loaded policy, idle, seeded random actions, and the
predefined scripted pattern on identical worlds/settings. `--baseline-brain`
adds the initial champion. Reports include per-seed results and `held_out`, which
is true only when all evaluation seeds are absent from known training seeds.
Do not choose a champion on held-out scores. Evaluate on more seeds/horizons
before claiming a robust gait or generalization.

A small completed experiment is included in [experiments/phase2b2](experiments/phase2b2/README.md),
including its resume checkpoint. Twenty generations used population 8, two elites,
training seeds 0/17/42, 12-second episodes, and a target 20 units right.
Training champion reward improved from **−0.1772 to +0.0481**. On seeds 101–105:

| Policy | Mean reward | Mean X displacement | Mean muscle energy | Targets reached |
| --- | --- | --- | --- | --- |
| Trained brain | −0.0368 | +2.014 | 5.727 | 0/5 |
| Initial population champion | −0.1564 | +0.595 | 5.626 | 0/5 |
| Idle | −0.1193 | −0.010 | 0 | 0/5 |
| Random actions | −0.3443 | −0.291 | 13.407 | 0/5 |
| Scripted pattern | +0.2577 | +8.904 | 7.088 | 0/5 |

All evaluated episodes were stable. This demonstrates a small improvement over
initialization and random/idle reward, with more physical progress than the
initial champion. **It does not demonstrate solved locomotion:** the scripted
pattern remains substantially better, and no policy reached the target. These
are one run and five held-out seeds, not evidence of a reliable crawling gait.

## Laboratory tabs and continuous brain simulation

Click the tabs or use **F1 Simulation**, **F2 Neural Network**, **F3 Training**,
and **F4 Split View**. Tab changes do not reset or pause the running simulation;
they release an active pointer grab. Space, R, +/−, Escape, and simulation-panel
mouse interactions remain available. Split view maps the mouse through its own
left-hand camera; resizing changes the viewport, not the physical body.

The neural view draws the loaded network's actual layers, neurons, and weighted
connections. Green/purple encode positive/negative weights and captured values;
hover a neuron for its value and bias. Muscle networks label all 30 inputs,
hidden layers, and eight muscle outputs. The old steering network shows its
actual 12 inputs and two outputs instead. Values are captured during real
inference and held between action intervals; rendering never evaluates a network
or fabricates activity. Paused views identify the last sample. Rules/manual modes
show no active network, and dead organisms stop displaying live activity.

During Live Learning, Training displays current evaluation state and actual
completed-generation metrics, updating automatically. In Survival with no live
session, it displays **saved, completed generations**, best/mean reward curves,
and top candidates. `--training-run` accepts a run directory or metrics JSON;
**F5** reloads an external CLI snapshot in this saved-results view. Live training
does not ask you to refresh. A retained inactive live session is labeled inactive.

`--brain` runs the saved network continuously in the normal survival simulation.
Its saved episode duration and target crossing do not stop the organism. Food,
regeneration, feeding, metabolism, muscle energy costs, growth, and starvation
remain active. The brain controls muscles at its saved action interval, with
compatibility propulsion disabled. Pause freezes the complete simulation;
R restarts its normal seeded world, biological state, muscles, and neural activity.
Tabs and resizing preserve the running simulation. Saved locomotion policies can
still settle or die through normal energy depletion.

The 30-value input schema is preserved. Without an episode deadline, the
`time_remaining` input stays at the neutral value 1; the saved duration only
scales cumulative muscle-energy observations. The saved target offset remains
an input reference without a completion condition. Saved physics, muscle settings,
and default body remain compatible. Gravity/friction overrides are rejected.
Rules and the old steering controller remain available without `--brain`.

Timed headless experiments and saved comparison results remain available through
`eidolon.train` and `eidolon.evolve evaluate`. Live Learning uses timed attempts;
a continuous Survival run includes biology and is not a timed
evaluation benchmark. To observe the included brain:

```powershell
.\.venv\Scripts\python.exe -m eidolon --brain experiments\phase2b2\best.json --training-run experiments\phase2b2 --view split --seed 101
```

Continuous-launch regression validation: **491 tests passed on Python 3.12 and
3.14**. SDL checks kept the saved brain active beyond its old 12-second cutoff,
across all tabs and pause/resize/restart. Timed held-out evaluation still matches
the published reference results exactly. Native Windows graphics remain unverified.

## Living biomass physics

`colony.py` owns particle positions/velocities/masses/radii, stable node IDs,
adjacency, local triangles, and a separate list of elastic bonds. In detailed mode, a seeded,
lopsided staggered mesh supplies 85–98 nodes and roughly 220–255 local bonds.
At the default size, core node radii are **3.3–3.7 world units**, previously about
8; neighboring centers are 9 units apart. The overall body remains a similar size.
Layouts with 62, 88, 98, and 108 nodes were compared during development; the
85–98 range balances local deformation and CPU cost. Neighbor bonds preserve
local tissue without long supports that resist pulling the entire body.
Adjacency, membrane faces, and connected
components are rebuilt only when bonds change. Nodes retain stable IDs throughout
tearing and reconnection; no fragments become independent organisms.

The solver uses velocity prediction followed by **XPBD-style distance constraints**:

- Physics substeps are at most **1/120 second**, with five constraint iterations.
- Compliance is scaled by timestep squared; inverse mass weights the corrections.
- Relative-velocity damping limits jitter and angular drift.
- A deterministic spatial grid finds local collision candidates instead of
  comparing every node pair during every solver iteration. It is rebuilt after
  constraint corrections, including during growth and interaction.
- Node separation and wall projection prevent collapse and keep the membrane in
  the world. Overlapping visual membranes represent one connected tissue mass.
- A hard extension limit and a velocity cap protect against unusually strong
  impulses. Locomotion eases near walls to avoid packing tissue into a corner.
  Nonfinite input/state raises an error rather than silently reseeding.

The active controller requests a movement direction/speed. This becomes traction
on supporting particles only. Neighboring particles
follow through the bonds with different velocities and small delays. There is no
rigid translation or prescribed sprite rotation during locomotion. Modest
traveling rest-length contractions (up to 6.5% when moving) and a 1.5% idle pulse
act through the constraints, so stretching and recovery are physical node motion.

Biological position is the mass-weighted centroid of the largest component
(ties use the lowest node ID). Movement energy uses its material displacement rather
than centroid jumps when membership changes. Growth changes node radii and bond rest-length targets;
nodes relax toward the expanded shape instead of being instantly rescaled. Node
count remains constant. Death stops decisions/feeding and switches the body to
passive dynamics, so an airborne corpse falls rather than freezing in place.

## Grabbing, tearing, and recovery

`interaction.py` provides a damped spring constraint independent of Pygame. It
preserves the initial click offset and applies force during physics substeps;
it never teleports a node. Force is capped at **14,000 × growth scale**, while
the existing **220 × √scale** velocity cap and membrane-aware walls still apply.
Controller braking is disabled on a held component, allowing the spring to pull
material gradually instead of fighting the entire colony's locomotion forces.

Only bonds incident to the held node can tear. Both sustained strong pulling
(at least **4,500 × scale** force) and **18% tensile strain for 0.12 simulation
seconds** are required. At most one bond breaks every 0.05 seconds. Brief pointer
jumps and ordinary locomotion do not cause uncontrolled cascades. The force
threshold accounts for the new floor load and friction during gentle pulling.

Unheld fragments seek the nearest part of the largest component. The main
component slows while fragments recover. Horizontal travel requires floor
support; released airborne fragments fall before traveling along the floor.
Contact capture reaches 1.65 node
spacings and uses velocity damping instead of rejecting fast contact. Healing
waits for pointer release and a **0.45-second simulation-time tear cooldown**.
It adds multiple local bonds where neighbors are available, with at most eight
new bonds per repair pass and eight neighbors per node. Capture starts at the
actual gap, then smoothly relaxes toward normal spacing; no node is teleported.

Repair continues after the graph becomes connected. Weak attachments search up
to two spacings to find additional support, preventing the old single-bridge
failure. New bonds have a grace period before they can tear again. Held
fragments remain under the pointer constraint until release. Recovery is
deterministic and uses no extra randomness.

Fragments share the original colony's energy, growth, and survival state.
Sensing/feeding use the largest component's centroid. Reconnection restores
connectivity, not every original bond or the original pose. Tearing currently
targets the grabbed node's bonds rather than modeling arbitrary material cracks.

## World and basic life

The world remains **1200 × 800 world units**, independent of window dimensions.
The fit camera letterboxes uniformly as needed. Resizing changes only display
scale, never node positions, bond lengths, biological size, or world state.
There are no pan/zoom controls in this milestone.

In Survival, food appears as small green dots: initially 32, with a default cap
of 64 and one regenerated every three simulation seconds. Food now spawns
directly on the ground, with zero velocity. A seeded choice inside the largest
free gap spreads resources without overlaps or awkward clusters. Default
clearances are 64 units from side walls, eight extra units between food edges,
and six extra units from the current biomass membrane. Crowded ground defers
spawns rather than placing food inside the organism or another resource.
A full world discards missed spawns. Each resource is consumed once and provides
18 energy/nutrition units. `WorldConfig.food_distribution`, `food_spacing`,
`food_wall_clearance`, and `food_body_clearance` configure placement; `"falling"`
retains bounded, overlap-checked airborne spawning for future experiments.
Resources have persistent position, velocity, radius, and mass (default 1,
configurable with `WorldConfig.food_mass`). Gravity still applies to displaced
or airborne resources: they land at `world.height - radius`, lose lateral speed through friction,
and sleep when stationary on the floor. Rendering interpolates previous/current
food positions without changing the physics. Equal masses are not required for
equal free-fall acceleration.

The organism senses food within 250 units of its centroid and eats within a
34-unit interaction radius scaled by growth, plus food radius. In rules mode, without food it
wanders and steers away from boundaries. Steering is rate-limited, while the
colony's actual response depends on its forces, damping, and elastic deformation.
The default 55 world units/second is a commanded speed, not an instantaneous
kinematic velocity.

Energy starts at 70/100. Metabolism costs `1 × scale²` per second; movement costs
0.025 energy per world unit actually traveled. Eating restores energy up to the
maximum and accumulates nutrition. Growth requires energy above 45%, proceeds at
at most 0.01 scale units/second, spends energy, and stops at 1.6×. Conservative
clearance also limits growth near side walls and the ceiling. Floor contact does
not block growth: radii and rest lengths expand while the floor supports the
body, allowing it to grow upward. `health` remains energy-derived vitality
(0–100), not an independent injury model.

At zero energy the colony dies: age, growth, decisions, and feeding stop. Tissue
continues passive gravity/contact/elasticity; automatic gait, propulsion,
grabbing, and fragment recovery stop. Food continues falling and regenerating.
Pause freezes every simulation system and random
state; restart regenerates the original graph and resource supply. Speed affects
every system, including physics and regeneration. Render interpolation uses the
previous/current node positions captured at each fixed simulation update.

`WorldConfig` and `LifeConfig` expose resource and survival settings. For a
starvation demonstration, run this Python snippet with the installed interpreter:

```python
from eidolon.app import Application
from eidolon.config import AppConfig, WorldConfig

Application(AppConfig(mode="survival", world=WorldConfig(
    initial_food=0, regeneration_interval=0,
))).run()
```

Both `SurvivalRules` and `NeuralController` implement
`Controller.decide(Senses) -> Decision`, without changing biology or physics.
Feeding and energy remain colony-wide systems in this prototype.

## Flat membrane rendering

`colony_rendering.py` blends cached, smooth density stamps and intact local tissue
into one coverage mask per connected component. Nodes are not displayed as large
circles. A smooth coverage threshold and 2× downsampling soften the membrane;
flat seeded pigment, four tiny material grains per node, and one restrained
internal core supply texture. The grains are anchored to tissue and have no
independent physics, decisions, or lifetime. There is no 3D lighting model.

Rendering never integrates or repositions nodes. It only interpolates stored world
coordinates and applies the camera. The old appearance helpers remain available
for compatibility/tests, but the application's active body renderer uses only
the colony mesh. The appearance generator supplies its colors. Each connected
component gets a local buffer so detached tissue remains visible; missing bonds
also remove their membrane faces. Separated components never share a density
field; after reconnection, their tissue blends in one field without an artificial
seam. Buffer sizes adapt to actual extents and the cache holds at most eight
sizes. The terrarium and UI layout are unchanged.

## Performance and tests

Target: **60 FPS at 1280 × 720**, Windows 11 / Python 3.14, Intel i5-1135G7,
Iris Xe, 8 GB RAM. No fluid solver, physics engine, ML framework, or new runtime
dependencies were added. One bounded I/O worker saves live progress; training
evaluation remains sequential on the main thread. The solver handles a bounded particle graph;
food queries scan the configured cap (64 by default). Local render buffers and
light stamps are reused, and HUD label caching remains bounded.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --controller neural --seed 17
.\.venv\Scripts\python.exe tools\benchmark_neural.py --iterations 20000
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --interaction
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --interaction --scale 1.6
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --no-propulsion
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --muscle-test wave
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --body simple --muscle-test wave
.\.venv\Scripts\python.exe tools\benchmark_muscles.py --seed 17
.\.venv\Scripts\python.exe tools\benchmark_muscles.py --seed 17 --floor-friction 0
.\.venv\Scripts\python.exe tools\stress_colony.py --seed 17 --cycles 6 --scale 1.6
.\.venv\Scripts\python.exe tools\stress_colony.py --seed 17 --cycles 6 --scale 1.6 --muscles
.\.venv\Scripts\python.exe tools\stress_colony.py --body simple --seed 17 --cycles 6 --muscles
.\.venv\Scripts\python.exe tools\benchmark_laboratory.py --brain experiments\phase2b2\best.json --training-run experiments\phase2b2 --view split --frames 600
.\.venv\Scripts\python.exe tools\benchmark_laboratory.py --mode live --view split --frames 7200
```

The benchmark measures uncapped physics/update/draw/presentation and reports node
and bond counts. `--interaction` includes scripted pulling, tearing, and recovery,
and reports component and bond-change counts. `--scale 1.6` exercises maximum
growth after warmup. `--headless` selects SDL's dummy driver for cloud runs; it does
not measure native Windows presentation or guarantee performance on target hardware.
The display-free stress tool checks repeated tearing/recovery, identity and mass
conservation, bond accounting, neighbor limits, walls, and finite/capped velocities.
`--muscles` adds actuator commands to the existing stress tool while retaining
compatibility fragment recovery. `benchmark_muscles.py` instead disables propulsion
and compares an active wave against a passive body: 3 seconds settling, 16 seconds
activation, 4 seconds relaxation by default. It removes food, metabolism, and
travel costs to isolate actuation; muscle energy costs remain enabled. It reports
post-settling displacement and passive-baseline-adjusted displacement, checking
conservation, bounded motion, and complete relaxation. With friction zero and no
side-wall contact, internal contractions should not move the horizontal mass center.

Live Learning validation: **512 tests passed on Python 3.12 and 3.14**. New checks
compare two incremental generations with headless training exactly, including
weights, scores, and RNG state; cover resume, retries, archives, frame budgets,
single-writer locks, failed/coalesced saves, actual neural/muscle readouts, mode
and tab preservation, and grounded spawn clearances. Existing survival, physics,
model compatibility, and headless tests remain passing. An SDL window lifecycle
smoke test completed real generations, exercised tabs and controls, and restored
saved progress after closing.

Linux / Python 3.14.8 / SDL dummy **Live Learning** at 1280 × 720, after 60 warmup
frames, including inference, physics, drawing, automatic resets, and saves:

| View | Measured frames | Mean ms/frame | p95 ms/frame | Max ms/frame |
| --- | --- | --- | --- | --- |
| Simulation | 7,200 | 1.492 | 1.857 | 7.930 |
| Split View | 7,200 | 2.527 | 2.923 | 9.083 |
| Training | 1,800 | 1.125 | 1.398 | 5.980 |

Simulation and Split each completed a full default generation and 14,494 physics
ticks; all three runs remained stable. These are cloud CPU/display-driver
measurements, **not native Windows/Iris Xe validation**. The laboratory benchmark
uses a disposable run directory unless `--live-run` is supplied.

Original Phase 2B.2 validation: **484 tests passed on Python 3.12 and 3.14**, including exact
checkpoint continuation, multi-seed evaluation, muscle brain validation, unchanged
steering-model loading, no propulsion, headless/replay trajectory equivalence,
actual activation snapshots, draw purity, tabs/camera/resize, and saved-metric
reload. CLI/UI checks were rerun after final display/launch fixes. SDL lifecycle
checks exercised all four views, pause/resize/reset, and clean shutdown with the
saved reference brain.

Linux / Python 3.14.8 / SDL dummy laboratory measurements at 1280 × 720, using the
reference brain and metrics (600 measured frames after 60 warmup):

| View | Mean ms/frame | p95 ms/frame | Max ms/frame |
| --- | --- | --- | --- |
| Simulation replay | 1.402 | 1.750 | 3.423 |
| Neural Network | 1.536 | 1.880 | 3.111 |
| Saved Training | 1.398 | 1.728 | 3.543 |
| Split View | 2.322 | 2.571 | 3.329 |

Physics and real neural decisions continued in every view. A sequential generation
of eight candidates × three seeds × 12-second episodes took **9.41 seconds** on
this cloud CPU, including reset/settling and inference. This is a throughput
sample, not an i5/Windows guarantee. The 20-generation reference experiment
evaluated **480 candidate/seed episodes (5,760 active simulation seconds)**; all
reported stable. Measurements and the evaluation results are stored with the
reference experiment. Native Windows/Iris Xe responsiveness remains unverified.
Additional stability-only checks kept the same brain weights and ran seeds
101/102 for 60 seconds in each body mode, with a distant target to avoid early
completion. All four episodes stayed stable through the time limit (240 simulated
seconds). These altered-horizon/body checks are not the held-out fitness protocol.

Previous Phase 2B.1 body-rendering reference:
Linux / Python 3.14.8 / NumPy 2.5.4 / SDL dummy measurements at 1280 × 720
(600 measured frames after 60 warmup frames per run):

| Body | World seed | Scenario | Mean ms/frame | p95 ms/frame | Max ms/frame |
| --- | --- | --- | --- | --- | --- |
| Simple | 17 | Rules compatibility baseline | 1.797 | 2.282 | 4.519 |
| Simple | 17 | Neural compatibility baseline | 1.606 | 2.029 | 3.024 |
| Simple | 17 | Wave, propulsion disabled | 1.460 | 1.960 | 3.047 |
| Simple | 17 | Wave + pulling, 1.6× growth, propulsion disabled | 2.038 | 2.294 | 6.611 |
| Detailed | 17 | Wave, propulsion disabled | 6.465 | 6.894 | 11.062 |

These are cloud measurements, not native Windows or Iris Xe verification.
The samples fit the 16.67 ms frame budget; native 60 FPS responsiveness still needs
Windows testing. Muscle-enabled stress checks completed **18 pull/recovery cycles
over 369 simulation seconds** across both topologies, including 1.6× growth, without lost nodes or
force/velocity/boundary/neighbor-limit violations. Node identity, mass, and bond
accounting remained intact. Six additional 120-second random-action episodes
(720 simulated seconds, seeds 0/17/42 in both bodies) remained stable, reached
their time limits, and spent 131–134 muscle energy each without starvation.

Headless laboratory episodes, seed 17, 120 Hz, 0.1-second actions, 20 seconds,
target 100 units right, after three seconds settling; propulsion disabled:

| Body | Pattern | Distance | Reward | Muscle energy | Physics ticks/s |
| --- | --- | --- | --- | --- | --- |
| Simple | Idle | −0.005 | −0.199 | 0 | 4983 |
| Simple | Random | −19.615 | −0.747 | 21.596 | 4706 |
| Simple | Scripted | +9.560 | −0.230 | 11.875 | 4828 |
| Detailed | Idle | +0.249 | −0.064 | 0 | 424 |
| Detailed | Random | −7.240 | −0.620 | 21.704 | 408 |
| Detailed | Scripted | −9.144 | −0.463 | 11.878 | 407 |

Displacement is in world units; negative means left. Frictionless horizontal
mass-center drift stayed below 1e−7 units. Airborne zero-gravity tests also retain
the mass center despite contractions. These checks exclude drive/teleport helpers
and demonstrate contact-driven movement, not a trained, fast, or directional gait.
Throughput includes stepping, validation, observations, and reward accounting,
excluding reset/settling. These rates are cloud CPU measurements.

Tests cover seeded connected graphs, force propagation, stretch/recovery,
self-collision, walls, timestep subdivision, growth, deterministic trajectories,
prolonged stability, membrane connectivity, render purity, and unchanged life
systems. Camera tests verify that resizing does not modify physical nodes or
world state. Interaction tests cover picking, click offsets, capped forces,
gentle versus sustained pulls, local breakage, graph updates, fragment recovery,
reconnection, deterministic pointer replay, and lifecycle cancellation. Repeated
tear/recover cycles check bounded positions/velocities and retained node counts.
Recovery regressions include fast close contact, multiple attachments, continued
repair after first capture, cooldowns, and dense growth. A spatial-grid test checks
candidate completeness against exact distances across cell boundaries.
Earlier relevant lifecycle, resource, energy, pause/reset, and
fixed-step tests remain; former rigid-movement assertions now check force-driven
acceleration and actual travel. Legacy appearance helpers retain their unit tests.
Neural tests check a hand-calculated forward pass, initialization, normalization,
invalid parameters/files, exact JSON round trips, controller/CLI selection,
unchanged survival mechanics, pause/death, restart, and fixed-step reproducibility
at different rendering rates. Drawing does not call the network.
Grounded checks cover configurable mass-independent gravity, no flight forces,
passive settling, floor traction/friction, physical food and collisions,
interpolation, grounded growth, passive corpses, deterministic episode resets,
and a subprocess that runs the rules/physics core without Pygame or NumPy.
Muscle checks cover deterministic material assignments, independent/rate-limited
activation, bounded shortening, relaxation, energy budgets, starvation, actual
bond deformation/recovery, no artificial propulsion or airborne translation,
traction-dependent displacement, nearby-fragment repair, conservation, unchanged
neural parameters, fixed-step/reset replay, manual key release, pause/resize,
and HUD render purity at three sizes. Manual and wave modes also passed an SDL
dummy lifecycle check with pause, resize, restart, and clean exit.
Laboratory checks add exact ten-node connectivity, all eight channels' physical
contraction/recovery, both body modes, 30 observations/eight actions, deterministic
pattern/reset replay, fixed action intervals, exact duration truncation, signed
reward/cycle cancellation, one-time target bonuses, fuel-independent long episodes,
fragment-safe centroids, physics-fault penalties, and a JSON CLI subprocess with
Pygame and NumPy imports blocked. Desktop SDL checks include both controllers and
manual muscles with the simple body, grabbing, resizing while paused, reset, and exit.

GitHub Actions covers Python 3.11–3.14 on Linux/Windows. Configuring CI does not
establish that it has run. Native Windows graphical compatibility and Iris Xe
performance require a Windows check: watch the colony forage and deform, resize,
try gentle and forceful pulls followed by release, pause/restart/speed, and exit
with Escape and the window close button.
**Appearance acceptance requires manual Windows review; automated masks and
Linux previews do not establish visual approval.**

## Architecture

```text
src/eidolon/
  __main__.py          CLI and launch error handling
  app.py               Pygame lifecycle and controls
  config.py            Validated world/life/application settings
  timing.py            Fixed-step scheduling and render interpolation alpha
  simulation.py        World/biology/behavior/physics coordination
  colony.py            Nodes, bonds, constraints, damping, and collisions
  interaction.py       Force-limited pointer constraint, without display dependencies
  spatial.py           Deterministic local neighbor grid
  locomotion.py        Switchable, floor-only compatibility propulsion
  muscles.py           Eight material regions, bounded activations, energy, test pattern
  training.py          Fixed-action headless episodes, observations, signed rewards
  train.py             JSON experiment CLI and throughput measurements; no optimizer
  muscle_controller.py Separate 30-input/eight-muscle policy and brain bundle
  evolution.py         Sequential mutation/selection, evaluation, checkpoints
  evolve.py            Training/resume/held-out comparison CLI
  live_learning.py     Budgeted visible candidates, automatic retries, live progress
  run_store.py         Single-writer locks and bounded atomic snapshot I/O
  experiment_io.py     Bounded JSON and atomic experiment snapshots
  replay.py            Graphical adapter over exact headless episode stepping
  laboratory.py        Tabs, actual network snapshots, live/saved reward graphs
  colony_rendering.py  Flat membrane from physical node positions
  biology.py           Colony-wide biological state
  behavior.py          Replaceable sensory/decision interface and survival rules
  neural_network.py    NumPy feedforward math, validation, JSON persistence
  neural_controller.py Normalized sensory adapter and neural decisions
  world.py             Seeded capped resources and world time
  camera.py            Display-independent camera transforms
  rendering.py         Terrarium, resources, and HUD
  organism*.py         Seeded palette and legacy appearance helpers
tests/                 Physics, biology, rendering, and lifecycle tests
tools/                 Optional rendering benchmark and physics stress checks
```

Rules-based simulation and physics run without Pygame, NumPy, or a display;
neural decisions add only the existing NumPy dependency. In Linux/cloud
tasks, use the existing checkout (no separate worktree is needed):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m eidolon  # Requires a graphical desktop
```

This is a simplified 2D elastic graph, not a continuum fluid or a biological
material model. Recovery can be interrupted by death; dead tissue settles
passively rather than continuing to reconnect. Body evolution, cognitive memory,
and reinforcement learning are outside this milestone. **Phase 2B.3 has not been started.**
