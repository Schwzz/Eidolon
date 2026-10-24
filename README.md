# EIDOLON

An experimental artificial-life sandbox for Windows desktop. **Phase 2A.1:
Grounded Physics Foundation** makes the chamber a gravity-based 2D side view.
Biomass and food fall, collide with the floor, and settle. The existing survival
rules remain the default, and the seeded NumPy neural controller is retained.
The biomass appearance, grabbing, tearing/reconnection, energy, growth,
starvation, fixed world, and model format remain.

There is no training, learning, memory, evolution, reproduction, or second
organism. Neural weights are initially random; neural mode is an experiment,
not a trained survival policy. No ML frameworks, pretrained models, LLMs, cloud
APIs, credentials, or network connection are required after installation.
No muscle actuators, reaching/gripping, rewards, or reinforcement learning are
implemented in this phase.

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
```

| Control | Action |
| --- | --- |
| Left mouse button, hold and move | Grab a nearby node; pull gently to stretch or firmly to tear local bonds; release to allow recovery |
| Space | Pause/resume physics, biology, world clocks, and food regeneration |
| R | Restart the complete seeded world, colony graph, and biological state; retain pause and speed |
| − / + (or =), including numeric keypad | Change simulation speed in 0.25× steps, from 0.25× to 2× |
| Escape / window close | Exit and release Pygame resources |
| `--seed INTEGER` | Reproduce colony layout, masses, palette, food, and behavior |

The default window is 1280 × 720; the minimum is 480 × 320. Ctrl+C also exits.
The 60 FPS limit is configurable; `--max-fps 0` removes it and may increase CPU use.
Grabbing uses the current fit camera, including letterboxing. Empty clicks are
ignored. Mouse release, leaving the window, loss of focus, resizing, pause,
restart, death, and shutdown cancel the grab; press again to resume interaction.
Dragging beyond the chamber clamps the target to the world boundary.

The same seed, settings, and input/update sequence reproduce the simulation.
Colony generation, appearance colors, and food spawning use separate private
random generators. Rendering consumes no random numbers and changes no physics
state. Floating-point/raster results may differ across Python or platform versions.

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
contacts, grabbing/tearing, and nearby material reconnection remain. This is a
passive-physics switch for future controllers; it does not add muscles or training.
Public node IDs, positions, velocities, masses, adjacency, and bond rest lengths/
compliance remain available for future local actuation work.

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

## Living biomass physics

`colony.py` owns particle positions/velocities/masses/radii, stable node IDs,
adjacency, local triangles, and a separate list of elastic bonds. A seeded,
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

Food appears as small green dots: initially 32, with a default cap of 64 and one
regenerated every three simulation seconds. A full world discards missed spawns.
Each resource is consumed once and provides 18 energy/nutrition units. Spawn
clearance remains conservative so food stays accessible at maximum growth.
Resources have persistent position, velocity, radius, and mass (default 1,
configurable with `WorldConfig.food_mass`). They fall from the seeded spawn
region, land at `world.height - radius`, lose lateral speed through friction,
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

Application(AppConfig(world=WorldConfig(
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
Iris Xe, 8 GB RAM. No fluid solver, physics engine, ML framework, workers, or new
runtime dependencies were added. The solver handles a bounded particle graph;
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
.\.venv\Scripts\python.exe tools\stress_colony.py --seed 17 --cycles 6 --scale 1.6
```

The benchmark measures uncapped physics/update/draw/presentation and reports node
and bond counts. `--interaction` includes scripted pulling, tearing, and recovery,
and reports component and bond-change counts. `--scale 1.6` exercises maximum
growth after warmup. `--headless` selects SDL's dummy driver for cloud runs; it does
not measure native Windows presentation or guarantee performance on target hardware.
The display-free stress tool checks repeated tearing/recovery, identity and mass
conservation, bond accounting, neighbor limits, walls, and finite/capped velocities.

Phase 2A.1 validation: **332 tests passed on Python 3.12 and 3.14**. The default
12→16→2 network averaged **8.42 µs per forward pass** and **13.33 µs per complete
decision**, including normalization and validation, over 20,000 iterations.
Linux / Python 3.14.8 / NumPy 2.5.4 / SDL dummy measurements at 1280 × 720
(600 measured frames after 60 warmup frames per run):

| Controller | World seed | Scenario | Mean ms/frame | p95 ms/frame | Max ms/frame |
| --- | --- | --- | --- | --- | --- |
| Rules | 17 | Grounded | 6.395 | 7.849 | 12.611 |
| Neural | 17 | Grounded | 6.419 | 6.928 | 10.649 |
| Rules | 42 | Grounded | 5.473 | 6.240 | 8.885 |
| Neural | 42 | Grounded | 5.666 | 6.366 | 9.348 |
| Rules | 17 | Pulling/recovery, 1.6× growth | 6.925 | 8.465 | 20.279 |
| Neural | 17 | Pulling/recovery, 1.6× growth | 6.656 | 7.525 | 45.003 |
| Rules | 17 | Propulsion disabled | 6.905 | 8.014 | 12.992 |

These are cloud measurements, not native Windows or Iris Xe verification.
Average/p95 timings are below the 16.67 ms frame budget, but isolated cloud
spikes reached 45 ms; native 60 FPS responsiveness still needs Windows testing.
Grounded stress checks completed **18 recovery cycles over 369 simulation
seconds** across three seeds, including 1.6× growth, without lost nodes or
force/velocity/boundary/neighbor-limit violations. Node identity, mass, and bond
accounting remained intact.

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
passively rather than continuing to reconnect. Learning and evolution remain
outside this milestone. **Phase 2A.2 has not been started.**
