# EIDOLON

An experimental artificial-life sandbox for Windows desktop. **Phase 1C-B:
Physical Interaction** adds grabbing, elastic pulling, local tearing, and fragment
recovery to the flexible colony of **23–26 biomass nodes**. The flat membrane
appearance, autonomous foraging, food, energy, growth, starvation, and fixed world
remain.

There are no neural networks, learning, evolution, reproduction, or multiple
organisms. No cloud APIs, credentials, or network connection are required after
installation.

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

## Living biomass physics

`colony.py` owns particle positions/velocities/masses/radii, stable node IDs,
adjacency, local triangles, and a separate list of elastic bonds. A seeded,
lopsided staggered mesh supplies 23–26 nodes and approximately 100–140 bonds.
Neighbor bonds preserve local tissue; more compliant longer bonds resist bulk
collapse without imposing a rigid pose. Adjacency, membrane faces, and connected
components are rebuilt only when bonds change. Nodes retain stable IDs throughout
tearing and reconnection; no fragments become independent organisms.

The solver uses velocity prediction followed by **XPBD-style distance constraints**:

- Physics substeps are at most **1/120 second**, with five constraint iterations.
- Compliance is scaled by timestep squared; inverse mass weights the corrections.
- Relative-velocity damping limits jitter and angular drift.
- Node separation and wall projection prevent collapse and keep the membrane in
  the world. Overlapping visual membranes represent one connected tissue mass.
- A hard extension limit and a velocity cap protect against unusually strong
  impulses. Nonfinite input/state raises an error rather than silently reseeding.

The rule controller requests a movement direction/speed. This becomes traction
on individual particles, stronger at the leading edge. Neighboring particles
follow through the bonds with different velocities and small delays. There is no
rigid translation or prescribed sprite rotation during locomotion. Modest
traveling rest-length contractions (up to 6.5% when moving) and a 1.5% idle pulse
act through the constraints, so stretching and recovery are physical node motion.

Biological position is the mass-weighted centroid of the largest component
(ties use the lowest node ID). Movement energy uses its material displacement rather
than centroid jumps when membership changes. Growth changes node radii and bond rest-length targets;
nodes relax toward the expanded shape instead of being instantly rescaled. Node
count remains constant. Death freezes all nodes and stops decisions/feeding.

## Grabbing, tearing, and recovery

`interaction.py` provides a damped spring constraint independent of Pygame. It
preserves the initial click offset and applies force during physics substeps;
it never teleports a node. Force is capped at **14,000 × growth scale**, while
the existing **220 × √scale** velocity cap and membrane-aware walls still apply.
Locomotion relaxes while the main component is held, allowing gentle stretching.

Only bonds incident to the held node can tear. Both sustained strong pulling
(at least **6,000 × scale** force) and **3.5% tensile strain for 0.12 simulation
seconds** are required. At most one bond breaks every 0.05 seconds. Brief pointer
jumps and ordinary locomotion do not cause uncontrolled cascades.

Unheld fragments seek the nearest part of the largest component. The main
component slows while fragments recover. Reconnection waits for close, reasonably
slow contact and a 0.7-second tear cooldown, then adds a compliant bridge at the
current separation to avoid snapping. Held fragments remain under the pointer's
constraint until released. Recovery is deterministic and uses no extra randomness.

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

The organism senses food within 250 units of its centroid and eats within a
34-unit interaction radius scaled by growth, plus food radius. Without food it
wanders and steers away from boundaries. Steering is rate-limited, while the
colony's actual response depends on its forces, damping, and elastic deformation.
The default 55 world units/second is a commanded speed, not an instantaneous
kinematic velocity.

Energy starts at 70/100. Metabolism costs `1 × scale²` per second; movement costs
0.025 energy per world unit actually traveled. Eating restores energy up to the
maximum and accumulates nutrition. Growth requires energy above 45%, proceeds at
at most 0.01 scale units/second, spends energy, and stops at 1.6×. Conservative
clearance also limits growth near walls. `health` remains energy-derived vitality
(0–100), not an independent injury model.

At zero energy the colony dies: age, physics, and feeding freeze. The world may
continue regenerating food. Pause freezes every simulation system and random
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

Decisions still use `Controller.decide(Senses) -> Decision`; a future controller
can replace `SurvivalRules` without rewriting biology, physics, or rendering.
Feeding and energy remain colony-wide systems in this prototype.

## Flat membrane rendering

`colony_rendering.py` draws the physical mesh as local filled triangles, rounded
bond connections, and overlapping node discs. Reusable local surfaces are drawn
at 2× resolution and downsampled for a soft outline. Flat pigment patches and a
few faint material lights preserve the seeded palette. There is no circular cell
rim, analytic squid body, rigid spike silhouette, or 3D lighting model.

Rendering never integrates or repositions nodes. It only interpolates stored world
coordinates and applies the camera. The old appearance helpers remain available
for compatibility/tests, but the application's active body renderer uses only
the colony mesh. The appearance generator supplies its colors. Each connected
component gets a local buffer so detached tissue remains visible; missing bonds
also remove their membrane faces. Buffer sizes adapt to actual extents and the
cache holds at most eight sizes. The intact colony's visual style is unchanged.

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
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600 --interaction
```

The benchmark measures uncapped physics/update/draw/presentation and reports node
and bond counts. `--interaction` includes scripted pulling, tearing, and recovery,
and reports component and bond-change counts. `--headless` selects SDL's dummy driver for cloud runs; it does
not measure native Windows presentation or guarantee performance on target hardware.

Tests cover seeded connected graphs, force propagation, stretch/recovery,
self-collision, walls, timestep subdivision, growth, deterministic trajectories,
prolonged stability, membrane connectivity, render purity, and unchanged life
systems. Camera tests verify that resizing does not modify physical nodes or
world state. Interaction tests cover picking, click offsets, capped forces,
gentle versus sustained pulls, local breakage, graph updates, fragment recovery,
reconnection, deterministic pointer replay, and lifecycle cancellation. Repeated
tear/recover cycles check bounded positions/velocities and retained node counts.
Earlier relevant lifecycle, resource, energy, pause/reset, and
fixed-step tests remain; former rigid-movement assertions now check force-driven
acceleration and actual travel. Legacy appearance helpers retain their unit tests.

GitHub Actions covers Python 3.11–3.14 on Linux/Windows. Configuring CI does not
establish that it has run. Native Windows graphical compatibility and Iris Xe
performance require a Windows check: watch the colony forage and deform, resize,
try gentle and forceful pulls followed by release, pause/restart/speed, and exit
with Escape and the window close button.

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
  colony_rendering.py  Flat membrane from physical node positions
  biology.py           Colony-wide biological state
  behavior.py          Replaceable sensory/decision interface and survival rules
  world.py             Seeded capped resources and world time
  camera.py            Display-independent camera transforms
  rendering.py         Terrarium, resources, and HUD
  organism*.py         Seeded palette and legacy appearance helpers
tests/                 Physics, biology, rendering, and lifecycle tests
tools/                 Optional performance benchmark
```

Simulation and physics run without Pygame, NumPy, or a display. In Linux/cloud
tasks, use the existing checkout (no separate worktree is needed):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m eidolon  # Requires a graphical desktop
```

This is a simplified 2D elastic graph, not a continuum fluid or a biological
material model. Recovery can be interrupted by death; dead tissue freezes rather
than continuing to reconnect. Learning and evolution remain outside this milestone.
**No subsequent phase has been started.**
