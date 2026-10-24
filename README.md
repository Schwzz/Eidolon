# EIDOLON

An experimental artificial-life sandbox for Windows desktop. **Phase 1B:
Environment & Basic Life** adds a bounded world, autonomous exploration and
feeding, energy, growth, and starvation to the existing 2D symbiote design.
One organism follows lightweight deterministic survival rules. There are no neural
networks, memory/learning, evolution, reproduction, or other organisms.
No cloud APIs, external AI services, credentials, or network connection are
required to run the installed application.

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

The explicit interpreter paths work without activating the environment or changing
PowerShell execution policy. If you already have the checkout, start with `cd`
and skip cloning. Runtime dependencies are **Pygame-CE** and **NumPy**; the `dev`
extra adds **pytest**. NumPy builds small radial glow stamps at startup or resize;
it does not run a fluid simulation or an AI model. Install `pygame-ce`,
not the separate `pygame` distribution: both use the `pygame` import name and
should not share a virtual environment.

## Run and configure

```powershell
.\.venv\Scripts\python.exe -m eidolon --seed 17 --max-fps 60 --width 1280 --height 720
.\.venv\Scripts\eidolon.exe --max-fps 120
```

Drag a window edge to resize. The minimum supported size is 480 × 320.
The default window is 1280 × 720. Close the window or press **Escape** to exit.
**Ctrl+C** in the launching terminal also releases Pygame resources and exits.

| Control | Action |
| --- | --- |
| Space | Pause/resume the complete world, including food regeneration and biology |
| R | Restart the seeded world, food supply, organism, and clocks; retain pause and speed |
| − / + (or =), including numeric keypad | Change simulation speed in 0.25× steps, from 0.25× to 2× |
| `--seed INTEGER` | Choose a reproducible world and appearance, including negative seeds |

The same seed, settings, and update/input sequence produce the same simulation.
Appearance generation and food spawning use separate private random generators;
rendering consumes neither. Exact rasterized pixels can differ between platforms.

Phase 1B preserves the Phase 1A.2 appearance generator and seed mapping. To compare
the five body families, try `--seed 0` (jelly), `--seed 2` (veil), `--seed 3`
(creeper), `--seed 4` (blob), and `--seed 8` (parasite).

The default rendering limit is 60 FPS; `--max-fps 0` disables the limit and can
increase CPU use. Simulation updates use a separate fixed 60 Hz timestep.
Rendering can run faster or slower without changing that timestep. Long stalls
are clamped to 0.25 seconds and catch-up is capped at eight updates per frame;
excess elapsed time is intentionally dropped to preserve responsiveness. Simulation
time therefore falls behind wall time during sustained overload or debugging pauses.
Defaults and timing limits live in `src/eidolon/config.py`.

Rendering interpolates world position, heading, body scale, and animation clocks
with up to one simulation step of visual latency. Rendering never advances the
world. Speed scales **all** simulation time, not just animation. Integration is
subdivided into steps no larger than 1/60 second so faster speed or larger caller
timesteps do not skip feeding interactions. Pause freezes the complete simulation
and random state. Dead organisms freeze while the world's food regeneration may
continue; restart restores the original seeded state.

## World, sensing, and survival

The default world is **1200 × 800 world units**, independent of the window. The
camera fits the complete terrarium uniformly and letterboxes when needed. Resizing
changes pixels per world unit, never world positions or physical body scale.
There are no pan/zoom controls in this phase.

Food appears as small green resource dots. Up to 64 may exist, with 32 initially
spawned and one regenerated every three simulation seconds. Full worlds discard
missed spawns rather than banking a burst. Each resource has a unique identifier,
is consumed once, and supplies 18 energy/nutrition units. Spawns stay inside an
inset reachable even by the fully grown organism.

The organism senses the nearest food within 250 units, turns toward it, and eats
within a 34-unit interaction radius scaled by growth (plus the food radius).
Without sensed food it wanders, with an inward steering bias near boundaries.
Turning is rate-limited. A conservative collision disk contains the complete
placeholder sprite, including its appendages, and keeps it inside the world.
It is a simple boundary model, not detailed tissue collision physics.

Energy starts at 70/100. Basal metabolism costs `1 × scale²` energy per second;
movement costs 0.025 energy per world unit actually traveled. Speed starts at 55
world units per second and decreases slightly with size. Food restores energy up
to the maximum and accumulates nutrition. Growth requires energy above 45%, uses
nutrition to set its target size, proceeds at at most 0.01 scale units/second, and
costs energy. It stops at 1.6× and waits if there is insufficient boundary clearance.
Growth changes the physical radius in world units, not the camera.

At zero energy the organism dies, stops deciding/moving/eating/growing, and its age
and animation freeze. `health` is currently energy-derived vitality (0–100), not an
independent injury model. The HUD shows energy, age, physical scale, food count,
alive/dead status, pause state, and simulation speed.

`WorldConfig` and `LifeConfig` in `config.py` expose dimensions, resource capacity,
regeneration, energy, movement, sensing, feeding, and growth settings. For example,
run this Python snippet with the installed virtual-environment interpreter to
observe starvation without food:

```python
from eidolon.app import Application
from eidolon.config import AppConfig, WorldConfig

Application(AppConfig(world=WorldConfig(
    initial_food=0, regeneration_interval=0,
))).run()
```

Decisions use the `Controller.decide(Senses) -> Decision` interface. The default
`SurvivalRules` reads an immutable sensory snapshot and has no learning or memory.
A future controller can replace it through `Simulation(controller=...)` without
changing world integration, energy accounting, or rendering.

## Procedural appearance

`generate_appearance(seed)` in `src/eidolon/organism.py` returns immutable data:

| Parameters | Purpose |
| --- | --- |
| Membrane, tissue, accent, and core RGB colors | Teal, blue, violet, rose, or mineral-gold tissue with contrasting internal light |
| Body profile | Blob: rounded/lopsided; parasite: elongated; veil: broad membrane; jelly: scalloped bell; creeper: low trailing mass |
| Width/height, shear, fold depth, lobe amplitude, phase | Seeded proportions and a shared, continuously deforming contraction field |
| Fold count, membrane opacity, and trailing direction | A few translucent 2D sheets and a coherent flow through the silhouette |
| 7–11 appendages | Each stores clustered attachment position, length, curl, sway, frequency, phase, thickness, and heading |
| Core radius, elongation, offset, pulse rate, phase | A profile-specific location, irregular luminous sheath, and subtle pulse |

Dimensions are relative to the nominal body radius; angles use radians and
frequencies use radians per animation second. Appearance is generated once.
The stored tendril `angle` encodes a periodic position along the body contour;
attachments occupy a limited trailing part of the contour. Roots start beneath
the body, follow its local outward normal, then gradually turn into the body's
shared flow direction. Long, tapering ribbons form S-bends and curling tips,
rather than rigid outward horns or a radial star. Body contraction drives
proximal motion, with delayed waves toward the tips and modest per-appendage
variation. Small membrane webs fuse neighboring roots. Surface deformation stays
analytic; biological movement and survival are separate world systems.

Closed Catmull–Rom contours interpolate rounded masses with unequal lobes and
gentle recesses. Each body plan has its own contour, attachment region, trailing
direction, and internal core location. The core moves with the same tissue field.

## Flat 2D rendering

The art uses one flat ink color across each appendage, a translucent body fill,
and a few irregular, low-contrast membrane washes. All shapes are drawn in the
same 2D plane. There are **no tube highlights, directional surface lighting,
bevels, rib bands, armor plates, concentric shells, or projected 3D geometry**.
The old blade/crown/cleft/mantle profiles have been replaced.

Only the embedded core emits a cached soft radial glow. Its irregular luminous
organ and two faint flowing filaments sit within the tissue silhouette; they do
not behave like a glossy sphere or a mechanical lamp. Flat translucent sheets
and negative spaces, rather than simulated 3D volume, provide depth and detail.

`Symbiote` holds only appearance and animation clocks; `OrganismPose` contains
read-only drift, rotation, breathing, and core pulse values. No image assets,
audio initialization, GPU-specific features, workers, or new dependencies are used.

## Performance

The target is **60 FPS at 1280 × 720** on an i5-1135G7 / Iris Xe / 8 GB RAM.
The renderer uses 96 membrane samples and 32 segments per appendage. A single
filled silhouette replaces the former per-appendage volume/shading strips.
Sampling positions, the core light stamp, and three local surfaces (drawing,
compositing, and silhouette mask) are reused. NumPy light calculations run only
when appearance or view scale changes. There are no particles, real-time blurs,
fluid simulations, or per-frame full-window effect surfaces. In the simulation,
sprite radius is physical world radius times camera scale (rounded to pixels);
there is no window-dependent biological resizing. Local surfaces rebuild only
when pixel radius changes through growth or resizing. Drawing stays in the world
viewport. Food queries are linear scans over the configured cap (64 by default), and HUD text
caching is bounded rather than retaining every age/energy label indefinitely.
Tissue compositing is restricted to the current body's bounding rectangle, rather
than processing transparent margins around its appendages.

Run the optional uncapped benchmark on the target machine:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_rendering.py --frames 600
```

It measures update, draw, and display presentation at the requested resolution;
it is not a guarantee of sustained performance. `--headless` uses SDL's dummy
driver for Linux/cloud checks and does not measure native Windows presentation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

Tests cover movement, sensing/approach, one-time feeding, energy costs and limits,
growth requirements and caps, death, boundaries, seeded resources/regeneration,
complete pause/reset, replaceable controllers, deterministic simulation, and
camera resizing without world or physical-size changes. Existing tests cover
settings, seeded appearance, all five rounded/asymmetric profiles,
clustered nonradial attachments, continuous deformation, broad attached roots,
tapering, connected silhouettes, flat appendage cross-sections, rendering/cache
reuse, and simulation controls. The existing
fixed-step timing, interpolation, catch-up, startup, resize, shutdown, and CLI tests
remain in place. SDL's dummy video/audio drivers make the tests suitable for a headless
machine. The application tests create real Pygame surfaces but inject events and
elapsed times to keep assertions deterministic.

GitHub Actions is configured to run these tests on Linux and Windows with Python
3.11–3.14. A configured CI job is not evidence that it has run. Headless tests do
not verify native Windows graphics, interactive resizing, or performance on the
target Intel Iris Xe hardware. Those require a manual Windows check: launch the
application, watch food consumption and energy/growth changes, try the full-world
pause/reset/speed controls, resize in both directions, and exit with both Escape
and the window close button. Native Windows graphics and target-hardware
performance have not been verified in the Linux cloud environment.

## Architecture

```text
src/eidolon/
  __main__.py            CLI and launch error reporting
  app.py                 Pygame lifecycle, events, and main loop
  config.py              Validated immutable application settings
  timing.py              Fixed-step scheduling, catch-up cap, interpolation alpha
  simulation.py          World/biology/decision integration and complete reset
  biology.py             Biological state and read-only interpolated life poses
  world.py               Bounded coordinates and seeded food lifecycle
  behavior.py            Sensory/decision interface and simple survival rules
  camera.py              Display-independent fit camera and coordinate mapping
  organism.py            Seeded parameters, animation state, and sampled poses
  organism_geometry.py   Display-independent contours and appendage meshes
  rendering.py           Chamber layout and controls overlay
  organism_rendering.py  Local procedural drawing and cached effects
tests/                   Unit and headless application tests
tools/                   Optional rendering benchmark
```

The application coordinates simulation updates and rendering. The simulation
accepts seconds per update and can run without Pygame or a display. The renderers
read simulation data and recompute layout from the current surface size. `finally`
cleanup releases Pygame even if the loop fails. The original empty chamber remains
available to the chamber renderer when no simulation is supplied.

## Linux development / cloud workspace

Use the existing checkout in an isolated cloud task; a separate Git worktree is
not necessary. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m eidolon  # Requires a graphical desktop
```

Phase 2 is outside this implementation and requires explicit approval.
