# EIDOLON

An experimental artificial-life sandbox for Windows desktop. **Phase 1A.1:
Symbiote Visual Rework** gives the single organism an irregular alien silhouette,
thick muscular appendages, overlapping translucent folds, and an embedded glowing
core. Its tissue contracts and deforms while the specimen gently drifts and turns.
These are procedural idle animations; there are no neural networks, memory,
genetics, evolution, reproduction, or environmental interactions.
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
| Space | Pause/resume animation; the window remains responsive |
| R | Restart the same specimen at time zero; retain pause and speed |
| − / + (or =), including numeric keypad | Change animation speed in 0.25× steps, from 0.25× to 2× |
| `--seed INTEGER` | Choose a reproducible appearance, including negative seeds |

The same seed, parameters, and simulation time produce the same appearance and
pose without consuming global random state. Different seeds vary the specimen;
only one organism exists at a time. Exact rasterized pixels can differ between
Pygame/platform versions.

This visual rework changes the appearance associated with older Phase 1A seeds.
The same seed remains reproducible within the new appearance generator. To compare
the four body families, try `--seed 0` (blade), `--seed 2` (crown), `--seed 3`
(mantle), and `--seed 8` (cleft).

The default rendering limit is 60 FPS; `--max-fps 0` disables the limit and can
increase CPU use. Simulation updates use a separate fixed 60 Hz timestep.
Rendering can run faster or slower without changing that timestep. Long stalls
are clamped to 0.25 seconds and catch-up is capped at eight updates per frame;
excess elapsed time is intentionally dropped to preserve responsiveness. Simulation
time therefore falls behind wall time during sustained overload or debugging pauses.
Defaults and timing limits live in `src/eidolon/config.py`.

Rendering interpolates the previous/current animation clocks using the fixed-step
accumulator, with up to one simulation step of visual latency. All motion is a
function of this clock; rendering never advances simulation state. Pausing freezes
the organism's animation rather than merely hiding updates. Speed changes scale
animation time without changing the simulation timestep or frame limit.

## Procedural appearance

`generate_appearance(seed)` in `src/eidolon/organism.py` returns immutable data:

| Parameters | Purpose |
| --- | --- |
| Membrane, tissue, accent, and core RGB colors | Teal, blue, violet, rose, or mineral-gold tissue with contrasting internal light |
| Body profile | Mantle: elongated and folded; cleft: deeply recessed; crown: unequal lobes; blade: sweeping lateral anatomy |
| Width/height, shear, fold depth, lobe amplitude, phase | Seeded proportions and a shared, continuously deforming contraction field |
| Ridge count and membrane opacity | Embedded curved skeletal struts and overlapping translucent tissue |
| 7–11 appendages | Each stores attachment position, length, curl, sway, frequency, phase, and thickness |
| Core radius, elongation, offset, pulse rate, phase | A profile-specific location, irregular luminous sheath, and subtle pulse |

Dimensions are relative to the nominal body radius; angles use radians and
frequencies use radians per animation second. Appearance is generated once.
The stored tendril `angle` encodes a periodic position along the body contour;
roots follow its local outward normal instead of forming evenly spaced radial
antennae. A broad root starts beneath the body and becomes a smoothly tapered
filled ribbon. Body contraction drives proximal motion, with smaller delayed
waves toward the tips and modest per-appendage variation. These are analytic
animations, not muscle physics or environmental behavior.

Closed Catmull–Rom contours provide genuinely different anatomy rather than
perturbing an oval. The core moves with the same tissue deformation, and masked
folds, curved struts, and illuminated conduits connect it to the surrounding mass.
`Symbiote` holds only appearance and animation clocks; `OrganismPose` contains
read-only drift, rotation, breathing, and core pulse values. No image assets,
audio initialization, GPU-specific features, workers, or new dependencies are used.

## Performance

The target is **60 FPS at 1280 × 720** on an i5-1135G7 / Iris Xe / 8 GB RAM.
The renderer uses 80 membrane samples and 24 segments per appendage. Filled ribbon
meshes with a small number of shading strips provide rounded muscle volume;
masked tissue layers and soft cached lighting add depth without fluid simulation.
Sampling positions, radial light stamps, and three local surfaces (drawing,
compositing, and silhouette mask) are reused. NumPy light calculations run only
when appearance or view scale changes. There are no particles, real-time blurs,
fluid simulations, or per-frame full-window effect surfaces. Sprite scale is
capped and drawing is clipped to the chamber after resizing.
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

Tests cover settings, seeded appearance, all four profiles, noncircular/asymmetric
contours, continuous deformation, broad attached roots, tapering, connected
silhouettes, rendering/cache reuse, and unchanged simulation controls. The existing
fixed-step timing, interpolation, catch-up, startup, resize, shutdown, and CLI tests
remain in place. SDL's dummy video/audio drivers make the tests suitable for a headless
machine. The application tests create real Pygame surfaces but inject events and
elapsed times to keep assertions deterministic.

GitHub Actions is configured to run these tests on Linux and Windows with Python
3.11–3.14. A configured CI job is not evidence that it has run. Headless tests do
not verify native Windows graphics, interactive resizing, or performance on the
target Intel Iris Xe hardware. Those require a manual Windows check: launch the
application, confirm smooth membrane/tendril motion and a pulsing core, try the
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
  simulation.py          Single-organism ownership and animation controls
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

Phase 1B is outside this implementation and requires explicit approval.
