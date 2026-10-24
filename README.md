# EIDOLON

An experimental artificial-life sandbox for Windows desktop. Phase 0 establishes
an empty, dark simulation chamber and a maintainable Python foundation.
There are no organisms, neural networks, genetics, memory, or evolution yet.
No cloud APIs, external AI services, credentials, or network connection are
required to run the installed application.

## Windows setup (PowerShell)

Install **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/)
with the Python launcher enabled, and install Git. Python 3.11 or newer is required;
3.12 is the recommended starting version.

```powershell
git clone https://github.com/Schwzz/Eidolon.git
cd Eidolon
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m eidolon
```

The explicit interpreter paths work without activating the environment or changing
PowerShell execution policy. If you already have the checkout, start with `cd`
and skip cloning. Runtime dependencies are **Pygame-CE** and **NumPy**; the `dev`
extra adds **pytest**. NumPy is reserved for future numerical models and is not
used to invent artificial-life functionality in this phase. Install `pygame-ce`,
not the separate `pygame` distribution: both use the `pygame` import name and
should not share a virtual environment.

## Run and configure

```powershell
.\.venv\Scripts\python.exe -m eidolon --max-fps 60 --width 1100 --height 720
.\.venv\Scripts\eidolon.exe --max-fps 120
```

Drag a window edge to resize. The minimum supported size is 480 × 320.
Close the window or press **Escape** to exit. **Ctrl+C** in the launching terminal
also releases Pygame resources and exits.

The default rendering limit is 60 FPS; `--max-fps 0` disables the limit and can
increase CPU use. Simulation updates use a separate fixed 60 Hz timestep.
Rendering can run faster or slower without changing that timestep. Long stalls
are clamped to 0.25 seconds and catch-up is capped at eight updates per frame;
excess elapsed time is intentionally dropped to preserve responsiveness. Simulation
time therefore falls behind wall time during sustained overload or debugging pauses.
Defaults and timing limits live in `src/eidolon/config.py`.

The initial scene uses flat fills, borders, and bundled fonts without external
assets, audio initialization, GPU-specific features, or background workers.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

Tests cover settings, empty simulation state, fixed-step timing across 30/60/144
render FPS, bounded catch-up, rendering, startup, resizing, shutdown, and CLI
settings. SDL's dummy video/audio drivers make the tests suitable for a headless
machine. The application tests create real Pygame surfaces but inject events and
elapsed times to keep assertions deterministic.

GitHub Actions is configured to run these tests on Linux and Windows with Python
3.11–3.13. A configured CI job is not evidence that it has run. Headless tests do
not verify native Windows graphics, interactive resizing, or performance on the
target Intel Iris Xe hardware. Those require a manual Windows check: launch the
application, confirm the empty chamber, resize in both directions, and exit with
both Escape and the window close button.

## Architecture

```text
src/eidolon/
  __main__.py   CLI and launch error reporting
  app.py        Pygame lifecycle, events, and main loop
  config.py     Validated immutable application settings
  timing.py     Fixed-step scheduling and bounded catch-up
  simulation.py Empty simulation state; no Pygame dependency
  rendering.py  Dark chamber layout and drawing; no simulation mutation
tests/          Unit and headless application tests
```

The application coordinates simulation updates and rendering. The simulation
accepts seconds per update and can run without a display. The renderer only draws
the chamber and recalculates layout from the current surface size. `finally`
cleanup releases Pygame even if the loop fails.

## Linux development / cloud workspace

Use the existing checkout in an isolated cloud task; a separate Git worktree is
not necessary. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m eidolon  # Requires a graphical desktop
```

Phase 1 is outside this foundation and requires explicit approval.
