"""Uncapped live learning or continuous Survival with laboratory rendering."""

import argparse
from math import ceil, isfinite
import json
import os
import platform
import tempfile
from statistics import fmean
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain")
    parser.add_argument("--training-run")
    parser.add_argument("--mode", choices=("auto", "live", "survival"), default="auto")
    parser.add_argument("--live-run", help="optional live run to benchmark; otherwise uses a disposable run")
    parser.add_argument("--view", choices=("simulation", "neural", "training", "split"), default="split")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
    from eidolon.app import Application
    from eidolon.config import AppConfig
    from eidolon.laboratory import LaboratoryRenderer
    from eidolon.rendering import ChamberRenderer
    temporary = tempfile.TemporaryDirectory(prefix="eidolon-benchmark-")
    try:
        app = Application(AppConfig(body="simple", brain_path=args.brain, training_path=args.training_run, view=args.view,
                                    mode=args.mode, live_run=args.live_run or temporary.name))
        pygame.display.init()
        pygame.font.init()
        surface = pygame.display.set_mode((1280, 720))
        renderer = LaboratoryRenderer(ChamberRenderer(), view=args.view, metrics=app.saved_metrics)
        samples = []
        for frame in range(args.frames + 60):
            started = perf_counter()
            pygame.event.pump()
            app.simulation.update(1 / 60)
            renderer.draw(surface, app.simulation, 1)
            pygame.display.flip()
            if frame >= 60:
                samples.append((perf_counter() - started) * 1000)
        ordered = sorted(samples)
        print(json.dumps(dict(python=platform.python_version(), platform=platform.system(), driver=pygame.display.get_driver(),
                              view=args.view, replay=getattr(app.simulation, "is_replay", False), frames=len(samples), resolution=[1280, 720],
                              mean_ms=fmean(samples), p95_ms=ordered[ceil(len(samples) * 0.95) - 1], max_ms=max(samples),
                              elapsed_simulation_seconds=app.simulation.elapsed_seconds,
                              live_learning=app.mode == "live",
                              completed_generations=len(app.simulation.trainer.history) if app.mode == "live" else 0,
                              physics_ticks=app.simulation.total_ticks if app.mode == "live" else None,
                              physics_stable=all(isfinite(v) for node in app.simulation.colony.nodes
                                                 for v in (*node.position, *node.velocity))), indent=2))
    finally:
        if 'app' in locals():
            app.close()
        temporary.cleanup()
        pygame.quit()


if __name__ == "__main__":
    main()
