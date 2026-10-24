"""Optional uncapped single-organism benchmark; no performance assertions in CI."""

import argparse
import json
from math import ceil, isfinite
import os
import platform
from statistics import fmean
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--controller", choices=("rules", "neural"), default="rules")
    parser.add_argument("--network-seed", type=int, help="initialization seed for neural comparisons")
    parser.add_argument("--scale", type=float, default=1.0, help="Initial biological growth scale (1.0–1.6)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--interaction", action="store_true", help="Exercise repeated pulling and fragment recovery")
    args = parser.parse_args()
    if args.network_seed is not None and args.controller != "neural":
        parser.error("--network-seed requires --controller neural")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    # Set SDL options before importing Pygame.
    import pygame

    from eidolon.config import AppConfig
    from eidolon.rendering import ChamberRenderer
    from eidolon.simulation import Simulation

    try:
        config = AppConfig(width=args.width, height=args.height, seed=args.seed, controller=args.controller)
    except ValueError as error:
        parser.error(str(error))
    if not isfinite(args.scale) or not 1 <= args.scale <= config.life.max_scale:
        parser.error(f"--scale must be between 1 and {config.life.max_scale}")
    samples = []
    warmup = 60
    try:
        pygame.display.init()
        pygame.font.init()
        surface = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("EIDOLON rendering benchmark")
        renderer = ChamberRenderer()
        controller = None
        if config.controller == "neural":
            from eidolon.neural_controller import NeuralController

            controller = NeuralController.from_seed(config.seed if args.network_seed is None else args.network_seed)
        simulation = Simulation(config.seed, config.world, config.life, controller)
        simulation.life.body_scale = args.scale
        simulation.colony.set_growth(args.scale)
        peak_components = 1
        for frame in range(args.frames + warmup):
            started = perf_counter()
            if any(
                event.type == pygame.QUIT
                or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                for event in pygame.event.get()
            ):
                break
            if args.interaction:
                cycle_frame = frame % (config.simulation_hz * 8)
                if cycle_frame == 0:
                    origin = max(simulation.colony.nodes, key=lambda node: node.position[0]).position
                    simulation.begin_grab(origin)
                if cycle_frame < config.simulation_hz * 2.5:
                    simulation.move_grab((origin[0] + cycle_frame / config.simulation_hz * 140, origin[1]))
                else:
                    simulation.release_grab()
            # Exercise ten simulation seconds per 600 samples, uncapped.
            simulation.update(1 / config.simulation_hz)
            peak_components = max(peak_components, len(simulation.colony.components))
            renderer.draw(surface, simulation)
            pygame.display.flip()
            elapsed = (perf_counter() - started) * 1000
            if frame >= warmup:
                samples.append(elapsed)
        if not samples:
            print("No measured frames; benchmark closed during warmup.")
            return
        ordered = sorted(samples)
        print(json.dumps({
            "python": platform.python_version(),
            "platform": platform.system(),
            "pygame_ce": pygame.version.ver,
            "video_driver": pygame.display.get_driver(),
            "resolution": list(surface.get_size()),
            "seed": config.seed,
            "controller": simulation.controller_name,
            "initial_scale": args.scale,
            "nodes": len(simulation.colony.nodes),
            "visual_grains": len(renderer._colony_renderer._grains),
            "bonds": len(simulation.colony.bonds),
            "interaction": args.interaction,
            "peak_components": peak_components,
            "broken_bonds": simulation.colony.broken_bonds,
            "reconnected_bonds": simulation.colony.reconnected_bonds,
            "frames": len(samples),
            "mean_ms": round(fmean(samples), 3),
            "p95_ms": round(ordered[ceil(len(ordered) * 0.95) - 1], 3),
            "max_ms": round(max(samples), 3),
            "mean_fps": round(1000 / fmean(samples), 1),
        }, indent=2))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
