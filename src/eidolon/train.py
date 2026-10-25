"""Evaluate headless muscle episodes; this command does not train a network."""

import argparse
from dataclasses import replace
import json
import platform
from random import Random
from time import perf_counter

from eidolon.config import PhysicsConfig
from eidolon.muscles import wave_activations
from eidolon.training import EpisodeConfig, LocomotionEnvironment


def evaluate(environment: LocomotionEnvironment, pattern: str) -> dict:
    if pattern not in ("idle", "random", "scripted"):
        raise ValueError("pattern must be idle, random, or scripted")
    random = Random(environment.seed ^ 0xAC710)
    while not environment.done:
        if pattern == "idle":
            actions = (0.0,) * 8
        elif pattern == "random":
            actions = tuple(random.random() for _ in range(8))
        else:
            actions = wave_activations(3 + environment.elapsed_seconds)
        environment.step(actions)
    return dict(pattern=pattern, **environment.metrics())


def main(argv=None) -> int:
    defaults = EpisodeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=("simple", "detailed"), default=defaults.body)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--duration", type=float, default=defaults.duration)
    parser.add_argument("--target-distance", type=float, default=defaults.target_distance)
    parser.add_argument("--hz", type=int, default=120, help="fixed physics ticks/second (30–1000)")
    parser.add_argument("--action-interval", type=float, default=defaults.action_interval)
    parser.add_argument("--settle-seconds", type=float, default=defaults.settle_seconds)
    parser.add_argument("--pattern", choices=("idle", "random", "scripted"), default="idle")
    parser.add_argument("--episodes", type=int, default=1, help="evaluate successive seeds, without updating weights")
    parser.add_argument("--floor-friction", type=float, default=defaults.physics.floor_friction)
    parser.add_argument("--gravity", type=float, default=defaults.physics.gravity)
    parser.add_argument("--benchmark", action="store_true", help="also measure stepping throughput and setup time")
    args = parser.parse_args(argv)
    if not 30 <= args.hz <= 1000 or args.episodes <= 0:
        parser.error("hz must be between 30 and 1000; episodes must be positive")
    try:
        config = EpisodeConfig(seed=args.seed, body=args.body, timestep=1 / args.hz, duration=args.duration,
                               target_distance=args.target_distance, action_interval=args.action_interval,
                               settle_seconds=args.settle_seconds,
                               physics=PhysicsConfig(propulsion_enabled=False, floor_friction=args.floor_friction,
                                                     gravity=args.gravity))
        reports, total_seconds, ticks, simulated = [], 0.0, 0, 0.0
        for episode in range(args.episodes):
            started = perf_counter()
            environment = LocomotionEnvironment(replace(config, seed=config.seed + episode))
            setup_seconds = perf_counter() - started
            started = perf_counter()
            report = evaluate(environment, args.pattern)
            elapsed = perf_counter() - started
            if args.benchmark:
                report.update(setup_wall_seconds=setup_seconds, step_wall_seconds=elapsed)
            reports.append(report)
            total_seconds += elapsed
            ticks += report["physics_ticks"]
            simulated += report["elapsed_seconds"]
        output = dict(episodes=reports)
        if args.benchmark:
            output["benchmark"] = dict(python=platform.python_version(), platform=platform.system(),
                                       physics_ticks_per_second=ticks / max(total_seconds, 1e-9),
                                       simulated_seconds_per_wall_second=simulated / max(total_seconds, 1e-9))
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0 if all(report["stable"] for report in reports) else 1
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
