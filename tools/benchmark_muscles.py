"""Headless physical-displacement measurement; no steering, training, or rendering."""

import argparse
import json
from math import hypot, isfinite
import platform
from time import perf_counter

from eidolon.colony import MEMBRANE_RADIUS
from eidolon.config import LifeConfig, PhysicsConfig, WorldConfig
from eidolon.muscles import wave_activations
from eidolon.simulation import Simulation


def episode(seed: int, friction: float, cycles: int, driven: bool, body: str = "detailed") -> dict:
    simulation = Simulation(seed, WorldConfig(initial_food=0, regeneration_interval=0),
                            LifeConfig(metabolism=0, movement_cost=0),
                            physics_config=PhysicsConfig(propulsion_enabled=False, floor_friction=friction),
                            muscle_test="manual", body=body)
    colony = simulation.colony
    identities = tuple(id(node) for node in colony.nodes)
    original_mass = sum(node.mass for node in colony.nodes)
    started = perf_counter()
    max_speed = 0.0
    settle_frames, active_frames, recovery_frames = 180, cycles * 240, 240
    settled_x = colony.material_center[0]
    for frame in range(settle_frames + active_frames + recovery_frames):
        if driven and frame < settle_frames + active_frames:
            simulation.set_muscle_activations(wave_activations(frame / 60))
        else:
            simulation.relax_muscles()
        simulation.update(1 / 60)
        if frame + 1 == settle_frames:
            settled_x = colony.material_center[0]
        assert simulation.life.alive and len(colony.components) == 1
        for node in colony.nodes:
            assert all(isfinite(v) for v in (*node.position, *node.velocity))
            speed = hypot(*node.velocity)
            max_speed = max(max_speed, speed)
            assert speed <= colony.physics.max_speed + 1e-6
            radius = node.radius * MEMBRANE_RADIUS
            assert radius <= node.position[0] <= 1200 - radius
            assert radius <= node.position[1] <= 800 - radius
    assert colony.muscles.activations == (0,) * 8
    assert tuple(id(node) for node in colony.nodes) == identities
    assert sum(node.mass for node in colony.nodes) == original_mass
    assert colony.broken_bonds == 0
    return dict(displacement_after_settling=round(colony.material_center[0] - settled_x, 9),
                muscle_energy=round(simulation.life.muscle_energy_spent, 6),
                max_speed=round(max_speed, 6), nodes=len(colony.nodes), bonds=len(colony.bonds),
                simulation_seconds=round(simulation.elapsed_seconds, 3),
                wall_seconds=round(perf_counter() - started, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--body", choices=("simple", "detailed"), default="detailed")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--floor-friction", type=float, default=0.55)
    args = parser.parse_args()
    if args.cycles <= 0 or not isfinite(args.floor_friction) or args.floor_friction < 0:
        parser.error("cycles must be positive and floor friction finite and nonnegative")
    active = episode(args.seed, args.floor_friction, args.cycles, True, args.body)
    passive = episode(args.seed, args.floor_friction, args.cycles, False, args.body)
    print(json.dumps(dict(python=platform.python_version(), platform=platform.system(), seed=args.seed, body=args.body,
                          floor_friction=args.floor_friction, propulsion_enabled=False,
                          settle_seconds=3, wave_seconds=args.cycles * 4, relaxation_seconds=4,
                          active=active, passive=passive,
                          baseline_adjusted_dx=round(active["displacement_after_settling"]
                                                     - passive["displacement_after_settling"], 9)), indent=2))


if __name__ == "__main__":
    main()
