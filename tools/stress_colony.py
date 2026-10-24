"""Repeat deterministic tears and recovery without a display; raises on lost/stuck tissue."""

import argparse
import json
from math import cos, hypot, isfinite, sin
from time import perf_counter

from eidolon.colony import Colony, MEMBRANE_RADIUS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.cycles <= 0 or not isfinite(args.scale) or not 1 <= args.scale <= 1.6:
        parser.error("--cycles must be positive and --scale must be between 1 and 1.6")
    colony = Colony(args.seed, (600, 400), 24)
    colony.set_growth(args.scale)
    identities = tuple(id(node) for node in colony.nodes)
    initial_bonds = len(colony.bonds)
    initial_mass = sum(node.mass for node in colony.nodes)
    steps, peak_components, recovered, max_speed = 0, 1, 0, 0.0
    started = perf_counter()
    for cycle in range(args.cycles):
        axis = cycle % 2
        direction = 1 if colony.center[axis] < (600, 400)[axis] else -1
        node = max(colony.nodes, key=lambda item: item.position[axis] * direction)
        origin = node.position
        colony.begin_grab(origin)
        # 2.5 seconds pulling, then 18 seconds with normal locomotion/recovery.
        for frame in range(2460):
            if frame < 300:
                target = list(origin)
                target[axis] += direction * (frame + 1) / 120 * 160
                colony.move_grab(tuple(target))
            elif frame == 300:
                colony.release_grab()
            t = steps / 120
            colony.advance(1 / 120, (25 * cos(t * 0.1), 25 * sin(t * 0.1)), (1200, 800))
            steps += 1
            peak_components = max(peak_components, len(colony.components))
            assert len(colony.bonds) == initial_bonds - colony.broken_bonds + colony.reconnected_bonds
            for node in colony.nodes:
                assert all(isfinite(value) for value in (*node.position, *node.velocity))
                radius = node.radius * MEMBRANE_RADIUS
                assert radius <= node.position[0] <= 1200 - radius
                assert radius <= node.position[1] <= 800 - radius
                speed = hypot(*node.velocity)
                max_speed = max(max_speed, speed)
                assert speed <= 220 * args.scale**0.5 + 1e-6
                assert len(node.connections) <= colony.MAX_NEIGHBORS
            if colony.grab is not None:
                assert colony.grab.applied_force <= 14000 * args.scale + 1e-6
        assert len(colony.components) == 1, f"Unrecovered fragment after cycle {cycle + 1}: {colony.components}"
        assert tuple(id(node) for node in colony.nodes) == identities
        assert sum(node.mass for node in colony.nodes) == initial_mass
        recovered += 1
    assert peak_components > 1 and colony.broken_bonds > 0 and colony.reconnected_bonds > 0
    print(json.dumps(dict(seed=args.seed, scale=args.scale, nodes=len(colony.nodes),
                          bonds=len(colony.bonds), simulation_seconds=steps / 120,
                          cycles_recovered=recovered, peak_components=peak_components,
                          broken_bonds=colony.broken_bonds, reconnected_bonds=colony.reconnected_bonds,
                          max_speed=round(max_speed, 3), wall_seconds=round(perf_counter() - started, 3)), indent=2))


if __name__ == "__main__":
    main()
