"""Shared muscle observations for survival simulation and timed experiments."""

from math import hypot


def body_extent(colony):
    return tuple(max(n.position[axis] + n.radius for n in colony.nodes)
                 - min(n.position[axis] - n.radius for n in colony.nodes) for axis in (0, 1))


def muscle_observations(colony, *, world_size, target_x, target_distance, initial_extent,
                        time_remaining, last_energy, total_energy, action_interval, energy_horizon):
    x, y = colony.material_center
    vx, vy = colony.material_velocity
    width, height = body_extent(colony)
    strains = [hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                     colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1])
               / (b.rest_length * colony.scale) - 1 for b in colony.bonds]
    settings = colony.muscles.config
    energy_bound = settings.holding_cost + settings.contraction_cost * settings.contraction_rate
    clip = lambda v: min(1.0, max(-1.0, v))
    return tuple(map(clip, (
        x / world_size[0] * 2 - 1, y / world_size[1] * 2 - 1,
        vx / colony.physics.max_speed, vy / colony.physics.max_speed,
        (target_x - x) / abs(target_distance),
        colony.floor_contacts / len(colony.nodes),
        width / initial_extent[0] - 1, height / initial_extent[1] - 1,
        sum(strains) / max(1, len(strains)), max(map(abs, strains), default=0),
        (len(colony.components) - 1) / (len(colony.nodes) - 1), time_remaining,
        *colony.muscles.activations, *colony.muscles.targets,
        last_energy / max(1e-9, energy_bound * action_interval),
        total_energy / max(1e-9, energy_bound * energy_horizon),
    )))
