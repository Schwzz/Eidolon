"""Stateless neural decisions over normalized world senses, without graphics/physics."""

from math import cos, hypot, isfinite, pi, sin, tau

import numpy as np

from eidolon.behavior import Decision, Senses
from eidolon.neural_network import FeedForwardNetwork


INPUT_NAMES = ("energy", "food_dx", "food_dy", "food_distance", "heading_cos", "heading_sin",
               "speed", "wall_left", "wall_right", "wall_top", "wall_bottom", "food_detected")


def encode_senses(senses: Senses) -> np.ndarray:
    x, y = senses.position
    width, height = senses.world_size
    values = (x, y, width, height, senses.direction, senses.energy_fraction,
              senses.speed, senses.speed_reference, senses.sensing_radius, senses.clearance)
    if not all(isfinite(value) for value in values):
        raise ValueError("neural senses must contain finite numbers")
    if (min(width, height, senses.sensing_radius) <= 0 or senses.speed < 0
            or senses.speed_reference < 0 or senses.clearance < 0):
        raise ValueError("neural world dimensions/range must be positive; speed and clearance nonnegative")

    def unit(value):
        return min(1.0, max(0.0, value))

    food_x, food_y, distance, detected = 0.0, 0.0, 1.0, 0.0
    if senses.food is not None:
        if not all(isfinite(value) for value in senses.food.position):
            raise ValueError("neural food coordinates must be finite")
        dx, dy = senses.food.position[0] - x, senses.food.position[1] - y
        length = hypot(dx, dy)
        if not isfinite(length):
            raise ValueError("neural food distance overflowed")
        food_x, food_y = (dx / length, dy / length) if length else (0.0, 0.0)
        distance, detected = unit(length / senses.sensing_radius), 1.0
    boundaries = (x, width - x, y, height - y)
    return np.array((unit(senses.energy_fraction), food_x, food_y, distance,
                     cos(senses.direction), sin(senses.direction),
                     unit(senses.speed / senses.speed_reference) if senses.speed_reference else 0.0,
                     *(unit(1 - (gap - senses.clearance) / senses.sensing_radius) for gap in boundaries),
                     detected), dtype=np.float64)


class NeuralController:
    name = "neural"

    def __init__(self, network: FeedForwardNetwork) -> None:
        if not isinstance(network, FeedForwardNetwork):
            raise ValueError("NeuralController needs a FeedForwardNetwork")
        if network.layers[0] != len(INPUT_NAMES) or network.layers[-1] != 2:
            raise ValueError(f"neural controller requires {len(INPUT_NAMES)} inputs and 2 outputs")
        self.network = network

    @classmethod
    def from_seed(cls, seed: int, hidden_layers=(16,)) -> "NeuralController":
        return cls(FeedForwardNetwork((len(INPUT_NAMES), *hidden_layers, 2), seed))

    def decide(self, senses: Senses) -> Decision:
        steering, effort = self.network.forward(encode_senses(senses))
        return Decision((senses.direction + float(steering) * pi) % tau, (float(effort) + 1) * 0.5)
