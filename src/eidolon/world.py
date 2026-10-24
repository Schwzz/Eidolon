"""Bounded world and a capped, reproducible food supply. No display imports."""

from dataclasses import dataclass
from math import isfinite
from random import Random

from eidolon.config import WorldConfig


@dataclass(frozen=True, slots=True)
class Food:
    identifier: int
    position: tuple[float, float]
    energy: float
    radius: float


class World:
    def __init__(self, config: WorldConfig, seed: int, spawn_margin: float) -> None:
        self.config = config
        self.spawn_margin = max(spawn_margin, config.food_radius)
        if not isfinite(self.spawn_margin) or self.spawn_margin * 2 >= min(config.width, config.height):
            raise ValueError("world must have room for the fully grown organism and food")
        self._random = Random(seed ^ 0xE1D010)
        self.foods: list[Food] = []
        self.elapsed_seconds = 0.0
        self.regeneration_elapsed = 0.0
        self._next_id = 0
        for _ in range(config.initial_food):
            self.spawn_food()

    def spawn_food(self) -> Food | None:
        if len(self.foods) >= self.config.food_capacity:
            return None
        margin = self.spawn_margin
        food = Food(
            self._next_id,
            (self._random.uniform(margin, self.config.width - margin),
             self._random.uniform(margin, self.config.height - margin)),
            self.config.food_energy,
            self.config.food_radius,
        )
        self._next_id += 1
        self.foods.append(food)
        return food

    def advance(self, dt: float) -> None:
        if not isfinite(dt) or dt < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        self.elapsed_seconds += dt
        interval = self.config.regeneration_interval
        if interval == 0:
            return
        self.regeneration_elapsed += dt
        due = int((self.regeneration_elapsed + interval * 1e-9) / interval)
        self.regeneration_elapsed = max(0.0, self.regeneration_elapsed - due * interval)
        # Full worlds discard missed spawns rather than banking a future burst.
        for _ in range(min(due, self.config.food_capacity - len(self.foods))):
            self.spawn_food()

    def nearest_food(self, position: tuple[float, float], radius: float) -> Food | None:
        best, best_distance = None, radius * radius
        for food in self.foods:
            distance = sum((a - b) ** 2 for a, b in zip(food.position, position))
            if distance <= radius * radius and (
                best is None or (distance, food.identifier) < (best_distance, best.identifier)
            ):
                best, best_distance = food, distance
        return best

    def consume(self, identifier: int) -> Food | None:
        """Remove and return a resource once; repeated consumption returns None."""
        for index, food in enumerate(self.foods):
            if food.identifier == identifier:
                return self.foods.pop(index)
        return None
