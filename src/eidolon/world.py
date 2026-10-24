"""Bounded world and a capped, reproducible food supply. No display imports."""

from dataclasses import dataclass, field, replace
from math import ceil, exp, hypot, isfinite
from random import Random

from eidolon.config import PhysicsConfig, WorldConfig


@dataclass(frozen=True, slots=True)
class Food:
    identifier: int
    position: tuple[float, float]
    energy: float
    radius: float
    mass: float = 1.0
    velocity: tuple[float, float] = (0.0, 0.0)
    previous_position: tuple[float, float] | None = field(default=None, compare=False)

    def sample(self, alpha: float = 1.0) -> tuple[float, float]:
        previous = self.previous_position if self.previous_position is not None else self.position
        alpha = min(1.0, max(0.0, alpha))
        return tuple(a + (b - a) * alpha for a, b in zip(previous, self.position))


class World:
    MAX_STEP = 1 / 120

    def __init__(self, config: WorldConfig, seed: int, spawn_margin: float,
                 physics: PhysicsConfig | None = None) -> None:
        self.config = config
        self.physics = physics if physics is not None else PhysicsConfig()
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
            self.config.food_mass,
        )
        self._next_id += 1
        self.foods.append(food)
        return food

    def advance(self, dt: float) -> None:
        if not isfinite(dt) or dt < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        if dt == 0:
            return
        if not self.foods and self.config.regeneration_interval == 0:
            self.elapsed_seconds += dt
            return
        count = max(1, ceil(dt / self.MAX_STEP - 1e-9))
        for _ in range(count):
            self._step(dt / count)

    def begin_step(self) -> None:
        self.foods = [replace(food, previous_position=food.position)
                      if food.previous_position != food.position else food for food in self.foods]

    def _step(self, dt: float) -> None:
        settings = self.physics
        damping = exp(-settings.linear_damping * dt)
        for index, food in enumerate(self.foods):
            if (not all(isfinite(value) for value in (*food.position, *food.velocity, food.mass, food.radius))
                    or food.mass <= 0 or food.radius <= 0
                    or food.radius * 2 >= min(self.config.width, self.config.height)):
                raise ValueError("physical food needs finite state, positive mass, and a radius that fits the world")
            floor = self.config.height - food.radius
            if (food.position[1] == floor and food.velocity == (0.0, 0.0)
                    and food.radius <= food.position[0] <= self.config.width - food.radius):
                continue  # Sleeping floor resources require no new allocation or integration.
            vx, vy = food.velocity[0] * damping, food.velocity[1] * damping + settings.gravity * dt
            speed = hypot(vx, vy)
            if speed > settings.max_speed:
                vx, vy = vx * settings.max_speed / speed, vy * settings.max_speed / speed
            x, y = food.position[0] + vx * dt, food.position[1] + vy * dt
            if x < food.radius or x > self.config.width - food.radius:
                x, vx = min(self.config.width - food.radius, max(food.radius, x)), 0.0
            if y >= floor:
                x_friction = settings.floor_friction * max(settings.gravity * dt, vy)
                vx = (1 if vx > 0 else -1) * max(0.0, abs(vx) - x_friction)
                y, vy = floor, 0.0
            elif y < food.radius:
                y, vy = food.radius, max(0.0, vy)
            self.foods[index] = replace(food, position=(x, y), velocity=(vx, vy))
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
