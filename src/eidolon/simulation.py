"""Fixed-step coordination of a world, biology, rules, and presentation clocks."""

from math import atan2, ceil, cos, hypot, isfinite, sin, tau

from eidolon.behavior import Controller, Senses, SurvivalRules
from eidolon.biology import LifeState, SPRITE_EXTENT
from eidolon.config import LifeConfig, WorldConfig
from eidolon.colony import Colony
from eidolon.organism import Symbiote
from eidolon.world import World


class Simulation:
    MAX_STEP = 1 / 60

    def __init__(
        self, seed: int = 17, world_config: WorldConfig | None = None,
        life_config: LifeConfig | None = None, controller: Controller | None = None,
    ) -> None:
        self.seed = seed
        self.world_config = world_config if world_config is not None else WorldConfig()
        self.life_config = life_config if life_config is not None else LifeConfig()
        self.controller = controller if controller is not None else SurvivalRules(seed)
        self.organism = Symbiote(seed)  # Appearance/animation; no biological data.
        self._paused = False
        self.animation_speed = 1.0  # Kept as a compatibility name; scales all systems.
        self.reset()

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value
        if value:
            # Freeze interpolation once, at the pause event, rather than letting
            # paused updates mutate world state or consume random numbers.
            self.life.begin_step()
            self.colony.release_grab()
            self.colony.begin_step()
            self.organism.update(0.0)

    @property
    def speed(self) -> float:
        return self.animation_speed

    def reset(self) -> None:
        """Recreate the whole seeded world, preserving pause and speed controls."""
        settings = self.life_config
        margin = settings.base_radius * settings.max_scale * SPRITE_EXTENT + self.world_config.food_radius
        self.world = World(self.world_config, self.seed, margin)
        self.life = LifeState(
            (self.world_config.width / 2, self.world_config.height / 2),
            (self.seed % 65536) * 0.61803398875 % tau,
            settings.base_radius, settings.initial_energy, settings.max_energy,
        )
        self.colony = Colony(self.seed, self.life.position, settings.base_radius)
        self.life.position = self.colony.center
        self.life.begin_step()
        self.steps = 0
        self.elapsed_seconds = 0.0
        self.organism.reset()

    def set_speed(self, speed: float) -> None:
        if not isfinite(speed) or not 0.25 <= speed <= 2.0:
            raise ValueError("simulation speed must be between 0.25 and 2.0")
        self.animation_speed = speed

    def set_animation_speed(self, speed: float) -> None:
        """Compatibility entry point; speed now scales the complete simulation."""
        self.set_speed(speed)

    def begin_grab(self, position: tuple[float, float], padding: float = 0.0) -> int | None:
        if self.paused or not self.life.alive:
            self.release_grab()
            return None
        if not all(isfinite(value) for value in position):
            raise ValueError("pointer coordinates must be finite")
        if not (0 <= position[0] <= self.world_config.width and 0 <= position[1] <= self.world_config.height):
            self.release_grab()
            return None
        return self.colony.begin_grab(position, padding)

    def move_grab(self, position: tuple[float, float]) -> None:
        if not self.life.alive:
            self.release_grab()
            return
        if self.paused:
            return
        if not all(isfinite(value) for value in position):
            raise ValueError("pointer coordinates must be finite")
        self.colony.move_grab((min(self.world_config.width, max(0.0, position[0])),
                               min(self.world_config.height, max(0.0, position[1]))))

    def release_grab(self) -> None:
        self.colony.release_grab()

    def update(self, dt: float) -> None:
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.paused:
            return
        self.steps += 1
        elapsed = dt * self.speed
        was_alive = self.life.alive
        if was_alive:
            self.life.begin_step()
            self.colony.begin_step()
        age_before = self.life.age_seconds
        # Keep movement/feeding stable for larger caller timesteps and at 2x.
        count = max(1, ceil(elapsed / self.MAX_STEP - 1e-9))
        for _ in range(count):
            self._integrate(elapsed / count)
        self.elapsed_seconds += elapsed
        if was_alive:
            self.organism.update(self.life.age_seconds - age_before)

    def _eat(self) -> None:
        state, settings = self.life, self.life_config
        food = self.world.nearest_food(
            state.position, settings.feeding_radius * state.body_scale + self.world_config.food_radius,
        )
        if food is not None:
            consumed = self.world.consume(food.identifier)
            if consumed is not None:
                state.energy = min(state.max_energy, state.energy + consumed.energy)
                state.nutrition += consumed.energy
                state.foods_eaten += 1

    def _integrate(self, dt: float) -> None:
        self.world.advance(dt)
        state, settings = self.life, self.life_config
        if not state.alive:
            self.colony.freeze()
            return
        if state.energy <= 1e-10:
            state.die()
            self.colony.freeze()
            return
        self._eat()
        sensed_food = self.world.nearest_food(state.position, settings.sensing_radius)
        decision = self.controller.decide(Senses(
            state.position, state.direction, state.age_seconds,
            state.energy / state.max_energy, state.collision_radius,
            (self.world_config.width, self.world_config.height), sensed_food,
        ))
        if not isfinite(decision.direction) or not isfinite(decision.speed_fraction):
            raise ValueError("controller decisions must be finite")
        turn = atan2(sin(decision.direction - state.direction), cos(decision.direction - state.direction))
        state.direction = (state.direction + min(settings.turn_rate * dt, max(-settings.turn_rate * dt, turn))) % tau
        speed = settings.movement_speed / state.body_scale**0.5 * min(1.0, max(0.0, decision.speed_fraction))
        displacement = self.colony.advance(dt, (cos(state.direction) * speed, sin(state.direction) * speed),
                                           (self.world_config.width, self.world_config.height))
        state.position = self.colony.center
        state.velocity = tuple(value / dt for value in displacement)
        # Membership changes can move a centroid without moving any material.
        # Charge actual primary-component motion, not that bookkeeping jump.
        traveled = hypot(*displacement)
        if self.colony.wall_contacts and speed > 0 and traveled < speed * dt * 0.2:
            state.direction = atan2(self.world_config.height / 2 - state.position[1],
                                    self.world_config.width / 2 - state.position[0]) % tau
        state.age_seconds += dt
        state.energy = max(0.0, state.energy - settings.metabolism * state.body_scale**2 * dt
                           - settings.movement_cost * traveled)
        if state.energy <= 1e-10:
            state.die()
            self.colony.freeze()
            return
        self._eat()
        target_scale = min(settings.max_scale, 1 + state.nutrition / settings.nutrition_per_scale)
        # Growth waits for enough space instead of pushing or teleporting the
        # organism through a wall; food is spawned inside this reachable region.
        x, y = state.position
        available_radius = min(x, y, self.world_config.width - x, self.world_config.height - y)
        target_scale = min(target_scale, available_radius / (settings.base_radius * SPRITE_EXTENT))
        reserve = state.max_energy * settings.growth_energy_threshold
        if state.energy > reserve and state.body_scale < target_scale:
            growth = min(settings.growth_rate * dt, target_scale - state.body_scale)
            if settings.growth_energy_cost > 0:
                growth = min(growth, (state.energy - reserve) / settings.growth_energy_cost)
            state.body_scale += growth
            self.colony.set_growth(state.body_scale)
            state.energy -= growth * settings.growth_energy_cost
            if state.energy <= 1e-10:
                state.die()
                self.colony.freeze()
