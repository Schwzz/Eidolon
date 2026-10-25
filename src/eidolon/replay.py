"""Window-facing adapter over the exact headless locomotion episode integrator."""

from dataclasses import replace
from math import isfinite

from eidolon.biology import LifeState
from eidolon.config import WorldConfig
from eidolon.organism import Symbiote
from eidolon.training import LocomotionEnvironment
from eidolon.world import World


class MuscleReplay:
    is_replay = True
    muscle_test = None
    controller_name = "neural muscles"

    def __init__(self, controller, *, seed=None, body=None):
        self.controller = controller
        self.episode = replace(controller.episode, seed=controller.episode.seed if seed is None else seed,
                               body=controller.episode.body if body is None else body)
        self.seed = self.episode.seed
        self.world_config = WorldConfig(width=self.episode.world_size[0], height=self.episode.world_size[1],
                                        initial_food=0, food_capacity=0, regeneration_interval=0)
        self.world = World(self.world_config, self.seed, 0, self.episode.physics)
        self.organism = Symbiote(self.seed)
        self.paused = False
        self.animation_speed = 1.0
        self.reset()

    @property
    def colony(self):
        return self.environment.colony

    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, value):
        self._paused = value
        if value:
            self.colony.release_grab()
            self.colony.begin_step()
            self.life.begin_step()

    @property
    def speed(self):
        return self.animation_speed

    @property
    def elapsed_seconds(self):
        return self.environment.elapsed_seconds

    @property
    def muscle_displacement(self):
        return self.environment.metrics()["distance"]

    def reset(self):
        self.environment = LocomotionEnvironment(self.episode, allow_grab=True)
        self.life = LifeState(self.colony.center, 0, self.episode.base_radius, 100, 100)
        self.organism.reset()
        self.controller.clear_activity()
        self._accumulator = 0.0
        self._actions = (0.0,) * 8
        self.steps = 0
        self.intervened = self.episode.body != self.controller.episode.body

    def set_speed(self, speed):
        if not isfinite(speed) or not 0.25 <= speed <= 2:
            raise ValueError("simulation speed must be between 0.25 and 2.0")
        self.animation_speed = speed

    def update(self, dt):
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.paused or self.environment.done:
            return
        self.steps += 1
        self.life.begin_step()
        self.colony.begin_step()
        self._accumulator += dt * self.speed
        timestep = self.episode.timestep
        while self._accumulator + 1e-10 >= timestep and not self.environment.done:
            new_action = self.environment._ticks % self.environment._action_ticks == 0
            if new_action:
                self._actions = self.controller.act(self.environment.observe())
            self.environment.step(self._actions, ticks=1, new_action=new_action)
            self._accumulator = max(0.0, self._accumulator - timestep)
        self.life.position = self.colony.center
        self.life.velocity = self.colony.material_velocity
        self.life.age_seconds = self.elapsed_seconds
        self.life.muscle_energy_spent = self.environment.metrics()["muscle_energy"]
        self.organism.update(self.elapsed_seconds - self.organism.age_seconds)

    def begin_grab(self, position, padding=0.0):
        if self.paused or self.environment.done:
            return None
        if not all(isfinite(v) for v in position):
            raise ValueError("pointer coordinates must be finite")
        if not all(0 <= v <= limit for v, limit in zip(position, self.episode.world_size)):
            return None
        identifier = self.colony.begin_grab(position, padding)
        self.intervened |= identifier is not None
        return identifier

    def move_grab(self, position):
        if not all(isfinite(v) for v in position):
            raise ValueError("pointer coordinates must be finite")
        if not self.paused and not self.environment.done:
            self.colony.move_grab(tuple(min(limit, max(0.0, v)) for v, limit in zip(position, self.episode.world_size)))

    def release_grab(self):
        self.colony.release_grab()

    def relax_muscles(self):
        # Resize/focus events clear manual controls, not held neural commands.
        pass
