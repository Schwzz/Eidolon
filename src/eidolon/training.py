"""Display-free locomotion episodes and rewards, without learning algorithms.

Episodes reuse Colony directly: there is no steering controller, survival clock,
food, or finite fuel tank. Actuator energy is measured and penalized separately.
"""

from dataclasses import dataclass, field, replace
from math import hypot, isclose, isfinite

from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.config import BodyConfig, MuscleConfig, PhysicsConfig


@dataclass(frozen=True, slots=True)
class RewardConfig:
    energy_cost: float = 0.02
    unproductive_cost: float = 0.01  # Per second without forward progress.
    fragment_cost: float = 0.1  # Per extra component per second.
    target_bonus: float = 5.0
    instability_cost: float = 5.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    seed: int = 17
    body: str = "simple"
    timestep: float = 1 / 120
    duration: float = 20.0
    action_interval: float = 0.1
    target_distance: float = 100.0  # Signed horizontal offset; negative means left.
    settle_seconds: float = 3.0
    world_size: tuple[float, float] = (1200.0, 800.0)
    base_radius: float = 24.0
    physics: PhysicsConfig = field(default_factory=lambda: PhysicsConfig(propulsion_enabled=False))
    muscles: MuscleConfig = field(default_factory=MuscleConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        BodyConfig(self.body)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("episode seed must be an integer")
        for name in ("timestep", "duration", "action_interval", "base_radius"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 / 1000 <= self.timestep <= 1 / 30:
            raise ValueError("timestep must be between 1/1000 and 1/30 seconds")
        if (isinstance(self.settle_seconds, bool) or not isinstance(self.settle_seconds, (int, float))
                or not isfinite(self.settle_seconds) or self.settle_seconds < 0):
            raise ValueError("settle_seconds must be finite and nonnegative")
        for name in ("duration", "action_interval", "settle_seconds"):
            ticks = getattr(self, name) / self.timestep
            if (not isclose(ticks, round(ticks), rel_tol=0, abs_tol=1e-7)
                    or (name != "settle_seconds" and round(ticks) < 1)):
                raise ValueError(f"{name} must be a whole number of fixed timesteps")
        if len(self.world_size) != 2 or any(not isfinite(v) or v <= self.base_radius * 16 for v in self.world_size):
            raise ValueError("world_size must fit the body with clearance")
        if (isinstance(self.target_distance, bool) or not isinstance(self.target_distance, (int, float))
                or not isfinite(self.target_distance) or self.target_distance == 0
                or abs(self.target_distance) >= self.world_size[0] / 2 - self.base_radius * 6):
            raise ValueError("target_distance must be nonzero, finite, and inside side-wall clearance")


def reward_for_tick(previous: float, current: float, distance: float, energy: float, dt: float,
                    config: RewardConfig, *, components: int = 1, reached: bool = False,
                    stable: bool = True) -> float:
    """Signed bounded potential differences telescope; oscillation earns no progress.

    Only a stable tick can earn progress/bonus. The environment ends on the first
    target crossing, preventing repeated bonuses or reward beyond the target.
    """
    if not stable:
        return -config.instability_cost
    if not all(isfinite(v) for v in (previous, current, distance, energy, dt)) or distance <= 0 or energy < 0 or dt <= 0:
        raise ValueError("reward measurements must be finite; distance/dt positive and energy nonnegative")
    potential = lambda value: min(distance, max(-distance, value))
    delta = potential(current) - potential(previous)
    return (delta / distance - config.energy_cost * energy
            - config.unproductive_cost * dt * (delta <= 0.001 * dt)
            - config.fragment_cost * dt * max(0, components - 1)
            + config.target_bonus * reached)


OBSERVATION_NAMES = (
    "position_x", "position_y", "velocity_x", "velocity_y", "target_dx", "ground_contact",
    "width_change", "height_change", "mean_bond_strain", "max_bond_strain", "fragmentation", "time_remaining",
    *(f"activation_{i + 1}" for i in range(8)), *(f"action_{i + 1}" for i in range(8)),
    "last_muscle_energy", "total_muscle_energy",
)


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class LocomotionEnvironment:
    ACTION_COUNT = 8
    OBSERVATION_NAMES = OBSERVATION_NAMES

    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self.config = config if config is not None else EpisodeConfig()
        self.seed = self.config.seed
        self.reset()

    def reset(self, *, seed: int | None = None) -> tuple[float, ...]:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("episode seed must be an integer")
            self.seed = seed
        settings = self.config
        self.colony = Colony(self.seed, (settings.world_size[0] / 2, settings.world_size[1] / 2),
                             settings.base_radius, replace(settings.physics, propulsion_enabled=False),
                             settings.muscles, settings.body)
        bottom = max(node.position[1] + node.radius * MEMBRANE_RADIUS for node in self.colony.nodes)
        self.colony.translate_to((settings.world_size[0] / 2,
                                  self.colony.center[1] + settings.world_size[1] - bottom - 2))
        self._identity = tuple((n.identifier, n.mass) for n in self.colony.nodes)
        for _ in range(round(settings.settle_seconds / settings.timestep)):
            self.colony.advance(settings.timestep, (0.0, 0.0), settings.world_size, active=False)
        self.colony.begin_step()
        self._start_x = self.colony.material_center[0]
        self._initial_extent = self._extent()
        self._ticks = 0
        self._duration_ticks = round(settings.duration / settings.timestep)
        self._action_ticks = round(settings.action_interval / settings.timestep)
        self._progress = self._energy = self._last_energy = self._reward = 0.0
        self._stable = self._valid_physics()
        if not self._stable:
            raise ValueError("episode did not initialize with stable physics")
        self._reached = self._done = False
        self._peak_components = len(self.colony.components)
        self._last_positions = tuple(n.position for n in self.colony.nodes)
        self._observation = self.observe()
        return self._observation

    @property
    def elapsed_seconds(self) -> float:
        return self._ticks * self.config.timestep

    @property
    def done(self) -> bool:
        return self._done

    def _extent(self) -> tuple[float, float]:
        return tuple(max(n.position[axis] + n.radius for n in self.colony.nodes)
                     - min(n.position[axis] - n.radius for n in self.colony.nodes) for axis in (0, 1))

    def _valid_physics(self) -> bool:
        colony, settings = self.colony, self.config
        if (colony.physics.propulsion_enabled or colony.grab is not None
                or tuple((n.identifier, n.mass) for n in colony.nodes) != self._identity):
            return False
        for node in colony.nodes:
            radius = node.radius * MEMBRANE_RADIUS
            if (not all(isfinite(v) for v in (*node.position, *node.velocity))
                    or hypot(*node.velocity) > colony.physics.max_speed + 1e-6
                    or any(not radius - 1e-7 <= value <= size - radius + 1e-7
                           for value, size in zip(node.position, settings.world_size))):
                return False
        # Side walls cannot be used as locomotion anchors in this floor laboratory.
        if colony.side_contacts:
            return False
        return all(hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                         colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1])
                   <= b.rest_length * colony.scale * 2.5 for b in colony.bonds)

    def observe(self) -> tuple[float, ...]:
        colony, settings = self.colony, self.config
        x, y = colony.material_center
        vx, vy = colony.material_velocity
        width, height = self._extent()
        strains = [hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                         colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1])
                   / (b.rest_length * colony.scale) - 1 for b in colony.bonds]
        energy_bound = settings.muscles.holding_cost + settings.muscles.contraction_cost * settings.muscles.contraction_rate
        clip = lambda v: min(1.0, max(-1.0, v))
        return tuple(map(clip, (
            x / settings.world_size[0] * 2 - 1, y / settings.world_size[1] * 2 - 1,
            vx / colony.physics.max_speed, vy / colony.physics.max_speed,
            (self._start_x + settings.target_distance - x) / abs(settings.target_distance),
            colony.floor_contacts / len(colony.nodes),
            width / self._initial_extent[0] - 1, height / self._initial_extent[1] - 1,
            sum(strains) / max(1, len(strains)), max(map(abs, strains), default=0),
            (len(colony.components) - 1) / (len(colony.nodes) - 1),
            max(0.0, 1 - self.elapsed_seconds / settings.duration),
            *colony.muscles.activations, *colony.muscles.targets,
            self._last_energy / max(1e-9, energy_bound * settings.action_interval),
            self._energy / max(1e-9, energy_bound * settings.duration),
        )))

    def metrics(self) -> dict:
        return dict(seed=self.seed, body=self.config.body, nodes=len(self.colony.nodes), bonds=len(self.colony.bonds),
                    elapsed_seconds=self.elapsed_seconds, physics_ticks=self._ticks,
                    distance=self._progress * (1 if self.config.target_distance > 0 else -1),
                    target_progress=self._progress, target_distance=self.config.target_distance,
                    reward=self._reward, muscle_energy=self._energy, stable=self._stable,
                    reached_target=self._reached, components=len(self.colony.components),
                    peak_components=self._peak_components, floor_contacts=self.colony.floor_contacts,
                    terminated=self._done and (self._reached or not self._stable),
                    truncated=self._done and self._stable and not self._reached,
                    propulsion_enabled=self.colony.physics.propulsion_enabled)

    def step(self, actions) -> StepResult:
        if self.done:
            raise RuntimeError("episode is finished; call reset before stepping again")
        # Validation happens before time, physics, rewards, or RNG state changes.
        self.colony.muscles.set_targets(actions)
        settings, colony = self.config, self.colony
        reward, self._last_energy = 0.0, 0.0
        for _ in range(min(self._action_ticks, self._duration_ticks - self._ticks)):
            self._stable = (self._valid_physics()
                            and tuple(n.position for n in colony.nodes) == self._last_positions)
            if self._stable:
                old_center = colony.material_center
                try:
                    colony.advance(settings.timestep, (0.0, 0.0), settings.world_size)
                    self._stable = self._valid_physics()
                except (FloatingPointError, ValueError, OverflowError):
                    self._stable = False
            self._ticks += 1
            if not self._stable:
                reward -= settings.rewards.instability_cost
                self._reward -= settings.rewards.instability_cost
                break
            center = colony.material_center
            if abs(center[0] - old_center[0]) > colony.physics.max_speed * settings.timestep + 1e-6:
                self._stable = False
                reward -= settings.rewards.instability_cost
                self._reward -= settings.rewards.instability_cost
                break
            progress = (center[0] - self._start_x) * (1 if settings.target_distance > 0 else -1)
            energy = colony.muscle_energy_used
            if not isfinite(energy) or energy < 0 or not isfinite(self._energy + energy):
                self._stable = False
                reward -= settings.rewards.instability_cost
                self._reward -= settings.rewards.instability_cost
                break
            self._energy += energy
            self._last_energy += energy
            self._reached = progress >= abs(settings.target_distance) and len(colony.components) == 1
            tick_reward = reward_for_tick(self._progress, progress, abs(settings.target_distance), energy,
                                          settings.timestep, settings.rewards, components=len(colony.components),
                                          reached=self._reached)
            reward += tick_reward
            self._reward += tick_reward
            self._progress = progress
            self._peak_components = max(self._peak_components, len(colony.components))
            self._last_positions = tuple(n.position for n in colony.nodes)
            if self._reached:
                break
        terminated = self._reached or not self._stable
        truncated = self._ticks >= self._duration_ticks and not terminated
        self._done = terminated or truncated
        # Numerical faults return the last finite observation, not a NaN vector.
        if self._stable:
            self._observation = self.observe()
        return StepResult(self._observation, reward, terminated, truncated, self.metrics())
