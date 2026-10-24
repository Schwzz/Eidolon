"""Validated application settings; no window or simulation side effects."""

from dataclasses import dataclass, field
from math import isfinite


MIN_WINDOW_WIDTH = 480
MIN_WINDOW_HEIGHT = 320


@dataclass(frozen=True, slots=True)
class WorldConfig:
    width: float = 1200.0
    height: float = 800.0
    initial_food: int = 32
    food_capacity: int = 64
    food_energy: float = 18.0
    food_radius: float = 5.0
    regeneration_interval: float = 3.0  # Zero disables regeneration.

    def __post_init__(self) -> None:
        for name in ("width", "height", "food_energy", "food_radius"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("initial_food", "food_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.initial_food > self.food_capacity:
            raise ValueError("initial_food cannot exceed food_capacity")
        if not isfinite(self.regeneration_interval) or self.regeneration_interval < 0:
            raise ValueError("regeneration_interval must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class LifeConfig:
    base_radius: float = 24.0  # Nominal colony size/spacing in world units.
    max_scale: float = 1.6
    initial_energy: float = 70.0
    max_energy: float = 100.0
    movement_speed: float = 55.0
    turn_rate: float = 2.8  # Radians per simulation second.
    metabolism: float = 1.0
    movement_cost: float = 0.025  # Energy per world unit actually traveled.
    sensing_radius: float = 250.0
    feeding_radius: float = 34.0
    growth_rate: float = 0.01  # Scale per simulation second.
    nutrition_per_scale: float = 150.0
    growth_energy_threshold: float = 0.45
    growth_energy_cost: float = 12.0

    def __post_init__(self) -> None:
        for name in ("base_radius", "max_energy", "sensing_radius", "feeding_radius", "nutrition_per_scale"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("movement_speed", "turn_rate", "metabolism", "movement_cost", "growth_rate", "growth_energy_cost"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isfinite(self.max_scale) or self.max_scale < 1:
            raise ValueError("max_scale must be finite and at least 1")
        if not isfinite(self.initial_energy) or not 0 <= self.initial_energy <= self.max_energy:
            raise ValueError("initial_energy must be between 0 and max_energy")
        if not isfinite(self.growth_energy_threshold) or not 0 <= self.growth_energy_threshold <= 1:
            raise ValueError("growth_energy_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AppConfig:
    width: int = 1280
    height: int = 720
    max_fps: int = 60  # Zero means uncapped rendering.
    simulation_hz: int = 60
    max_frame_seconds: float = 0.25
    max_updates_per_frame: int = 8
    seed: int = 17
    world: WorldConfig = field(default_factory=WorldConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    controller: str = "rules"

    def __post_init__(self) -> None:
        if self.controller not in ("rules", "neural"):
            raise ValueError("controller must be 'rules' or 'neural'")
        for name in ("width", "height", "max_fps", "simulation_hz", "max_updates_per_frame", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.width < MIN_WINDOW_WIDTH or self.height < MIN_WINDOW_HEIGHT:
            raise ValueError(
                f"window size must be at least {MIN_WINDOW_WIDTH} x {MIN_WINDOW_HEIGHT}"
            )
        if self.max_fps < 0:
            raise ValueError("max_fps must be nonnegative (0 means uncapped)")
        if self.simulation_hz <= 0 or self.max_updates_per_frame <= 0:
            raise ValueError("simulation_hz and max_updates_per_frame must be positive")
        if not isfinite(self.max_frame_seconds) or self.max_frame_seconds <= 0:
            raise ValueError("max_frame_seconds must be finite and positive")
