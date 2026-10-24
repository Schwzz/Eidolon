import pytest

from eidolon.config import AppConfig, LifeConfig, WorldConfig


def test_defaults_and_uncapped_rendering() -> None:
    assert AppConfig().max_fps == 60
    assert AppConfig().simulation_hz == 60
    assert AppConfig(max_fps=0).max_fps == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"width": 0}, {"height": 200}, {"width": 640.5},
        {"max_fps": -1}, {"max_fps": True}, {"simulation_hz": 0},
        {"max_frame_seconds": 0}, {"max_frame_seconds": float("nan")},
        {"max_frame_seconds": float("inf")}, {"max_updates_per_frame": 0},
        {"seed": 1.5}, {"seed": True},
    ],
)
def test_reject_invalid_settings(settings) -> None:
    with pytest.raises(ValueError):
        AppConfig(**settings)


@pytest.mark.parametrize("settings", [
    {"width": 0}, {"height": float("inf")}, {"initial_food": -1},
    {"initial_food": 65}, {"food_capacity": True}, {"food_radius": 0},
    {"food_energy": 0}, {"regeneration_interval": -1},
])
def test_invalid_world_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        WorldConfig(**settings)


@pytest.mark.parametrize("settings", [
    {"base_radius": 0}, {"max_scale": 0.9}, {"max_energy": 0},
    {"initial_energy": 101}, {"movement_speed": -1}, {"metabolism": -1},
    {"growth_rate": -1}, {"growth_energy_threshold": 1.1},
    {"sensing_radius": float("nan")},
])
def test_invalid_life_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        LifeConfig(**settings)
