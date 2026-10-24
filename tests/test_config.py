import pytest

from eidolon.config import AppConfig


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
    ],
)
def test_reject_invalid_settings(settings) -> None:
    with pytest.raises(ValueError):
        AppConfig(**settings)
