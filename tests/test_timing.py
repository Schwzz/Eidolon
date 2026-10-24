import pytest

from eidolon.config import AppConfig
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock


@pytest.mark.parametrize("render_fps", [30, 60, 144])
def test_simulation_rate_is_independent_of_render_rate(render_fps) -> None:
    clock = FixedStepClock(AppConfig())
    simulation = Simulation()
    steps = []

    def update(dt):
        steps.append(dt)
        simulation.update(dt)

    for _ in range(render_fps * 2):
        clock.advance(1 / render_fps, update)
    assert simulation.steps == 120
    assert simulation.elapsed_seconds == pytest.approx(2)
    assert steps == [1 / 60] * 120


def test_retains_fractional_time_between_frames() -> None:
    clock = FixedStepClock(AppConfig())
    steps = []
    assert clock.advance(1 / 120, steps.append) == 0
    assert clock.advance(1 / 120, steps.append) == 1
    assert steps == [1 / 60]


def test_catch_up_is_bounded_and_excess_backlog_is_dropped() -> None:
    clock = FixedStepClock(AppConfig())
    steps = []
    assert clock.advance(10, steps.append) == 8
    assert len(steps) == 8
    assert clock.advance(0, steps.append) == 0
    assert clock.advance(1 / 60, steps.append) == 1


def test_frame_time_is_clamped_before_catch_up() -> None:
    clock = FixedStepClock(AppConfig(max_updates_per_frame=100))
    assert clock.advance(10, lambda dt: None) == 15


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_rejects_invalid_wall_time(elapsed) -> None:
    with pytest.raises(ValueError):
        FixedStepClock(AppConfig()).advance(elapsed, lambda dt: None)
