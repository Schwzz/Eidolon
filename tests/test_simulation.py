import pytest

from eidolon.simulation import Simulation


def test_simulation_advances_without_rendering() -> None:
    simulation = Simulation()
    assert simulation.steps == 0
    assert simulation.elapsed_seconds == 0
    for _ in range(60):
        simulation.update(1 / 60)
    assert simulation.steps == 60
    assert simulation.elapsed_seconds == pytest.approx(1)
    assert simulation.organism.age_seconds == pytest.approx(1)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_timestep_does_not_change_state(dt) -> None:
    simulation = Simulation()
    with pytest.raises(ValueError):
        simulation.update(dt)
    assert simulation.steps == 0
    assert simulation.elapsed_seconds == 0
    assert simulation.organism.age_seconds == 0


def test_pause_freezes_organism_and_simulation_time() -> None:
    simulation = Simulation()
    simulation.update(1 / 60)
    simulation.paused = True
    simulation.update(1 / 60)
    frozen = simulation.organism.pose()
    for _ in range(120):
        simulation.update(1 / 60)
    assert simulation.organism.pose(0) == frozen
    assert simulation.organism.pose(0.75) == frozen
    assert simulation.steps == 1
    assert simulation.elapsed_seconds == pytest.approx(1 / 60)
    simulation.paused = False
    simulation.update(1 / 60)
    assert simulation.organism.age_seconds == pytest.approx(2 / 60)


def test_speed_scales_world_time_and_animation_with_fixed_step_accounting() -> None:
    simulation = Simulation()
    simulation.set_animation_speed(0.5)
    for _ in range(60):
        simulation.update(1 / 60)
    assert simulation.elapsed_seconds == pytest.approx(0.5)
    assert simulation.world.elapsed_seconds == pytest.approx(0.5)
    assert simulation.life.age_seconds == pytest.approx(0.5)
    assert simulation.steps == 60
    assert simulation.organism.age_seconds == pytest.approx(0.5)


def test_reset_repeats_appearance_and_initial_pose() -> None:
    simulation = Simulation(seed=711)
    initial = simulation.organism.pose()
    appearance = simulation.organism.appearance
    simulation.update(10)
    simulation.set_animation_speed(2)
    simulation.paused = True
    simulation.reset()
    assert simulation.organism.appearance == appearance
    assert simulation.organism.pose(0) == simulation.organism.pose(1) == initial
    assert simulation.steps == 0
    assert simulation.elapsed_seconds == 0
    assert simulation.animation_speed == 2
    assert simulation.paused


@pytest.mark.parametrize("speed", [0, -1, 0.24, 2.01, float("nan"), float("inf")])
def test_invalid_animation_speed_is_rejected(speed) -> None:
    simulation = Simulation()
    with pytest.raises(ValueError):
        simulation.set_animation_speed(speed)
    assert simulation.animation_speed == 1
