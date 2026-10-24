import pytest

from eidolon.simulation import Simulation


def test_empty_simulation_advances_without_rendering() -> None:
    simulation = Simulation()
    assert simulation.steps == 0
    assert simulation.elapsed_seconds == 0
    for _ in range(60):
        simulation.update(1 / 60)
    assert simulation.steps == 60
    assert simulation.elapsed_seconds == pytest.approx(1)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_timestep_does_not_change_state(dt) -> None:
    simulation = Simulation()
    with pytest.raises(ValueError):
        simulation.update(dt)
    assert simulation.steps == 0
    assert simulation.elapsed_seconds == 0
