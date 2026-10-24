import pytest

from eidolon.config import LifeConfig, WorldConfig
from eidolon.simulation import Simulation
from eidolon.world import Food, World


def test_resources_are_seeded_capped_and_consumed_once() -> None:
    settings = WorldConfig(width=100, height=80, initial_food=3, food_capacity=3)
    first, second = World(settings, 17, 15), World(settings, 17, 15)
    assert first.foods == second.foods
    assert first.foods != World(settings, 18, 15).foods
    assert first.spawn_food() is None
    food = first.foods[0]
    assert first.consume(food.identifier) == food
    assert first.consume(food.identifier) is None
    replacement = first.spawn_food()
    assert replacement.identifier > food.identifier
    assert len(first.foods) == 3


def test_regeneration_uses_simulation_time_and_never_banks_full_world_spawns() -> None:
    settings = WorldConfig(width=100, height=80, initial_food=0, food_capacity=3, regeneration_interval=0.5)
    world = World(settings, 17, 15)
    world.advance(1.2)
    assert len(world.foods) == 2
    assert world.regeneration_elapsed == pytest.approx(0.2)
    world.advance(0.3)
    assert len(world.foods) == 3
    random_state = world._random.getstate()
    world.advance(100)
    assert len(world.foods) == 3
    assert world._random.getstate() == random_state
    world.consume(world.foods[0].identifier)
    world.advance(0.25)
    assert len(world.foods) == 2
    world.advance(0.25)
    assert len(world.foods) == 3


def test_regeneration_can_be_disabled() -> None:
    settings = WorldConfig(initial_food=0, regeneration_interval=0)
    world = World(settings, 17, 200)
    world.advance(1000)
    assert not world.foods


def test_spawn_locations_are_reachable_even_at_maximum_growth() -> None:
    simulation = Simulation()
    required = simulation.life_config.base_radius * simulation.life_config.max_scale * 6
    assert simulation.world.spawn_margin >= required + simulation.world_config.food_radius
    for food in simulation.world.foods:
        x, y = food.position
        assert required <= x <= simulation.world_config.width - required
        assert required <= y <= simulation.world_config.height - required


def test_nearest_food_has_a_range_limit_and_stable_tie_breaking() -> None:
    world = World(WorldConfig(initial_food=0), 17, 200)
    right, left = Food(2, (610, 400), 18, 5), Food(1, (590, 400), 18, 5)
    world.foods = [right, left]
    assert world.nearest_food((600, 400), 9) is None
    assert world.nearest_food((600, 400), 10) == left


def test_world_must_fit_the_organism_at_full_growth() -> None:
    with pytest.raises(ValueError, match="room"):
        Simulation(world_config=WorldConfig(width=100, height=100), life_config=LifeConfig())


@pytest.mark.parametrize("dt", [-1, float("nan"), float("inf")])
def test_invalid_world_time_is_rejected(dt) -> None:
    world = World(WorldConfig(), 17, 200)
    with pytest.raises(ValueError):
        world.advance(dt)
