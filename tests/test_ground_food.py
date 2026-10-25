from math import hypot

import pytest

from eidolon.colony import MEMBRANE_RADIUS
from eidolon.config import WorldConfig
from eidolon.simulation import Simulation
from eidolon.world import World


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_ground_food_is_seeded_spread_and_outside_body_and_walls(seed):
    first, second = Simulation(seed), Simulation(seed)
    assert first.world.foods == second.world.foods and len(first.world.foods) == 32
    settings = first.world_config
    for food in first.world.foods:
        assert food.position[1] == settings.height - food.radius and food.velocity == (0, 0)
        assert settings.food_wall_clearance <= food.position[0] <= settings.width - settings.food_wall_clearance
        for other in first.world.foods:
            if food.identifier != other.identifier:
                assert hypot(food.position[0] - other.position[0], food.position[1] - other.position[1]) >= food.radius + other.radius + settings.food_spacing - 1e-7
        for node in first.colony.nodes:
            assert hypot(food.position[0] - node.position[0], food.position[1] - node.position[1]) >= food.radius + node.radius * MEMBRANE_RADIUS + settings.food_body_clearance - 1e-7


def test_regeneration_checks_current_body_footprint_and_crowding_defers_spawns():
    simulation = Simulation(world_config=WorldConfig(initial_food=0, regeneration_interval=0.1))
    simulation.colony.translate_to((850, 755))
    simulation.world.advance(0.5)
    assert len(simulation.world.foods) == 5
    for food in simulation.world.foods:
        for node in simulation.colony.nodes:
            assert hypot(food.position[0] - node.position[0], food.position[1] - node.position[1]) >= food.radius + node.radius * MEMBRANE_RADIUS + simulation.world_config.food_body_clearance - 1e-7
    tiny = World(WorldConfig(width=100, height=80, initial_food=0, food_wall_clearance=10), 0, 15,
                 exclusions=lambda: [(50, 75, 100)])
    assert tiny.spawn_food() is None
    assert tiny.foods == []


def test_falling_distribution_remains_available_and_ground_food_can_be_consumed_once():
    first = World(WorldConfig(food_distribution="falling", initial_food=5), 17, 200)
    assert all(200 <= food.position[1] <= 600 for food in first.foods)
    world = World(WorldConfig(initial_food=1), 17, 200)
    food = world.foods[0]
    assert world.consume(food.identifier) == food and world.consume(food.identifier) is None
    blocked = World(WorldConfig(food_distribution="falling", initial_food=0), 17, 200,
                    exclusions=lambda: [(600, 400, 2000)])
    assert blocked.spawn_food() is None


@pytest.mark.parametrize("options", [dict(food_distribution="invalid"), dict(food_spacing=-1),
    dict(food_wall_clearance=float("nan")), dict(food_body_clearance=-1)])
def test_food_distribution_configuration_is_validated(options):
    with pytest.raises(ValueError):
        WorldConfig(**options)
