from dataclasses import asdict
from math import hypot, pi

import pytest

from eidolon.behavior import Decision
from eidolon.config import LifeConfig, WorldConfig
from eidolon.simulation import Simulation
from eidolon.world import Food
from eidolon.colony import MEMBRANE_RADIUS


EMPTY = WorldConfig(initial_food=0, food_capacity=0, regeneration_interval=0)


class FixedController:
    def __init__(self, direction=0, speed=1):
        self.direction = direction
        self.speed = speed
        self.senses = []

    def decide(self, senses):
        self.senses.append(senses)
        return Decision(self.direction, self.speed)


def place_food(simulation, position=None, energy=18):
    food = Food(1000, position or simulation.life.position, energy, simulation.world.config.food_radius)
    simulation.world.foods = [food]
    return food


def snapshot(simulation):
    return (
        asdict(simulation.life), tuple(simulation.world.foods), simulation.steps,
        simulation.elapsed_seconds, simulation.world.elapsed_seconds,
        simulation.world.regeneration_elapsed, simulation.world._random.getstate(),
        simulation.world._next_id, simulation.organism.age_seconds,
        simulation.organism.previous_age_seconds,
        tuple(asdict(node) for node in simulation.colony.nodes),
        tuple(simulation.colony.bonds), simulation.colony.elapsed_seconds,
    )


def test_movement_velocity_age_and_energy_cost_use_world_units() -> None:
    controller = FixedController()
    config = LifeConfig(growth_rate=0)
    simulation = Simulation(world_config=EMPTY, life_config=config, controller=controller)
    state = simulation.life
    state.direction = 0
    start = state.position
    simulation.update(0.1)
    assert state.position[0] > start[0]
    assert 0 < state.speed < config.movement_speed  # Finite force-driven acceleration.
    assert state.position == simulation.colony.center
    traveled = hypot(state.position[0] - start[0], state.position[1] - start[1])
    assert state.energy == pytest.approx(70 - 0.1 - traveled * 0.025, abs=1e-5)
    assert state.age_seconds == pytest.approx(0.1)
    assert state.health == pytest.approx(state.energy)


def test_stationary_organism_pays_metabolism_but_no_movement_cost() -> None:
    config = LifeConfig(metabolism=2, movement_cost=100)
    simulation = Simulation(world_config=EMPTY, life_config=config, controller=FixedController(speed=0))
    initial_position = simulation.life.position
    simulation.update(1)
    assert simulation.life.energy == pytest.approx(68)
    assert simulation.life.position == pytest.approx(initial_position)
    assert simulation.life.velocity == pytest.approx((0, 0), abs=1e-8)


def test_rules_wander_when_no_food_is_visible() -> None:
    simulation = Simulation(world_config=EMPTY)
    start, heading = simulation.life.position, simulation.life.direction
    simulation.update(3)
    assert simulation.life.position != start
    assert simulation.life.direction != heading
    assert simulation.life.foods_eaten == 0


def test_turning_toward_food_is_rate_limited() -> None:
    simulation = Simulation(world_config=EMPTY)
    simulation.life.direction = 0
    x, y = simulation.life.position
    place_food(simulation, (x - 150, y))
    simulation.update(1 / 60)
    assert simulation.life.direction == pytest.approx(simulation.life_config.turn_rate / 60)


def test_sensing_has_a_finite_range_and_controller_is_replaceable() -> None:
    controller = FixedController(speed=0)
    config = LifeConfig(sensing_radius=60, feeding_radius=5)
    simulation = Simulation(world_config=EMPTY, life_config=config, controller=controller)
    x, y = simulation.life.position
    place_food(simulation, (x + 100, y))
    simulation.update(1 / 60)
    assert controller.senses[-1].food is None
    food = place_food(simulation, (x + 50, y))
    simulation.update(1 / 60)
    assert controller.senses[-1].food == food
    assert simulation.life.foods_eaten == 0


def test_rules_approach_and_consume_detected_food() -> None:
    config = LifeConfig(metabolism=0, movement_cost=0)
    simulation = Simulation(world_config=EMPTY, life_config=config)
    x, y = simulation.life.position
    target = place_food(simulation, (x + 150, y))
    simulation.life.direction = 0
    simulation.life.energy = 40
    simulation.update(3)
    assert simulation.life.foods_eaten == 1
    assert simulation.world.consume(target.identifier) is None
    assert simulation.life.energy > 40
    assert simulation.life.nutrition == 18


def test_food_restores_energy_once_and_caps_it_at_maximum() -> None:
    config = LifeConfig(movement_speed=0, metabolism=0, movement_cost=0, growth_rate=0)
    simulation = Simulation(world_config=EMPTY, life_config=config)
    simulation.life.energy = 95
    food = place_food(simulation)
    simulation.update(1 / 60)
    assert simulation.life.energy == 100
    assert simulation.life.foods_eaten == 1
    assert simulation.life.nutrition == 18
    assert simulation.world.consume(food.identifier) is None
    simulation.update(1)
    assert simulation.life.foods_eaten == 1
    assert simulation.life.energy == 100


def test_growth_is_gradual_uses_nutrition_and_spends_energy() -> None:
    config = LifeConfig(movement_speed=0, metabolism=0, movement_cost=0, growth_rate=0.1)
    simulation = Simulation(world_config=EMPTY, life_config=config)
    place_food(simulation)
    simulation.update(1)
    assert simulation.life.body_scale == pytest.approx(1.1)
    assert simulation.life.physical_radius == pytest.approx(config.base_radius * 1.1)
    assert simulation.life.energy == pytest.approx(88 - 0.1 * config.growth_energy_cost)


@pytest.mark.parametrize("nutrition,energy", [(0, 100), (18, 20)])
def test_growth_requires_both_nutrition_and_available_energy(nutrition, energy) -> None:
    simulation = Simulation(world_config=EMPTY, life_config=LifeConfig(movement_speed=0, metabolism=0))
    simulation.life.nutrition = nutrition
    simulation.life.energy = energy
    simulation.update(10)
    assert simulation.life.body_scale == 1


def test_growth_is_capped_and_waits_for_room_at_a_boundary() -> None:
    config = LifeConfig(movement_speed=0, metabolism=0, growth_rate=0.2)
    simulation = Simulation(world_config=EMPTY, life_config=config)
    simulation.life.energy = 100
    simulation.life.nutrition = 1000
    simulation.life.position = (simulation.life.collision_radius, simulation.life.position[1])
    simulation.colony.translate_to(simulation.life.position)
    simulation.update(1)
    assert simulation.life.body_scale == pytest.approx(1)
    simulation.life.position = (600, 400)
    simulation.colony.translate_to(simulation.life.position)
    simulation.update(10)
    assert simulation.life.body_scale == config.max_scale
    assert simulation.life.physical_radius == config.base_radius * config.max_scale


def test_growth_cannot_leave_an_organism_alive_at_zero_energy() -> None:
    settings = LifeConfig(initial_energy=1, metabolism=0, movement_speed=0,
                          growth_energy_threshold=0, growth_energy_cost=1_000_000)
    simulation = Simulation(world_config=EMPTY, life_config=settings)
    simulation.life.nutrition = 100
    simulation.update(1 / 60)
    assert simulation.life.energy == 0
    assert not simulation.life.alive


def test_starvation_kills_and_dead_organisms_do_not_decide_eat_move_or_grow() -> None:
    controller = FixedController(speed=0)
    config = LifeConfig(initial_energy=1, metabolism=4)
    simulation = Simulation(world_config=EMPTY, life_config=config, controller=controller)
    simulation.update(1)
    assert not simulation.life.alive
    assert simulation.life.energy == simulation.life.health == 0
    assert simulation.life.velocity == (0, 0)
    calls = len(controller.senses)
    biological = asdict(simulation.life)
    frozen_pose = simulation.organism.pose()
    food = place_food(simulation)
    simulation.update(10)
    assert len(controller.senses) == calls
    assert asdict(simulation.life) == biological
    assert simulation.organism.pose() == frozen_pose
    assert food in simulation.world.foods
    assert simulation.world.elapsed_seconds == pytest.approx(11)


@pytest.mark.parametrize("direction", [0, pi / 2, pi, 3 * pi / 2])
def test_whole_organism_remains_within_world_boundaries(direction) -> None:
    config = LifeConfig(metabolism=0, movement_cost=0)
    simulation = Simulation(world_config=EMPTY, life_config=config, controller=FixedController(direction))
    radius = simulation.life.collision_radius
    simulation.life.position = (
        1200 - radius - 0.01 if direction == 0 else radius + 0.01 if direction == pi else 600,
        800 - radius - 0.01 if direction == pi / 2 else radius + 0.01 if direction == 3 * pi / 2 else 400,
    )
    simulation.life.direction = direction
    simulation.colony.translate_to(simulation.life.position)
    for _ in range(300):
        simulation.update(1 / 60)
        for node in simulation.colony.nodes:
            x, y = node.position
            radius = node.radius * MEMBRANE_RADIUS
            assert radius <= x <= 1200 - radius
            assert radius <= y <= 800 - radius


def test_pause_freezes_complete_world_biology_animation_and_random_state() -> None:
    simulation = Simulation()
    simulation.update(1)
    simulation.set_speed(2)
    simulation.paused = True
    frozen = snapshot(simulation)
    for _ in range(120):
        simulation.update(1 / 60)
    assert snapshot(simulation) == frozen


def test_restart_restores_world_food_biology_and_future_random_spawns() -> None:
    simulation = Simulation(42)
    fresh = Simulation(42)
    simulation.update(20)
    assert simulation.life.foods_eaten > 0
    simulation.paused = True
    simulation.set_speed(2)
    simulation.reset()
    assert snapshot(simulation) == snapshot(fresh)
    assert simulation.paused and simulation.speed == 2
    assert simulation.world.spawn_food() == fresh.world.spawn_food()


def test_speed_scales_resources_biology_and_animation_consistently() -> None:
    fast, normal = Simulation(17), Simulation(17)
    fast.set_speed(2)
    for _ in range(120):
        fast.update(1 / 60)
        normal.update(1 / 60)
        normal.update(1 / 60)
    assert fast.life.position == normal.life.position
    assert fast.life.energy == normal.life.energy
    assert fast.life.body_scale == normal.life.body_scale
    assert fast.life.foods_eaten == normal.life.foods_eaten
    assert fast.world.foods == normal.world.foods
    assert fast.world.elapsed_seconds == normal.world.elapsed_seconds
    assert fast.organism.age_seconds == normal.organism.age_seconds


def test_identical_seeds_and_inputs_produce_identical_worlds() -> None:
    first, second = Simulation(17), Simulation(17)
    for _ in range(1200):
        first.update(1 / 60)
        second.update(1 / 60)
    assert snapshot(first) == snapshot(second)
    assert first.world.foods != Simulation(18).world.foods


def test_large_timestep_is_subdivided_without_tunneling_past_food() -> None:
    coarse, fine = Simulation(17), Simulation(17)
    coarse.update(10)
    for _ in range(600):
        fine.update(1 / 60)
    assert coarse.life.position == fine.life.position
    assert coarse.life.energy == fine.life.energy
    assert coarse.life.foods_eaten == fine.life.foods_eaten > 0
    assert coarse.world.foods == fine.world.foods


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_default_ecosystem_demonstrates_exploration_feeding_and_growth(seed) -> None:
    simulation = Simulation(seed)
    start = simulation.life.position
    for _ in range(3600):
        simulation.update(1 / 60)
    assert hypot(simulation.life.position[0] - start[0], simulation.life.position[1] - start[1]) > 10
    assert simulation.life.foods_eaten >= 5
    assert simulation.life.body_scale > 1.4
    assert simulation.life.alive
    assert 0 < simulation.life.energy <= simulation.life.max_energy
