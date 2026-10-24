"""Passive physics and contact-drive invariants for reproducible side-view episodes."""

from dataclasses import asdict
from math import hypot, isfinite
import subprocess
import sys

import pytest

from eidolon.behavior import Decision
from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.config import PhysicsConfig, WorldConfig
from eidolon.locomotion import ground_drive
from eidolon.simulation import Simulation
from eidolon.world import Food, World


def physical_state(simulation):
    return (asdict(simulation.life), tuple(asdict(node) for node in simulation.colony.nodes),
            tuple(simulation.colony.bonds), tuple(asdict(food) for food in simulation.world.foods),
            simulation.steps, simulation.elapsed_seconds, simulation.world.elapsed_seconds,
            simulation.world._random.getstate(), simulation.world._next_id)


@pytest.mark.parametrize("gravity", [0, 50, 180, 400])
def test_downward_acceleration_is_configurable_and_mass_independent(gravity) -> None:
    settings = PhysicsConfig(gravity=gravity, linear_damping=0, propulsion_enabled=False)
    first, second = Colony(17, (600, 400), 24, settings), Colony(17, (600, 400), 24, settings)
    for node in second.nodes:
        node.mass *= 5
    first.advance(1 / 120, (0, 0), (1200, 800))
    second.advance(1 / 120, (0, 0), (1200, 800))
    assert first.velocity == pytest.approx((0, gravity / 120), abs=1e-8)
    assert first.center == pytest.approx((600, 400 + gravity / 120**2), abs=1e-8)
    assert second.center == pytest.approx(first.center, abs=1e-8)


def test_vertical_controller_requests_cannot_cancel_gravity_or_create_flight() -> None:
    upward, downward = Colony(17, (600, 400), 24), Colony(17, (600, 400), 24)
    for _ in range(60):
        upward.advance(1 / 120, (0, -10000), (1200, 800))
        downward.advance(1 / 120, (0, 10000), (1200, 800))
    assert [asdict(node) for node in upward.nodes] == [asdict(node) for node in downward.nodes]
    assert upward.center[1] > 410 and upward.center[0] == pytest.approx(600)
    assert ground_drive(55, 0, 100, 0, PhysicsConfig(), True) == 0


def test_passive_body_ignores_all_propulsion_requests_and_settles_without_explosions() -> None:
    settings = PhysicsConfig(propulsion_enabled=False)
    first, second = Colony(17, (600, 755), 24, settings), Colony(17, (600, 755), 24, settings)
    for step in range(1200):
        first.advance(1 / 120, (55, -55), (1200, 800))
        second.advance(1 / 120, (-55, 55), (1200, 800))
        if step % 120 == 0:
            for node in first.nodes:
                radius = node.radius * MEMBRANE_RADIUS
                assert radius <= node.position[0] <= 1200 - radius
                assert radius <= node.position[1] <= 800 - radius
                assert hypot(*node.velocity) <= settings.max_speed + 1e-6
    assert [asdict(node) for node in first.nodes] == [asdict(node) for node in second.nodes]
    assert first.floor_contacts > 0 and len(first.components) == 1
    assert max(hypot(*node.velocity) for node in first.nodes) < 0.5
    assert first.broken_bonds == 0
    assert ground_drive(55, 0, 100, 10, settings, True) == 0


def test_floor_friction_dissipates_lateral_motion_without_an_active_drive() -> None:
    traveled = []
    for friction in (0, 0.8):
        colony = Colony(17, (600, 755), 24,
                        PhysicsConfig(propulsion_enabled=False, linear_damping=0, floor_friction=friction))
        for node in colony.nodes:
            node.velocity = (30, 0)
        initial = colony.center[0]
        for _ in range(360):
            colony.advance(1 / 120, (0, 0), (1200, 800))
        traveled.append(colony.center[0] - initial)
    assert traveled[0] == pytest.approx(90, abs=0.01)
    assert 0 < traveled[1] < traveled[0] - 20


def test_ground_drive_uses_support_contacts_and_is_fully_switchable() -> None:
    settings = PhysicsConfig()
    assert ground_drive(55, 0, 10, 2, settings, True) == 1200
    assert ground_drive(55, 0, 10, 2, settings, False) == 0
    assert ground_drive(55, 0, 10, 0, settings, True) == 0
    assert ground_drive(-55, 0, 10, 2, settings, True) == -1200


def test_food_falls_without_mass_dependent_acceleration_and_lands_without_bouncing() -> None:
    world = World(WorldConfig(initial_food=0, regeneration_interval=0), 17, 200,
                  PhysicsConfig(linear_damping=0))
    world.foods = [Food(1, (600, 400), 18, 5, mass=1), Food(2, (600, 400), 18, 5, mass=5)]
    world.advance(1 / 120)
    assert world.foods[0].position == world.foods[1].position
    assert world.foods[0].velocity == pytest.approx((0, 1.5))
    world.advance(4)
    assert all(food.position == (600, 795) and food.velocity == (0, 0) for food in world.foods)
    settled = tuple(world.foods)
    world.advance(5)
    assert tuple(world.foods) == settled
    assert world.consume(1) == settled[0] and world.consume(1) is None


def test_food_floor_friction_walls_and_render_interpolation() -> None:
    world = World(WorldConfig(initial_food=0, regeneration_interval=0), 17, 200,
                  PhysicsConfig(linear_damping=0))
    world.foods = [Food(1, (600, 795), 18, 5, velocity=(50, 0)),
                   Food(2, (5.1, 5.1), 18, 5, velocity=(-100, -100))]
    world.begin_step()
    world.advance(1 / 120)
    assert world.foods[1].position == (5, 5) and world.foods[1].velocity == (0, 0)
    assert world.foods[0].sample(0) == (600, 795)
    assert world.foods[0].sample(1) == world.foods[0].position
    world.advance(2)
    assert world.foods[0].velocity == (0, 0)
    assert 600 < world.foods[0].position[0] < 625


def test_food_fixed_substeps_include_regeneration_and_reproduce_large_steps() -> None:
    settings = WorldConfig(initial_food=3, regeneration_interval=0.5)
    coarse, fine = World(settings, 17, 200), World(settings, 17, 200)
    coarse.advance(3)
    for _ in range(360):
        fine.advance(1 / 120)
    assert coarse.foods == fine.foods
    assert coarse._random.getstate() == fine._random.getstate()
    assert coarse.regeneration_elapsed == fine.regeneration_elapsed


def test_pause_and_episode_reset_restore_physics_food_and_random_state() -> None:
    simulation = Simulation()
    fresh = physical_state(simulation)
    for _ in range(120):
        simulation.update(1 / 60)
    trajectory = physical_state(simulation)
    simulation.paused = True
    paused = physical_state(simulation)
    simulation.update(3)
    assert physical_state(simulation) == paused
    simulation.reset()
    assert physical_state(simulation) == fresh
    simulation.paused = False
    for _ in range(120):
        simulation.update(1 / 60)
    assert physical_state(simulation) == trajectory


def test_dead_airborne_tissue_keeps_falling_without_decisions_or_feeding() -> None:
    class NoDecision:
        def decide(self, value):
            pytest.fail("dead tissue must not evaluate a controller")

    simulation = Simulation(controller=NoDecision())
    simulation.colony.translate_to((600, 400))
    simulation.life.position = simulation.colony.center
    simulation.life.energy = 0
    simulation.update(1)
    assert not simulation.life.alive and simulation.life.age_seconds == 0
    assert simulation.life.position[1] > 400
    assert simulation.colony.grab is None
    assert all(isfinite(value) for node in simulation.colony.nodes for value in node.velocity)


def test_rules_headless_core_needs_neither_pygame_nor_numpy() -> None:
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['pygame'] = None
sys.modules['numpy'] = None
from eidolon.simulation import Simulation
s = Simulation()
s.update(1 / 60)
assert s.steps == 1 and s.life.alive and s.colony.elapsed_seconds > 0
"""], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("values", [dict(gravity=-1), dict(gravity=float('nan')), dict(floor_friction=-1),
    dict(linear_damping=float('inf')), dict(max_speed=0), dict(max_speed=True), dict(contact_slop=-1),
    dict(propulsion_enabled=1), dict(drive_acceleration=-1)])
def test_invalid_physics_configuration_is_rejected(values) -> None:
    with pytest.raises(ValueError):
        PhysicsConfig(**values)
