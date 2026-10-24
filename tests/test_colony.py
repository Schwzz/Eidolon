from dataclasses import asdict
from math import cos, hypot, isfinite, sin

import pytest

from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.config import LifeConfig, PhysicsConfig, WorldConfig
from eidolon.simulation import Simulation


def strain(colony):
    return max(abs(hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                         colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1])
                   / (b.rest_length * colony.scale) - 1) for b in colony.bonds)


def physical_state(colony):
    return tuple(asdict(node) for node in colony.nodes), tuple(colony.bonds), colony.elapsed_seconds, colony.scale


@pytest.mark.parametrize("seed", [0, 1, 17, 42, 999, -3])
def test_seeded_colony_has_dense_bounded_connected_nodes_with_physical_data(seed) -> None:
    colony = Colony(seed, (600, 400), 24)
    assert 85 <= len(colony.nodes) <= 98
    assert max(node.radius for node in colony.nodes) < 3.7  # Previously roughly 8 world units.
    assert physical_state(colony) == physical_state(Colony(seed, (600, 400), 24))
    reached, pending = set(), [0]
    while pending:
        identifier = pending.pop()
        if identifier not in reached:
            reached.add(identifier)
            pending.extend(colony.nodes[identifier].connections)
    assert len(reached) == len(colony.nodes)
    for node in colony.nodes:
        assert node.mass > 0 and node.radius > 0
        assert node.velocity == (0, 0)
        assert node.connections
        for neighbor in node.connections:
            assert node.identifier in colony.nodes[neighbor].connections
    assert colony.triangles
    assert all(bond.rest_length > 0 and bond.compliance > 0 for bond in colony.bonds)


def test_different_seeds_change_shape_and_mass_distribution() -> None:
    assert physical_state(Colony(17, (600, 400), 24)) != physical_state(Colony(18, (600, 400), 24))


def test_force_drives_individual_nodes_with_lag_and_visible_strain() -> None:
    colony = Colony(17, (600, 755), 24)
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800), active=False)
    initial = [node.position for node in colony.nodes]
    for _ in range(120):
        colony.advance(1 / 120, (70, 0), (1200, 800))
    deltas = [node.position[0] - point[0] for node, point in zip(colony.nodes, initial)]
    assert colony.center[0] > 630
    assert max(deltas) - min(deltas) > 1
    assert max(node.velocity[0] for node in colony.nodes) - min(node.velocity[0] for node in colony.nodes) > 1
    assert 0.02 < strain(colony) < 0.75  # Gravity compresses the supported mesh.


def test_stretched_bonds_recover_without_restoring_a_rigid_pose() -> None:
    colony = Colony(17, (600, 400), 24, PhysicsConfig(gravity=0, propulsion_enabled=False))
    node = colony.nodes[-1]
    node.position = (node.position[0] + colony.spacing * 0.8, node.position[1])
    stretched = strain(colony)
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert stretched > 0.2
    assert strain(colony) < 0.05


def test_large_timestep_matches_the_same_fixed_substeps() -> None:
    coarse, fine = Colony(17, (600, 400), 24), Colony(17, (600, 400), 24)
    coarse.advance(1, (25, 10), (1200, 800))
    for _ in range(120):
        fine.advance(1 / 120, (25, 10), (1200, 800))
    assert physical_state(coarse) == physical_state(fine)


def test_overlapping_nodes_separate_and_large_velocity_is_bounded() -> None:
    colony = Colony(17, (600, 400), 24)
    colony.nodes[0].position = colony.nodes[1].position
    colony.nodes[-1].velocity = (1e8, -1e8)
    for _ in range(120):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    first, second = colony.nodes[:2]
    assert hypot(first.position[0] - second.position[0], first.position[1] - second.position[1]) > (first.radius + second.radius) * 0.8
    assert all(hypot(*node.velocity) <= 220.000001 for node in colony.nodes)
    assert strain(colony) < 0.1


def test_nodes_deform_against_world_walls_without_leaving_them() -> None:
    colony = Colony(42, (150, 100), 24)
    contacts = 0
    for _ in range(360):
        colony.advance(1 / 120, (-70, -30), (300, 200))
        contacts += colony.wall_contacts
        for node in colony.nodes:
            radius = node.radius * MEMBRANE_RADIUS
            assert radius <= node.position[0] <= 300 - radius
            assert radius <= node.position[1] <= 200 - radius
    assert contacts > 0
    assert strain(colony) < 0.75


def test_prolonged_motion_remains_finite_cohesive_and_bounded() -> None:
    colony = Colony(0, (600, 400), 24)
    for step in range(6000):
        t = step / 120
        colony.advance(1 / 120, (55 * cos(t * 0.23), 55 * sin(t * 0.31)), (1200, 800))
        if step % 120 == 0:
            assert strain(colony) < 0.8
            for node in colony.nodes:
                assert all(isfinite(value) for value in (*node.position, *node.velocity))
                assert 0 <= node.position[0] <= 1200 and 0 <= node.position[1] <= 800


def test_growth_expands_actual_node_geometry_without_adding_nodes() -> None:
    colony = Colony(17, (600, 400), 24)
    original = [node.position for node in colony.nodes]
    initial_width = max(p[0] for p in original) - min(p[0] for p in original)
    colony.set_growth(1.3)
    assert [node.position for node in colony.nodes] == original  # No rigid rescale.
    for _ in range(240):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    width = max(node.position[0] for node in colony.nodes) - min(node.position[0] for node in colony.nodes)
    assert width > initial_width * 1.2
    assert len(colony.nodes) == len(original)
    assert all(node.radius == node.rest_radius * 1.3 for node in colony.nodes)


def test_simulation_uses_forces_not_setup_translation(monkeypatch) -> None:
    simulation = Simulation(world_config=WorldConfig(initial_food=0, regeneration_interval=0),
                            life_config=LifeConfig(metabolism=0, movement_cost=0))

    def forbidden(*args):
        raise AssertionError("rigid setup translation used for locomotion")

    monkeypatch.setattr(simulation.colony, "translate_to", forbidden)
    start = simulation.colony.center
    simulation.update(1)
    assert simulation.life.position == simulation.colony.center != start


def test_pause_reset_and_death_apply_to_all_nodes() -> None:
    simulation = Simulation()
    fresh = physical_state(simulation.colony)
    simulation.update(1)
    simulation.paused = True
    frozen = physical_state(simulation.colony)
    simulation.update(1)
    assert physical_state(simulation.colony) == frozen
    simulation.reset()
    assert physical_state(simulation.colony) == fresh
    simulation.paused = False
    simulation.life.energy = 0
    simulation.update(1 / 60)
    age = simulation.life.age_seconds
    elapsed = simulation.colony.elapsed_seconds
    simulation.update(1)
    assert simulation.life.age_seconds == age and not simulation.life.alive
    assert simulation.colony.elapsed_seconds > elapsed  # Corpses remain passive physical objects.
    assert all(isfinite(value) for node in simulation.colony.nodes for value in node.velocity)


@pytest.mark.parametrize("dt", [0, -1, float("inf"), float("nan")])
def test_invalid_timestep_is_rejected(dt) -> None:
    with pytest.raises(ValueError):
        Colony(17, (600, 400), 24).advance(dt, (0, 0), (1200, 800))


def test_nonfinite_state_is_reported_not_silently_reseeded() -> None:
    colony = Colony(17, (600, 400), 24)
    colony.nodes[0].velocity = (float("nan"), 0)
    with pytest.raises(FloatingPointError):
        colony.advance(1 / 120, (0, 0), (1200, 800))


@pytest.mark.parametrize("world", [(0, 800), (1200, float("nan")), (5, 5)])
def test_invalid_or_impossibly_small_world_is_rejected(world) -> None:
    with pytest.raises(ValueError):
        Colony(17, (600, 400), 24).advance(1 / 120, (0, 0), world)
