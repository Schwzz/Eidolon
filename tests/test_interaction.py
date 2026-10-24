from copy import deepcopy
from dataclasses import asdict
from math import hypot, isfinite

import pytest

from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.config import AppConfig, LifeConfig, WorldConfig
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock


def state(colony):
    return (tuple(asdict(node) for node in colony.nodes), tuple(colony.bonds), colony.components,
            asdict(colony.grab) if colony.grab else None, colony.elapsed_seconds,
            colony.broken_bonds, colony.reconnected_bonds, dict(colony._tension), dict(colony._cooldown))


def start_pull(colony):
    node = max(colony.nodes, key=lambda candidate: candidate.position[0])
    origin = node.position
    assert colony.begin_grab(origin) == node.identifier
    return node.identifier, origin


@pytest.fixture(scope="module")
def torn_template():
    colony = Colony(17, (600, 400), 24)
    identifier, origin = start_pull(colony)
    for frame in range(360):
        colony.move_grab((origin[0] + (frame + 1), origin[1]))
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert colony.broken_bonds > 0 and len(colony.components) > 1
    return colony, identifier


def test_grab_selects_nearest_node_preserves_offset_and_never_teleports() -> None:
    colony = Colony(17, (600, 400), 24)
    node = colony.nodes[0]
    positions = [particle.position for particle in colony.nodes]
    pointer = (node.position[0] + 2, node.position[1] + 1)
    assert colony.begin_grab(pointer) == 0
    assert colony.grab.offset == (-2, -1)
    colony.move_grab((1000, 400))
    assert [particle.position for particle in colony.nodes] == positions
    colony.advance(1 / 120, (0, 0), (1200, 800))
    assert 0 < hypot(node.position[0] - positions[0][0], node.position[1] - positions[0][1]) < 5
    assert colony.grab.applied_force <= 14000
    colony.release_grab()
    assert colony.grab is None
    assert colony.begin_grab((50, 50)) is None


def test_gentle_pull_stretches_without_fragmenting() -> None:
    colony = Colony(17, (600, 400), 24)
    identifier, origin = start_pull(colony)
    start_bonds = tuple(colony.bonds)
    maximum_strain = 0.0
    for frame in range(480):
        colony.move_grab((origin[0] + 8 * (frame + 1) / 120, origin[1]))
        colony.advance(1 / 120, (0, 0), (1200, 800))
        for bond in colony.bonds:
            a, b = colony.nodes[bond.first], colony.nodes[bond.second]
            maximum_strain = max(maximum_strain, abs(hypot(a.position[0] - b.position[0], a.position[1] - b.position[1]) / bond.rest_length - 1))
    assert tuple(colony.bonds) == start_bonds
    assert colony.broken_bonds == 0 and len(colony.components) == 1
    assert maximum_strain > 0.01
    assert colony.nodes[identifier].position[0] > origin[0] + 5


def test_brief_pointer_jump_does_not_tear() -> None:
    colony = Colony(17, (600, 400), 24)
    _, origin = start_pull(colony)
    colony.move_grab((origin[0] + 400, origin[1]))
    for _ in range(4):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    colony.release_grab()
    for _ in range(120):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert colony.broken_bonds == 0


def test_forceful_pull_breaks_only_local_bonds_and_updates_components(torn_template) -> None:
    colony, grabbed = torn_template
    original = Colony(17, (600, 400), 24)
    remaining = {(bond.first, bond.second) for bond in colony.bonds}
    removed = [(bond.first, bond.second) for bond in original.bonds if (bond.first, bond.second) not in remaining]
    assert removed and all(grabbed in edge for edge in removed)
    assert sum(map(len, colony.components)) == len(colony.nodes)
    assert colony.components[0] == tuple(sorted(colony.components[0]))
    assert len(colony.components[0]) > len(colony.components[1])
    assert all(grabbed not in triangle for triangle in colony.triangles)
    for node in colony.nodes:
        for neighbor in node.connections:
            assert node.identifier in colony.nodes[neighbor].connections


def test_released_fragment_recovers_and_forms_a_stable_bond(torn_template) -> None:
    colony = deepcopy(torn_template[0])
    colony.release_grab()
    before = len(colony.bonds)
    for _ in range(120 * 12):
        colony.advance(1 / 120, (20, 0), (1200, 800))
        if len(colony.components) == 1:
            break
    assert len(colony.components) == 1
    assert colony.reconnected_bonds >= 1
    assert len(colony.bonds) > before
    healed = [bond for bond in colony.bonds if bond.created_at >= 0]
    assert all(bond.rest_length > 0 and bond.compliance > 0 for bond in healed)
    broken = colony.broken_bonds
    for _ in range(240):
        colony.advance(1 / 120, (20, 0), (1200, 800))
    assert len(colony.components) == 1 and colony.broken_bonds == broken


def test_holding_a_fragment_does_not_snap_it_back_or_heal_under_the_pointer(torn_template) -> None:
    colony = deepcopy(torn_template[0])
    for _ in range(120):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert colony.grab is not None
    assert len(colony.components) > 1
    assert colony.reconnected_bonds == 0


def test_connected_components_and_triangles_follow_graph_edits() -> None:
    colony = Colony(17, (600, 400), 24)
    isolated = len(colony.nodes) - 1
    colony.bonds = [bond for bond in colony.bonds if isolated not in (bond.first, bond.second)]
    colony.rebuild_topology()
    assert colony.components[-1] == (isolated,)
    assert colony.nodes[isolated].connections == ()
    assert all(isolated not in triangle for triangle in colony.triangles)
    assert colony.center == colony._mean(colony.components[0], "position")


def test_multinode_fragment_moves_toward_and_reconnects_with_largest_component() -> None:
    colony = Colony(17, (600, 400), 24)
    smaller = set(sorted(range(len(colony.nodes)), key=lambda index: colony.nodes[index].position[0])[:11])
    colony.bonds = [bond for bond in colony.bonds if (bond.first in smaller) == (bond.second in smaller)]
    colony.rebuild_topology()
    assert tuple(map(len, colony.components)) == (12, 11)
    for identifier in smaller:
        node = colony.nodes[identifier]
        node.position = (node.position[0] - 200, node.position[1])
    main = colony.center
    fragment = colony._mean(tuple(sorted(smaller)), "position")
    initial_distance = hypot(fragment[0] - main[0], fragment[1] - main[1])
    for _ in range(120):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    fragment = colony._mean(tuple(sorted(smaller)), "position")
    assert hypot(fragment[0] - colony.center[0], fragment[1] - colony.center[1]) < initial_distance - 20
    for _ in range(1200):
        colony.advance(1 / 120, (0, 0), (1200, 800))
        if len(colony.components) == 1:
            break
    assert len(colony.components) == 1 and colony.reconnected_bonds > 0


def test_component_membership_change_does_not_spend_energy_for_a_centroid_jump() -> None:
    simulation = Simulation(world_config=WorldConfig(initial_food=0, regeneration_interval=0),
                            life_config=LifeConfig(metabolism=0, movement_speed=0, movement_cost=100, growth_rate=0))
    node = max(simulation.colony.nodes, key=lambda candidate: candidate.position[0])
    simulation.colony.bonds = [bond for bond in simulation.colony.bonds if node.identifier not in (bond.first, bond.second)]
    simulation.colony.rebuild_topology()
    node.position = (950, 400)
    old_position = simulation.life.position
    simulation.update(1 / 60)
    assert simulation.life.position != old_position
    assert simulation.life.energy == pytest.approx(70, abs=1e-7)


def test_identical_pointer_inputs_reproduce_breaks_and_reconnection() -> None:
    first, second = Colony(17, (600, 400), 24), Colony(17, (600, 400), 24)
    _, origin = start_pull(first)
    start_pull(second)
    for step in range(960):
        if step < 300:
            target = (origin[0] + step * 1.2, origin[1])
            first.move_grab(target)
            second.move_grab(target)
        elif step == 300:
            first.release_grab()
            second.release_grab()
        first.advance(1 / 120, (0, 0), (1200, 800))
        second.advance(1 / 120, (0, 0), (1200, 800))
    assert state(first) == state(second)
    assert first.broken_bonds > 0 and first.reconnected_bonds > 0


def test_pointer_replay_is_independent_of_render_framerate() -> None:
    results = []
    for fps in (30, 60, 144):
        simulation = Simulation(17, life_config=LifeConfig(metabolism=0, movement_cost=0))
        clock = FixedStepClock(AppConfig())
        origin = max(simulation.colony.nodes, key=lambda node: node.position[0]).position

        def tick(dt):
            step = simulation.steps
            if step == 10:
                simulation.begin_grab(max(simulation.colony.nodes, key=lambda node: node.position[0]).position)
            if 10 <= step < 100:
                simulation.move_grab((origin[0] + (step - 10) * 2, origin[1]))
            if step == 100:
                simulation.release_grab()
            simulation.update(dt)

        for _ in range(fps * 3):
            clock.advance(1 / fps, tick)
        results.append((state(simulation.colony), asdict(simulation.life), tuple(simulation.world.foods)))
    assert results[0] == results[1] == results[2]


def test_pause_restart_and_death_cancel_grabs_and_preserve_single_colony() -> None:
    simulation = Simulation()
    node = simulation.colony.nodes[0]
    assert simulation.begin_grab(node.position) == 0
    simulation.paused = True
    assert simulation.colony.grab is None
    frozen = state(simulation.colony)
    assert simulation.begin_grab(node.position) is None
    simulation.update(1)
    assert state(simulation.colony) == frozen
    simulation.paused = False
    simulation.begin_grab(node.position)
    simulation.reset()
    assert simulation.colony.grab is None and len(simulation.colony.components) == 1
    simulation.begin_grab(simulation.colony.nodes[0].position)
    simulation.life.energy = 0
    simulation.update(1 / 60)
    assert simulation.colony.grab is None and not simulation.life.alive
    assert simulation.begin_grab(simulation.colony.nodes[0].position) is None
    assert all(node.velocity == (0, 0) for node in simulation.colony.nodes)


def test_far_pointer_targets_remain_force_velocity_and_boundary_limited() -> None:
    colony = Colony(17, (600, 400), 24)
    start_pull(colony)
    colony.move_grab((1e12, -1e12))
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
        assert colony.grab.applied_force <= 14000.000001
        for node in colony.nodes:
            r = node.radius * MEMBRANE_RADIUS
            assert r <= node.position[0] <= 1200 - r
            assert r <= node.position[1] <= 800 - r
            assert hypot(*node.velocity) <= 220.000001


def test_repeated_interaction_remains_bounded_and_recovers_all_fragments() -> None:
    colony = Colony(17, (600, 400), 24)
    original_count = len(colony.nodes)
    for cycle in range(3):
        _, origin = start_pull(colony)
        for frame in range(240):
            colony.move_grab((origin[0] + (frame + 1) * 1.5, origin[1] + cycle * 20))
            colony.advance(1 / 120, (0, 0), (1200, 800))
        colony.release_grab()
        for _ in range(1200):
            colony.advance(1 / 120, (0, 0), (1200, 800))
        assert len(colony.components) == 1
        assert len(colony.nodes) == original_count
        assert all(isfinite(value) for node in colony.nodes for value in (*node.position, *node.velocity))
    assert colony.broken_bonds > 0 and colony.reconnected_bonds > 0


@pytest.mark.parametrize("position", [(float("nan"), 0), (0, float("inf"))])
def test_nonfinite_pointer_inputs_are_rejected(position) -> None:
    colony = Colony(17, (600, 400), 24)
    with pytest.raises(ValueError):
        colony.begin_grab(position)
    with pytest.raises(ValueError):
        colony.move_grab(position)
