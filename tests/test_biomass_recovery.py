"""Dense tissue recovery, including contacts the former single-bridge repair missed."""

from math import hypot, isfinite

import pytest

from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.spatial import nearby_pairs


def isolate_tip(colony):
    node = max(colony.nodes, key=lambda item: item.position[0])
    removed = [bond for bond in colony.bonds if node.identifier in (bond.first, bond.second)]
    colony.bonds = [bond for bond in colony.bonds if bond not in removed]
    colony.broken_bonds += len(removed)
    colony._scars.update((node.identifier,))
    colony.rebuild_topology()
    return node


def assert_bounded(colony):
    assert sum(map(len, colony.components)) == len(colony.nodes)
    assert len({(bond.first, bond.second) for bond in colony.bonds}) == len(colony.bonds)
    assert max(len(node.connections) for node in colony.nodes) <= colony.MAX_NEIGHBORS
    for node in colony.nodes:
        assert all(isfinite(value) for value in (*node.position, *node.velocity))
        radius = node.radius * MEMBRANE_RADIUS
        assert radius <= node.position[0] <= 1200 - radius
        assert radius <= node.position[1] <= 800 - radius
        assert hypot(*node.velocity) <= 220 * colony.scale**0.5 + 1e-6


def test_fast_nearby_fragment_is_captured_without_a_velocity_veto_or_single_bridge() -> None:
    colony = Colony(17, (600, 400), 24)
    initial_bonds = len(colony.bonds)
    node = isolate_tip(colony)
    node.position = (node.position[0] + colony.spacing * 0.4, node.position[1])
    node.velocity = (-150, 0)  # Previously rejected above 90 even at close contact.
    colony._heal()
    assert len(colony.components) == 1
    assert len(node.connections) >= 2
    assert len(colony.bonds) == initial_bonds - colony.broken_bonds + colony.reconnected_bonds
    assert abs(node.velocity[0]) < 150
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert_bounded(colony)
    assert len(node.connections) >= 2


def test_repair_continues_after_connectivity_returns_and_compacts_capture_lengths() -> None:
    colony = Colony(17, (600, 400), 24)
    node = isolate_tip(colony)
    node.position = (node.position[0] + colony.spacing * 0.8, node.position[1])
    for _ in range(240):
        colony.advance(1 / 120, (0, 0), (1200, 800))
        if len(colony.components) == 1:
            break
    assert len(colony.components) == 1
    first_repairs = colony.reconnected_bonds
    for _ in range(480):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert colony.reconnected_bonds > first_repairs
    assert len(node.connections) >= 2
    for bond in colony.bonds:
        if bond.created_at >= 0:
            a, b = colony.nodes[bond.first], colony.nodes[bond.second]
            assert hypot(a.position[0] - b.position[0], a.position[1] - b.position[1]) < colony.spacing * 1.35
    assert_bounded(colony)


def test_contact_cooldown_expires_in_simulation_time_and_healed_tissue_has_grace() -> None:
    colony = Colony(17, (600, 400), 24)
    node = isolate_tip(colony)
    for other in colony.components[0]:
        colony._cooldown[tuple(sorted((node.identifier, other)))] = 0.45
    colony._heal()
    assert len(colony.components) == 2
    for _ in range(120):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert len(colony.components) == 1
    colony.begin_grab(node.position)
    node.position = (node.position[0] + 50, node.position[1])
    colony.grab.applied_force = 14000
    colony.elapsed_seconds = max(bond.created_at for bond in colony.bonds) + 0.1
    broken = colony.broken_bonds
    for _ in range(20):
        colony._tear(1 / 120)
    assert colony.broken_bonds == broken


@pytest.mark.parametrize("seed,scale", [(0, 1.0), (17, 1.6), (42, 1.0)])
def test_repeated_tearing_preserves_nodes_mass_and_bond_accounting(seed, scale) -> None:
    colony = Colony(seed, (600, 400), 24)
    colony.set_growth(scale)
    count, mass, bonds = len(colony.nodes), sum(node.mass for node in colony.nodes), len(colony.bonds)
    identities = tuple(id(node) for node in colony.nodes)
    for cycle in range(3):
        direction = -1 if cycle % 2 else 1
        node = max(colony.nodes, key=lambda item: direction * item.position[0])
        origin = node.position
        colony.begin_grab(origin)
        for step in range(300):
            colony.move_grab((origin[0] + direction * (step + 1) * 1.3, origin[1]))
            colony.advance(1 / 120, (0, 0), (1200, 800))
        colony.release_grab()
        for _ in range(1440):
            colony.advance(1 / 120, (20, 5), (1200, 800))
            if len(colony.components) == 1:
                # Continue after first capture to exercise local mesh repair.
                for _ in range(180):
                    colony.advance(1 / 120, (20, 5), (1200, 800))
                break
        assert len(colony.components) == 1
        assert len(colony.nodes) == count and tuple(id(node) for node in colony.nodes) == identities
        assert sum(node.mass for node in colony.nodes) == mass
        assert len(colony.bonds) == bonds - colony.broken_bonds + colony.reconnected_bonds
        assert_bounded(colony)
    assert colony.broken_bonds > 0 and colony.reconnected_bonds > 0


def test_spatial_grid_does_not_miss_close_pairs_at_cell_edges_or_growth_scales() -> None:
    colony = Colony(42, (600, 400), 24)
    colony.nodes[0].position = (-1.0, -1.0)
    colony.nodes[1].position = (0.0, 0.0)
    for scale in (1.0, 1.6):
        colony.set_growth(scale)
        reach = max(node.radius for node in colony.nodes) * 1.84
        positions = [node.position for node in colony.nodes]
        candidates = nearby_pairs(positions, reach)
        close = {(a, b) for a in range(len(positions)) for b in range(a + 1, len(positions))
                 if hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]) < reach}
        assert close.issubset(set(candidates))
        assert candidates == sorted(set(candidates))
        assert len(candidates) < len(positions) * (len(positions) - 1) // 4
