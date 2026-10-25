from dataclasses import asdict
from math import hypot, isfinite

import pygame
import pytest

from eidolon.camera import Camera
from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.colony_rendering import ColonyRenderer
from eidolon.config import AppConfig, BodyConfig, LifeConfig, PhysicsConfig, WorldConfig
from eidolon.organism import generate_appearance
from eidolon.simulation import Simulation


@pytest.mark.parametrize("seed", [0, 1, 17, 42, -3, 999])
def test_simple_body_has_exactly_ten_deterministic_connected_nodes(seed):
    first, second = Colony(seed, (600, 400), 24, body="simple"), Colony(seed, (600, 400), 24, body="simple")
    assert len(first.nodes) == 10 and len(first.bonds) == 17
    assert len(first.components) == 1 and len(first.triangles) == 8
    assert [asdict(n) for n in first.nodes] == [asdict(n) for n in second.nodes]
    assert first.bonds == second.bonds
    assert set(first._bond_regions) == set(range(8))
    assert min(len(n.connections) for n in first.nodes) >= 2
    assert 85 <= len(Colony(seed, (600, 400), 24, body="detailed").nodes) <= 98


@pytest.mark.parametrize("region", range(8))
def test_each_simple_muscle_region_contracts_material_and_recovers(region):
    colony = Colony(17, (600, 400), 24, PhysicsConfig(gravity=0, propulsion_enabled=False), body="simple")
    original = [n.position for n in colony.nodes]
    actions = [0.0] * 8
    actions[region] = 1
    colony.muscles.set_targets(actions)
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    lengths = [hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                     colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1]) / b.rest_length
               for b, r in zip(colony.bonds, colony._bond_regions) if r == region]
    assert sum(lengths) / len(lengths) < 0.98
    assert max(hypot(n.position[0] - p[0], n.position[1] - p[1]) for n, p in zip(colony.nodes, original)) > 0.5
    assert colony.material_center == pytest.approx((600, 400), abs=1e-7)
    colony.muscles.relax()
    for _ in range(480):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    for bond in colony.bonds:
        a, b = colony.nodes[bond.first], colony.nodes[bond.second]
        assert hypot(a.position[0] - b.position[0], a.position[1] - b.position[1]) / bond.rest_length == pytest.approx(1, abs=0.03)


def test_simple_body_settles_and_normal_survival_remains_enabled():
    simulation = Simulation(body="simple", physics_config=PhysicsConfig(propulsion_enabled=False),
                            world_config=WorldConfig(initial_food=0, regeneration_interval=0))
    simulation.update(5)
    assert simulation.colony.floor_contacts > 0
    assert max(hypot(*n.velocity) for n in simulation.colony.nodes) < 0.2
    from eidolon.world import Food

    simulation.world.foods = [Food(123, simulation.life.position, 20, 5)]
    simulation.update(1)
    assert simulation.life.foods_eaten == 1 and simulation.life.body_scale > 1
    starving = Simulation(body="simple", life_config=LifeConfig(initial_energy=0.1),
                          world_config=WorldConfig(initial_food=0, regeneration_interval=0))
    starving.update(1)
    assert not starving.life.alive


def test_simple_body_gentle_pull_stretches_without_tearing():
    simulation = Simulation(body="simple", physics_config=PhysicsConfig(propulsion_enabled=False))
    simulation.update(3)
    node = max(simulation.colony.nodes, key=lambda n: n.position[0])
    origin = node.position
    simulation.begin_grab(origin)
    for tick in range(120):
        simulation.move_grab((origin[0] + (tick + 1) / 120 * 15, origin[1]))
        simulation.update(1 / 120)
    assert simulation.colony.broken_bonds == 0 and len(simulation.colony.components) == 1
    assert node.position[0] > origin[0] + 1
    simulation.release_grab()


def test_simple_nearby_fragments_reconnect_with_propulsion_disabled():
    colony = Colony(17, (600, 770), 24, PhysicsConfig(propulsion_enabled=False), body="simple")
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    for _ in range(3):
        tip = max(colony.nodes, key=lambda n: n.position[0])
        colony.bonds = [b for b in colony.bonds if tip.identifier not in (b.first, b.second)]
        colony._scars.add(tip.identifier)
        colony.rebuild_topology()
        assert len(colony.components) == 2
        for _ in range(240):
            colony.advance(1 / 120, (0, 0), (1200, 800))
        assert len(colony.nodes) == 10 and len(colony.components) == 1 and len(tip.connections) >= 2


@pytest.mark.parametrize("seed", [0, 17])
def test_simple_body_repeated_grabbing_breakage_and_reconnection(seed):
    colony = Colony(seed, (600, 770), 24, body="simple")
    identity, mass, bonds = tuple(id(n) for n in colony.nodes), sum(n.mass for n in colony.nodes), len(colony.bonds)
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    peak = 1
    for cycle in range(3):
        sign = 1 if cycle % 2 == 0 else -1
        node = max(colony.nodes, key=lambda n: sign * n.position[0])
        origin = node.position
        assert colony.begin_grab(origin) == node.identifier
        for tick in range(300):
            colony.move_grab((origin[0] + sign * (tick + 1) * 4, origin[1]))
            colony.advance(1 / 120, (0, 0), (1200, 800))
            peak = max(peak, len(colony.components))
        colony.release_grab()
        for tick in range(2160):
            colony.advance(1 / 120, (0, 0), (1200, 800))
            for n in colony.nodes:
                radius = n.radius * MEMBRANE_RADIUS
                assert all(isfinite(v) for v in (*n.position, *n.velocity))
                assert hypot(*n.velocity) <= 220 + 1e-6
                assert radius <= n.position[0] <= 1200 - radius
                assert radius <= n.position[1] <= 800 - radius
        assert len(colony.components) == 1
        assert tuple(id(n) for n in colony.nodes) == identity and sum(n.mass for n in colony.nodes) == mass
        assert len(colony.bonds) == bonds - colony.broken_bonds + colony.reconnected_bonds
    assert peak > 1 and colony.broken_bonds > 0 and colony.reconnected_bonds > 0


@pytest.mark.parametrize("size", [(480, 320), (1280, 720)])
def test_simple_membrane_is_cohesive_and_resizing_and_markers_are_render_only(size):
    from eidolon.rendering import ChamberRenderer

    pygame.font.init()
    try:
        simulation = Simulation(body="simple")
        before = [asdict(n) for n in simulation.colony.nodes]
        ChamberRenderer(show_nodes=True).draw(pygame.Surface(size), simulation)
        assert [asdict(n) for n in simulation.colony.nodes] == before
        surface = pygame.Surface((800, 600), pygame.SRCALPHA)
        colony = Colony(17, (400, 300), 24, body="simple")
        camera = Camera.fit((800, 600), (0, 0, 800, 600))
        ColonyRenderer().draw(surface, surface.get_rect(), colony, camera, generate_appearance(17))
        assert len([r for r in pygame.mask.from_surface(surface, 40).get_bounding_rects() if r.width * r.height > 16]) == 1
    finally:
        pygame.quit()


def test_body_selection_and_saved_neural_models_remain_compatible(tmp_path, monkeypatch):
    from eidolon.__main__ import main
    from eidolon.app import Application
    from eidolon.neural_controller import NeuralController

    path = tmp_path / "network.json"
    controller = NeuralController.from_seed(17)
    controller.network.save(path)
    apps = []
    monkeypatch.setattr(Application, "run", lambda app: apps.append(app))
    assert main(["--body", "simple", "--show-nodes", "--controller", "neural", "--load-network", str(path)]) == 0
    assert len(apps[0].simulation.colony.nodes) == 10 and apps[0].config.show_nodes
    assert apps[0].simulation.controller.network.to_parameters() == controller.network.to_parameters()
    apps[0].simulation.update(0.5)
    apps[0].simulation.reset()
    assert len(apps[0].simulation.colony.nodes) == 10
    assert AppConfig().body == BodyConfig().kind == "detailed"
    with pytest.raises(ValueError):
        Simulation(body="invalid")
