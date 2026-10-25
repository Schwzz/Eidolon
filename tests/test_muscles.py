"""Material actuators, their physical effect, and episode/input integration."""

from dataclasses import asdict
from math import hypot, isfinite
import subprocess
import sys

import pygame
import pytest

from eidolon.app import Application
from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.config import AppConfig, LifeConfig, MuscleConfig, PhysicsConfig, WorldConfig
from eidolon.muscles import MuscleSystem, wave_activations
from eidolon.simulation import Simulation


PASSIVE = PhysicsConfig(propulsion_enabled=False)
EMPTY = WorldConfig(initial_food=0, regeneration_interval=0)


def state(simulation):
    colony = simulation.colony
    return (tuple(asdict(node) for node in colony.nodes), tuple(colony.bonds), asdict(simulation.life),
            colony.muscles.activations, colony.muscles.targets, simulation.elapsed_seconds,
            colony.broken_bonds, colony.reconnected_bonds)


@pytest.mark.parametrize("seed", [0, 17, 42, -3])
def test_eight_material_regions_are_deterministic_nonempty_and_stable(seed):
    first, second = Colony(seed, (600, 400), 24), Colony(seed, (300, 200), 24)
    assert first.muscles.node_regions == second.muscles.node_regions
    assert first._bond_regions == second._bond_regions
    assert set(first.muscles.node_regions) == set(range(8))
    assert set(first._bond_regions) == set(range(8))
    regions = first.muscles.node_regions, first._bond_regions
    first.set_growth(1.6)
    for node in first.nodes:
        node.position = (node.position[0] * 0.5, node.position[1] + node.identifier)
    first.rebuild_topology()
    assert (first.muscles.node_regions, first._bond_regions) == regions


def test_independent_activation_has_rate_length_and_energy_limits_and_relaxes():
    muscles = Colony(17, (600, 400), 24).muscles
    muscles.set_targets([1, 0, 0, 0, 0, 0, 0, 0])
    cost = muscles.advance(0.1)
    assert muscles.activations == pytest.approx((0.15, 0, 0, 0, 0, 0, 0, 0))
    assert cost > 0 and muscles.length_factors[0] < 1
    for _ in range(20):
        muscles.advance(0.1)
    assert muscles.activations == (1, 0, 0, 0, 0, 0, 0, 0)
    assert muscles.length_factors[0] == pytest.approx(1 - muscles.config.max_contraction)
    muscles.relax()
    for _ in range(6):
        muscles.advance(0.1)
    assert muscles.activations == (0,) * 8 and muscles.length_factors == (1,) * 8


@pytest.mark.parametrize("values", [None, 1, [], [1] * 7, [0] * 9, [True] * 8, ["1"] * 8,
    [float("nan")] * 8, [float("inf")] * 8, [-0.1] * 8, [1.01] * 8, [1j] * 8])
def test_invalid_actions_fail_without_mutating_muscles(values):
    muscles = Colony(17, (600, 400), 24).muscles
    with pytest.raises(ValueError, match="muscle targets"):
        muscles.set_targets(values)
    assert muscles.activations == muscles.targets == (0,) * 8


@pytest.mark.parametrize("values", [dict(max_contraction=-1), dict(max_contraction=0.36),
    dict(contraction_rate=0), dict(relaxation_rate=0), dict(holding_cost=float("nan")),
    dict(contraction_cost=-1), dict(contraction_rate=True), dict(holding_cost="1")])
def test_invalid_limits_are_rejected(values):
    with pytest.raises(ValueError):
        MuscleConfig(**values)


def test_unpowered_tissue_cannot_contract_and_activated_tissue_relaxes():
    muscles = Colony(17, (600, 400), 24).muscles
    muscles.set_targets([1] * 8)
    assert muscles.advance(0.1, 0) == 0 and muscles.activations == (0,) * 8
    muscles.advance(1)
    muscles.advance(0.1, 0)
    assert muscles.activations == pytest.approx((0.8,) * 8)
    muscles.advance(1, active=False)
    assert muscles.activations == (0,) * 8


def test_local_contraction_deforms_real_bonds_and_restores_relaxed_geometry():
    colony = Colony(17, (600, 400), 24, PhysicsConfig(gravity=0, propulsion_enabled=False))

    def mean_length(region):
        return sum(hypot(colony.nodes[b.first].position[0] - colony.nodes[b.second].position[0],
                         colony.nodes[b.first].position[1] - colony.nodes[b.second].position[1])
                   / b.rest_length for b, r in zip(colony.bonds, colony._bond_regions) if r == region) / colony._bond_regions.count(region)

    colony.muscles.set_targets([1, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert mean_length(0) < 0.85
    assert mean_length(3) > 0.95
    colony.muscles.relax()
    for _ in range(480):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    assert mean_length(0) == pytest.approx(1, abs=0.03)
    assert colony.broken_bonds == 0


def test_internal_contractions_cannot_propulse_airborne_material(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("muscle-only physics used compatibility drive or setup translation")

    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    colony = Colony(17, (600, 400), 24, PhysicsConfig(gravity=0, linear_damping=0, propulsion_enabled=False))
    monkeypatch.setattr(colony, "translate_to", forbidden)
    initial = colony.material_center
    for step in range(480):
        colony.muscles.set_targets(wave_activations(3 + step / 120))
        colony.advance(1 / 120, (1e6, -1e6), (1200, 800))
    assert colony.material_center == pytest.approx(initial, abs=1e-8)


@pytest.mark.parametrize("friction", [0.0, 0.55])
def test_wave_has_measurable_floor_displacement_only_with_traction(friction, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("muscle test called the baseline drive")

    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    colony = Colony(17, (600, 755), 24, PhysicsConfig(propulsion_enabled=False, floor_friction=friction))
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    settled_x = colony.material_center[0]
    for step in range(1920):
        colony.muscles.set_targets(wave_activations(3 + step / 120))
        colony.advance(1 / 120, (-1000, -1000), (1200, 800))
    colony.muscles.relax()
    for _ in range(480):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    dx = colony.material_center[0] - settled_x
    if friction:
        assert abs(dx) > 2.0
    else:
        assert abs(dx) < 1e-7
    assert colony.floor_contacts > 0 and len(colony.components) == 1
    assert colony.broken_bonds == 0


def test_muscle_energy_is_paid_from_biology_and_cannot_exceed_available_fuel():
    simulation = Simulation(world_config=EMPTY, life_config=LifeConfig(metabolism=0, movement_cost=0), muscle_test="manual")
    before = simulation.life.energy
    simulation.set_muscle_activations([1] * 8)
    simulation.update(1)
    spent = simulation.life.muscle_energy_spent
    assert spent > 0 and simulation.life.energy == pytest.approx(before - spent)
    simulation.life.energy = 0.01
    simulation.update(2)
    assert simulation.life.energy >= 0
    assert simulation.life.muscle_energy_spent - spent <= 0.01 + 1e-10
    assert max(simulation.colony.muscles.activations) < 0.1


def test_death_relaxes_muscles_and_stops_energy_charges_and_commands():
    simulation = Simulation(muscle_test="manual")
    simulation.set_muscle_activations([1] * 8)
    simulation.update(1)
    spent = simulation.life.muscle_energy_spent
    simulation.life.energy = 0
    simulation.update(2)
    simulation.set_muscle_activations([1] * 8)
    assert not simulation.life.alive
    assert simulation.colony.muscles.targets == simulation.colony.muscles.activations == (0,) * 8
    assert simulation.life.muscle_energy_spent == spent


def test_feeding_growth_and_starvation_remain_active_with_muscles():
    from eidolon.world import Food

    simulation = Simulation(world_config=EMPTY, muscle_test="manual")
    simulation.world.foods = [Food(123, simulation.life.position, 30, 5)]
    simulation.set_muscle_activations([0.6] * 8)
    simulation.update(2)
    assert simulation.life.foods_eaten == 1 and simulation.world.foods == []
    assert simulation.life.body_scale > 1 and simulation.life.muscle_energy_spent > 0
    assert simulation.life.alive
    starving = Simulation(world_config=EMPTY, life_config=LifeConfig(initial_energy=1), muscle_test="wave")
    starving.set_muscle_activations([1] * 8)
    starving.update(2)
    assert not starving.life.alive and starving.life.energy == 0


@pytest.mark.parametrize("mode", ["manual", "wave"])
def test_invalid_mode_is_rejected_and_actions_are_ignored_during_pause(mode):
    simulation = Simulation(muscle_test=mode)
    simulation.set_muscle_activations([1] * 8)
    simulation.paused = True
    frozen = state(simulation)
    simulation.set_muscle_activations([0.5] * 8)
    assert state(simulation) == frozen
    with pytest.raises(ValueError):
        Simulation(muscle_test="invalid")


@pytest.mark.parametrize("dt,budget", [(0, 1), (float("nan"), 1), (0.1, -1), (0.1, float("nan"))])
def test_invalid_muscle_step_is_rejected_without_state_change(dt, budget):
    muscles = Colony(17, (600, 400), 24).muscles
    with pytest.raises(ValueError):
        muscles.advance(dt, budget)
    assert muscles.activations == (0,) * 8


def test_muscle_episode_pause_reset_and_fixed_steps_reproduce_exactly():
    first, second = Simulation(muscle_test="wave"), Simulation(muscle_test="wave")
    first.update(5)
    trajectory = state(first)
    for _ in range(300):
        second.update(1 / 60)
    # Presentation history is captured once per caller update, not per substep.
    assert [(n.position, n.velocity) for n in first.colony.nodes] == [(n.position, n.velocity) for n in second.colony.nodes]
    assert first.colony.muscles.activations == second.colony.muscles.activations
    assert first.life.energy == second.life.energy
    assert first.elapsed_seconds == pytest.approx(second.elapsed_seconds)
    first.paused = True
    frozen = state(first)
    first.update(4)
    assert state(first) == frozen
    first.reset()
    assert first.paused and first.colony.muscles.activations == (0,) * 8
    first.paused = False
    first.update(5)
    assert state(first) == trajectory


def test_manual_keys_and_resize_do_not_change_physical_state(monkeypatch):
    app = Application(AppConfig(muscle_test="manual", max_fps=0))
    events = iter([
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_8)],
        [pygame.event.Event(pygame.KEYUP, key=pygame.K_1)],
        [pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600)],
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr("eidolon.app.perf_counter", lambda: 0)
    before = state(app.simulation)
    observed = []

    def observe(renderer, surface, simulation, alpha):
        observed.append(simulation.colony.muscles.targets)
        assert state(simulation)[:3] == before[:3]

    monkeypatch.setattr("eidolon.app.ChamberRenderer.draw", observe)
    app.run()
    assert observed == [(1, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0,) * 8]
    assert not app.simulation.physics_config.propulsion_enabled


@pytest.mark.parametrize("size", [(480, 320), (1280, 720), (1600, 900)])
def test_muscle_hud_draws_at_resized_sizes_without_mutating_physics(size):
    from eidolon.rendering import ChamberRenderer

    pygame.font.init()
    try:
        simulation = Simulation(muscle_test="manual")
        simulation.set_muscle_activations([0.5] * 8)
        simulation.update(1 / 60)
        before = state(simulation)
        renderer = ChamberRenderer()
        renderer.draw(pygame.Surface(size), simulation)
        assert ("PHASE 2B.1 | MUSCLE MANUAL", renderer.ACCENT) in renderer._labels
        assert state(simulation) == before
    finally:
        pygame.quit()


def test_rules_and_neural_network_are_preserved_and_muscle_interface_is_separate():
    from eidolon.neural_controller import NeuralController

    controller = NeuralController.from_seed(17)
    parameters = controller.network.to_parameters()
    simulation = Simulation(controller=controller, physics_config=PASSIVE)
    simulation.set_muscle_activations([0.3] * 8)
    simulation.update(0.2)
    simulation.reset()
    assert controller.network.layers == (12, 16, 2)
    assert controller.network.to_parameters() == parameters
    assert simulation.controller_name == "neural" and Simulation().controller_name == "rules"


def test_muscle_mode_is_headless_without_numpy_or_pygame():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['pygame'] = None
sys.modules['numpy'] = None
from eidolon.simulation import Simulation
s = Simulation(muscle_test='wave')
s.update(4)
assert s.life.muscle_energy_spent > 0 and not s.physics_config.propulsion_enabled
"""], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_fragmented_material_keeps_regions_and_recaptures_nearby_bonds_with_muscles():
    colony = Colony(17, (600, 755), 24, PASSIVE)
    regions = colony.muscles.node_regions
    identities = tuple(id(node) for node in colony.nodes)
    mass = sum(node.mass for node in colony.nodes)
    for _ in range(360):
        colony.advance(1 / 120, (0, 0), (1200, 800))
    for cycle in range(3):
        tip = max(colony.nodes, key=lambda n: n.position[0])
        removed = [bond for bond in colony.bonds if tip.identifier in (bond.first, bond.second)]
        colony.bonds = [bond for bond in colony.bonds if bond not in removed]
        colony.broken_bonds += len(removed)
        colony._scars.add(tip.identifier)
        colony.rebuild_topology()
        assert len(colony.components) == 2
        for step in range(360):
            colony.muscles.set_targets(wave_activations(3 + (cycle * 360 + step) / 120))
            colony.advance(1 / 120, (0, 0), (1200, 800))
        assert len(colony.components) == 1 and len(tip.connections) >= 2
        assert colony.muscles.node_regions == regions
        assert colony._bond_regions == tuple(colony.muscles.bond_region(b.first, b.second) for b in colony.bonds)
        for node in colony.nodes:
            assert all(isfinite(v) for v in (*node.position, *node.velocity))
            assert hypot(*node.velocity) <= PASSIVE.max_speed + 1e-6
            r = node.radius * MEMBRANE_RADIUS
            assert r <= node.position[0] <= 1200 - r and r <= node.position[1] <= 800 - r
    assert tuple(id(node) for node in colony.nodes) == identities
    assert sum(node.mass for node in colony.nodes) == mass


@pytest.mark.parametrize("mode", ["manual", "wave"])
def test_cli_muscle_modes_always_disable_propulsion(mode, monkeypatch):
    from eidolon.__main__ import main

    apps = []
    monkeypatch.setattr(Application, "run", lambda app: apps.append(app))
    assert main(["--muscle-test", mode]) == 0
    assert apps[0].simulation.muscle_test == mode
    assert not apps[0].config.physics.propulsion_enabled


def test_test_mode_skips_compatibility_controller_decisions():
    class ForbiddenController:
        def decide(self, senses):
            pytest.fail("muscle test should not run a steering controller")

    simulation = Simulation(controller=ForbiddenController(), muscle_test="wave")
    simulation.update(1 / 60)
