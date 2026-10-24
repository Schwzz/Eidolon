from dataclasses import asdict, replace
from math import atanh, pi

import numpy as np
import pygame
import pytest

from eidolon.app import Application
from eidolon.behavior import Senses, SurvivalRules
from eidolon.config import AppConfig, LifeConfig, WorldConfig
from eidolon.neural_controller import INPUT_NAMES, NeuralController, encode_senses
from eidolon.neural_network import FeedForwardNetwork
from eidolon.rendering import ChamberRenderer
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock
from eidolon.world import Food


EMPTY = WorldConfig(initial_food=0, regeneration_interval=0)


def senses(**changes):
    original = Senses((200, 300), pi / 2, 10, 0.75, 40, (1000, 800),
                      Food(1, (260, 380), 18, 5), speed=27.5, sensing_radius=200)
    return replace(original, **changes)


def constant_controller(steering=0.0, effort=0.0):
    parameters = FeedForwardNetwork((12, 2)).to_parameters()
    parameters["weights"] = [np.zeros((2, 12)).tolist()]
    parameters["biases"] = [[steering, effort]]
    return NeuralController(FeedForwardNetwork.from_parameters(parameters))


def snapshot(simulation):
    return (asdict(simulation.life), tuple(asdict(node) for node in simulation.colony.nodes),
            tuple(simulation.colony.bonds), simulation.colony.components, tuple(simulation.world.foods),
            simulation.world._random.getstate(), simulation.steps, simulation.elapsed_seconds)


def test_sensory_vector_matches_documented_normalization_and_order() -> None:
    values = encode_senses(senses())
    assert len(INPUT_NAMES) == 12
    assert values == pytest.approx([0.75, 0.6, 0.8, 0.5, 0, 1, 0.5, 0.2, 0, 0, 0, 1])
    assert values.dtype == np.float64 and np.isfinite(values).all()
    assert np.all(np.abs(values) <= 1)


def test_missing_food_zero_distance_and_disabled_movement_have_sensible_defaults() -> None:
    missing = encode_senses(senses(food=None, speed_reference=0))
    assert missing[[1, 2, 3, 6, 11]] == pytest.approx([0, 0, 1, 0, 0])
    coincident = encode_senses(senses(food=Food(1, (200, 300), 18, 5)))
    assert coincident[[1, 2, 3, 11]] == pytest.approx([0, 0, 0, 1])
    clipped = encode_senses(senses(energy_fraction=2, speed=1000, position=(0, 0)))
    assert clipped[[0, 6, 7, 9]] == pytest.approx([1, 1, 1, 1])


@pytest.mark.parametrize("changes", [dict(direction=float("nan")), dict(energy_fraction=float("inf")),
    dict(world_size=(0, 100)), dict(sensing_radius=0), dict(speed=-1), dict(speed_reference=-1),
    dict(clearance=-1), dict(food=Food(1, (float("nan"), 1), 18, 5))])
def test_invalid_sensory_numbers_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        encode_senses(senses(**changes))


def test_network_outputs_request_steering_and_effort_without_touching_physics() -> None:
    controller = constant_controller(atanh(0.5), atanh(-0.5))
    decision = controller.decide(senses())
    assert decision.direction == pytest.approx(pi)
    assert decision.speed_fraction == pytest.approx(0.25)
    assert constant_controller().decide(senses()).speed_fraction == 0.5
    for layers in ((11, 16, 2), (12, 3)):
        with pytest.raises(ValueError, match="12 inputs and 2 outputs"):
            NeuralController(FeedForwardNetwork(layers))


def test_rules_default_and_neural_application_selection_keep_identical_initial_worlds() -> None:
    rules = Application(AppConfig(seed=42))
    neural = Application(AppConfig(seed=42, controller="neural"))
    assert isinstance(rules.simulation.controller, SurvivalRules)
    assert isinstance(neural.simulation.controller, NeuralController)
    assert rules.simulation.controller_name == "rules" and neural.simulation.controller_name == "neural"
    assert snapshot(rules.simulation) == snapshot(neural.simulation)
    assert rules.simulation.organism.appearance == neural.simulation.organism.appearance
    with pytest.raises(ValueError, match="controller"):
        AppConfig(controller="unknown")


def test_inference_runs_once_per_simulation_substep_and_never_while_paused_dead_or_drawing(monkeypatch) -> None:
    controller = NeuralController.from_seed(17)
    calls = []
    decide = controller.decide

    def observe(value):
        calls.append(value)
        return decide(value)

    monkeypatch.setattr(controller, "decide", observe)
    simulation = Simulation(controller=controller, life_config=LifeConfig(movement_speed=30, sensing_radius=111))
    simulation.update(1 / 30)
    assert len(calls) == 2
    assert calls[-1].speed_reference == 30 and calls[-1].sensing_radius == 111
    assert calls[-1].speed > 0
    simulation.set_speed(2)
    simulation.update(1 / 60)
    assert len(calls) == 4
    simulation.paused = True
    simulation.update(1)
    pygame.font.init()
    renderer = ChamberRenderer()
    for size in ((480, 320), (1280, 720)):
        before = snapshot(simulation)
        renderer.draw(pygame.Surface(size), simulation)
        assert snapshot(simulation) == before
    assert ("PHASE 2A | NEURAL", renderer.ACCENT) in renderer._labels
    assert len(calls) == 4
    simulation.paused = False
    simulation.life.energy = 0
    simulation.update(1 / 60)
    assert len(calls) == 4 and not simulation.life.alive
    pygame.quit()


def test_neural_motion_feeding_growth_and_starvation_use_existing_survival_systems() -> None:
    simulation = Simulation(world_config=EMPTY, controller=constant_controller(effort=100))
    simulation.life.direction = 0
    simulation.life.energy = 50
    x, y = simulation.life.position
    simulation.world.foods = [Food(1, (x + 70, y), 18, 5)]
    simulation.update(2)
    assert simulation.life.position[0] > x + 20
    assert simulation.life.foods_eaten == 1 and simulation.world.foods == []
    assert 50 < simulation.life.energy < 68 and simulation.life.body_scale > 1
    starving = Simulation(world_config=EMPTY, controller=constant_controller(effort=-100),
                          life_config=LifeConfig(initial_energy=1, metabolism=4))
    starving.update(1)
    assert not starving.life.alive and starving.life.energy == 0
    assert all(node.velocity == (0, 0) for node in starving.colony.nodes)


def test_loaded_network_survives_restart_and_repeats_trajectory(tmp_path) -> None:
    path = tmp_path / "network.json"
    NeuralController.from_seed(42, (8, 4)).network.save(path)
    controller = NeuralController(FeedForwardNetwork.load(path))
    simulation = Simulation(17, controller=controller)
    initial = snapshot(simulation)
    for _ in range(90):
        simulation.update(1 / 60)
    trajectory = snapshot(simulation)
    parameters = controller.network.to_parameters()
    simulation.paused = True
    simulation.reset()
    assert snapshot(simulation) == initial
    assert simulation.controller is controller and controller.network.to_parameters() == parameters
    simulation.paused = False
    for _ in range(90):
        simulation.update(1 / 60)
    assert snapshot(simulation) == trajectory


def test_neural_fixed_step_trajectory_is_independent_of_render_framerate() -> None:
    results = []
    for fps in (30, 60, 144):
        simulation = Simulation(17, controller=NeuralController.from_seed(123), world_config=EMPTY)
        clock = FixedStepClock(AppConfig())
        for _ in range(fps * 2):
            clock.advance(1 / fps, simulation.update)
        results.append(snapshot(simulation))
    assert results[0] == results[1] == results[2]
