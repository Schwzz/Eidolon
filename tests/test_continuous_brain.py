from dataclasses import asdict
from pathlib import Path

import pygame
import pytest

from eidolon.app import Application
from eidolon.config import AppConfig, LifeConfig, WorldConfig
from eidolon.laboratory import LaboratoryRenderer, neural_activity
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.simulation import Simulation
from eidolon.training import EpisodeConfig, LocomotionEnvironment
from eidolon.world import Food


EMPTY = WorldConfig(initial_food=0, regeneration_interval=0)


def brain():
    return NeuralMuscleController.from_seed(123, (4,), EpisodeConfig(duration=0.2, settle_seconds=0.1))


def physical_state(simulation):
    return (asdict(simulation.life), tuple(asdict(n) for n in simulation.colony.nodes),
            tuple(simulation.colony.bonds), tuple(simulation.world.foods), simulation.elapsed_seconds,
            simulation.world._random.getstate(), simulation.colony.muscles.targets)


@pytest.mark.parametrize("body", ["simple", "detailed"])
def test_brain_controls_continuously_after_training_duration_without_propulsion(body, monkeypatch):
    controller = brain()
    simulation = Simulation(body=body, world_config=EMPTY, muscle_controller=controller)
    calls = []
    act = controller.act

    def record(observation):
        calls.append((simulation.life.age_seconds, observation))
        return act(observation)

    def forbidden(*args, **kwargs):
        pytest.fail("continuous muscle simulation used compatibility drive or translation")

    monkeypatch.setattr(controller, "act", record)
    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    monkeypatch.setattr(simulation.colony, "translate_to", forbidden)
    for _ in range(120):
        simulation.update(1 / 60)
    assert simulation.life.alive and simulation.life.age_seconds == pytest.approx(2)
    assert simulation.life.age_seconds > controller.episode.duration
    assert len(calls) == 20 and calls[-1][0] > 1.8
    assert all(len(values) == 30 and values[11] == 1.0 for _, values in calls)
    assert simulation.life.muscle_energy_spent > 0
    assert simulation.life.energy < simulation.life_config.initial_energy
    assert not hasattr(simulation, "environment") and not simulation.physics_config.propulsion_enabled
    assert neural_activity(simulation)[1] is controller.network.last_activations


def test_published_brain_runs_past_its_twelve_second_episode_in_normal_application():
    path = Path(__file__).resolve().parents[1] / "experiments" / "phase2b2" / "best.json"
    app = Application(AppConfig(body="simple", brain_path=str(path)))
    simulation = app.simulation
    assert isinstance(simulation, Simulation) and len(simulation.world.foods) == 32
    for _ in range(840):
        simulation.update(1 / 60)
    assert simulation.life.alive and simulation.life.age_seconds == pytest.approx(14)
    assert simulation.controller.last_activations is not None
    assert "Actual" in neural_activity(simulation)[2]
    assert not getattr(simulation, "is_replay", False)


def test_pause_resize_and_restart_preserve_normal_brain_lifecycle_and_weights():
    controller = brain()
    simulation = Simulation(body="simple", muscle_controller=controller)
    parameters = controller.network.to_parameters()
    initial = physical_state(simulation)
    for _ in range(30):
        simulation.update(1 / 60)
    trajectory = physical_state(simulation)
    simulation.paused = True
    frozen = physical_state(simulation)
    simulation.update(3)
    simulation.relax_muscles()  # Application resize/focus cancellation.
    assert physical_state(simulation) == frozen
    simulation.reset()
    assert simulation.paused and physical_state(simulation) == initial
    assert controller.last_activations is None
    simulation.paused = False
    for _ in range(30):
        simulation.update(1 / 60)
    assert physical_state(simulation) == trajectory
    assert controller.network.to_parameters() == parameters
    node = simulation.colony.nodes[0]
    assert simulation.begin_grab(node.position) == node.identifier
    simulation.move_grab((node.position[0] + 5, node.position[1]))
    simulation.update(1 / 60)
    simulation.release_grab()


def test_normal_brain_retains_feeding_growth_regeneration_and_starvation():
    simulation = Simulation(body="simple", world_config=EMPTY, muscle_controller=brain())
    simulation.world.foods = [Food(123, simulation.life.position, 20, 5)]
    simulation.update(0.5)
    assert simulation.life.foods_eaten == 1 and simulation.life.body_scale > 1
    assert simulation.life.energy > simulation.life_config.initial_energy
    regenerating = Simulation(body="simple", muscle_controller=brain(),
                              world_config=WorldConfig(initial_food=0, regeneration_interval=0.5))
    regenerating.update(1)
    assert len(regenerating.world.foods) == 2
    starving = Simulation(body="simple", world_config=EMPTY, life_config=LifeConfig(initial_energy=0.05),
                          muscle_controller=brain())
    starving.update(1)
    assert not starving.life.alive and starving.life.energy == 0
    assert starving.colony.muscles.targets == (0,) * 8
    assert neural_activity(starving)[1] is None and "dead" in neural_activity(starving)[2]


def test_tabs_keep_brain_active_past_training_duration_without_pausing_or_resetting(tmp_path, monkeypatch):
    controller = brain()
    controller.save(tmp_path / "brain.json")
    app = Application(AppConfig(body="simple", brain_path=str(tmp_path / "brain.json"), max_fps=0))
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=key)]
                   for key in (pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F1)]
                  + [[pygame.event.Event(pygame.QUIT)]])
    times = iter(i / 10 for i in range(5))
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr("eidolon.app.perf_counter", lambda: next(times))
    snapshots = []
    draw = LaboratoryRenderer.draw

    def record(renderer, surface, simulation, alpha):
        before = physical_state(simulation)
        trace = simulation.controller.last_activations
        draw(renderer, surface, simulation, alpha)
        assert physical_state(simulation) == before
        snapshots.append((renderer.view, simulation.life.age_seconds, simulation.paused, trace))

    monkeypatch.setattr(LaboratoryRenderer, "draw", record)
    app.run()
    assert [row[0] for row in snapshots] == ["neural", "training", "split", "simulation"]
    assert [row[1] for row in snapshots] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert all(not row[2] and row[3] is not None for row in snapshots)
    assert len({id(row[3]) for row in snapshots}) == 4
    assert app.simulation.life.alive and not pygame.display.get_init()


def test_headless_training_still_stops_at_its_episode_duration():
    controller = brain()
    environment = LocomotionEnvironment(controller.episode)
    while not environment.done:
        result = environment.step(controller.act(environment.observe()))
    assert result.truncated and result.info["elapsed_seconds"] == 0.2
    with pytest.raises(RuntimeError):
        environment.step([0] * 8)
