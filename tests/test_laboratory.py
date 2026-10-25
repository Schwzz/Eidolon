from dataclasses import asdict, replace

import numpy as np
import pygame
import pytest

from eidolon.app import Application
from eidolon.config import AppConfig
from eidolon.evolution import EvolutionTrainer
from eidolon.laboratory import LaboratoryRenderer, neural_activity
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.rendering import ChamberRenderer
from eidolon.replay import MuscleReplay
from eidolon.simulation import Simulation
from eidolon.training import EpisodeConfig, LocomotionEnvironment


def policy():
    return NeuralMuscleController.from_seed(123, (4,), EpisodeConfig(duration=0.5, settle_seconds=0.1))


@pytest.mark.parametrize("body", ["simple", "detailed"])
def test_replay_uses_identical_physics_actions_and_observations_as_headless_episode(monkeypatch, body):
    replay_policy = policy()
    replay_policy.episode = replace(replay_policy.episode, body=body)
    replay = MuscleReplay(replay_policy)
    controller = policy()
    controller.episode = replay_policy.episode
    environment = LocomotionEnvironment(controller.episode)
    while not environment.done:
        environment.step(controller.act(environment.observe()))
    def forbidden(*args, **kwargs):
        pytest.fail("replay used a translation/propulsion helper")
    monkeypatch.setattr(replay.colony, "translate_to", forbidden)
    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    for _ in range(30):
        replay.update(1 / 60)
    assert replay.environment.metrics() == environment.metrics()
    assert [(n.position, n.velocity) for n in replay.colony.nodes] == [(n.position, n.velocity) for n in environment.colony.nodes]
    assert replay.environment.observe() == environment.observe()
    np.testing.assert_array_equal(replay.controller.last_activations[-1], controller.last_activations[-1])


def test_replay_pause_reset_grab_resize_semantics():
    replay = MuscleReplay(policy())
    replay.update(1 / 60)
    replay.paused = True
    before = [asdict(n) for n in replay.colony.nodes], replay.environment.metrics()
    replay.update(1)
    replay.relax_muscles()
    assert ([asdict(n) for n in replay.colony.nodes], replay.environment.metrics()) == before
    replay.reset()
    assert replay.paused and replay.environment.elapsed_seconds == 0
    assert replay.controller.last_activations is None
    replay.paused = False
    node = replay.colony.nodes[0]
    assert replay.begin_grab(node.position) == node.identifier
    assert replay.intervened
    replay.move_grab((node.position[0] + 5, node.position[1]))
    replay.update(1 / 60)
    assert replay.environment.metrics()["stable"]
    replay.release_grab()


def test_activity_is_actual_inference_and_never_invented_when_inactive():
    simulation = Simulation()
    assert neural_activity(simulation) == (None, None, "No neural controller active")
    replay = MuscleReplay(policy())
    assert neural_activity(replay)[1] is None
    replay.update(1 / 60)
    network, trace, status = neural_activity(replay)
    assert network is replay.controller.network and trace is network.last_activations
    replay.paused = True
    assert "Paused" in neural_activity(replay)[2]
    replay.paused = False
    replay.update(1)
    assert neural_activity(replay)[1] is None and "completed" in neural_activity(replay)[2]
    replay.environment._stable = False
    assert neural_activity(replay)[1] is None and "unstable" in neural_activity(replay)[2]


@pytest.mark.parametrize("size", [(480, 320), (1280, 720), (1600, 900)])
@pytest.mark.parametrize("view", ["simulation", "neural", "training", "split"])
def test_tabs_draw_resize_without_changing_physics_or_running_inference(size, view, monkeypatch):
    pygame.font.init()
    try:
        replay = MuscleReplay(policy())
        replay.update(1 / 60)
        before = replay.environment.metrics(), [asdict(n) for n in replay.colony.nodes]
        def forbidden(*args, **kwargs):
            pytest.fail("drawing called inference or training")
        monkeypatch.setattr(replay.controller.network, "forward", forbidden)
        monkeypatch.setattr(replay.controller.network, "forward_with_activations", forbidden)
        monkeypatch.setattr(EvolutionTrainer, "generation", forbidden)
        renderer = LaboratoryRenderer(ChamberRenderer(), view=view)
        renderer.draw(pygame.Surface(size), replay, 0.5)
        assert (replay.environment.metrics(), [asdict(n) for n in replay.colony.nodes]) == before
    finally:
        pygame.quit()


def test_click_tabs_and_split_pointer_use_correct_viewport():
    app = Application(AppConfig(body="simple"))
    app.simulation.update(1 / 60)
    before = app.simulation.steps, app.simulation.elapsed_seconds, app.simulation.paused
    for view in ("neural", "training", "split", "simulation"):
        app._switch_view(view)
        assert (app.simulation.steps, app.simulation.elapsed_seconds, app.simulation.paused) == before
    app._switch_view("split")
    node = app.simulation.colony.nodes[0]
    camera = ChamberRenderer.camera_for_window((640, 720), (1200, 800))
    point = camera.world_to_screen(node.position)
    app._pointer_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point), (1280, 720))
    assert app.simulation.colony.grab.node == node.identifier
    app._switch_view("neural")
    assert app.simulation.colony.grab is None


def test_window_hotkeys_switch_views_without_pausing_or_resetting(monkeypatch):
    app = Application(AppConfig(max_fps=0))
    keys = [pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F1]
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=key)] for key in keys] + [[pygame.event.Event(pygame.QUIT)]])
    times = iter(i / 60 for i in range(5))
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr("eidolon.app.perf_counter", lambda: next(times))
    views = []
    original = LaboratoryRenderer.draw
    def draw(renderer, surface, simulation, alpha):
        original(renderer, surface, simulation, alpha)
        views.append((renderer.view, simulation.steps, simulation.paused))
    monkeypatch.setattr(LaboratoryRenderer, "draw", draw)
    app.run()
    assert views == [("neural", 1, False), ("training", 2, False), ("split", 3, False), ("simulation", 4, False)]
    assert not pygame.display.get_init()


def test_brain_cli_defaults_to_saved_body_and_validates_files_before_window(tmp_path, monkeypatch):
    from eidolon.__main__ import main
    path = tmp_path / "brain.json"
    policy().save(path)
    apps = []
    monkeypatch.setattr(Application, "run", lambda app: apps.append(app))
    assert main(["--brain", str(path), "--seed", "101", "--view", "split"]) == 0
    assert apps[0].simulation.is_replay and len(apps[0].simulation.colony.nodes) == 10
    assert apps[0].simulation.seed == 101 and apps[0].view == "split"
    saved = policy()
    saved.episode = replace(saved.episode, seed=53)
    saved.save(path)
    assert main(["--brain", str(path)]) == 0 and apps[-1].simulation.seed == 53
    for args in (["--brain", "missing.json"], ["--brain", str(path), "--controller", "neural"],
                 ["--brain", str(path), "--gravity", "100"], ["--training-run", "missing.json"]):
        with pytest.raises(SystemExit):
            main(args)


def test_training_tab_displays_validated_saved_generations_and_reload_does_not_train(tmp_path, monkeypatch):
    from eidolon.evolution import EvolutionConfig
    trainer = EvolutionTrainer(EvolutionConfig(population=2, elites=1, seeds=(0, 17), hidden_layers=(4,),
        episode=EpisodeConfig(duration=0.1, settle_seconds=0.1)))
    trainer.generation()
    trainer.save(tmp_path)
    app = Application(AppConfig(training_path=str(tmp_path), view="training", max_fps=0))
    assert app.saved_metrics["completed_generations"] == 1
    trainer.generation()
    trainer.save(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("UI trained a generation")
    monkeypatch.setattr(EvolutionTrainer, "generation", forbidden)
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F5)], [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    app.run()
    assert app.laboratory.metrics["completed_generations"] == 2
    assert app.simulation.steps >= 0 and not app.simulation.paused


def test_existing_steering_network_has_its_actual_two_output_snapshot():
    from eidolon.neural_controller import NeuralController
    simulation = Simulation(controller=NeuralController.from_seed(17))
    simulation.update(1 / 60)
    network, trace, status = neural_activity(simulation)
    assert [len(layer) for layer in trace] == [12, 16, 2]
    assert network is simulation.controller.network and "Actual" in status
