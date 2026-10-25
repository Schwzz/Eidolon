from dataclasses import asdict
from pathlib import Path
import threading

import pygame
import pytest

from eidolon.app import Application
from eidolon.config import AppConfig
from eidolon.evolution import EvolutionConfig, EvolutionTrainer
from eidolon.experiment_io import read_json, write_json
from eidolon.laboratory import LaboratoryRenderer, neural_activity
from eidolon.live_learning import LiveLearning
from eidolon.rendering import ChamberRenderer
from eidolon.run_store import RunLock, SnapshotWriter
from eidolon.training import EpisodeConfig


def config():
    return EvolutionConfig(population=3, elites=1, seeds=(0, 17), hidden_layers=(4,),
                           episode=EpisodeConfig(duration=0.1, settle_seconds=0.1))


def advance_to(session, condition):
    for _ in range(3000):
        session.update(0.1)
        if condition():
            return
    pytest.fail("live evaluation did not reach the expected boundary")


def test_real_live_generations_match_headless_training_and_mutate_weights(tmp_path, monkeypatch):
    reference = EvolutionTrainer(config())
    for _ in range(2):
        reference.generation()
    live = LiveLearning(tmp_path, config(), budget_ms=100)
    initial = [network.to_parameters() for network in live.trainer.population]
    def forbidden(*args, **kwargs):
        pytest.fail("live training used a full-population evaluator or artificial propulsion")
    monkeypatch.setattr(EvolutionTrainer, "generation", forbidden)
    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    try:
        advance_to(live, lambda: len(live.trainer.history) == 2)
        assert live.trainer.history == reference.history
        assert [n.to_parameters() for n in live.trainer.population] == [n.to_parameters() for n in reference.population]
        assert live.trainer.rng.bit_generator.state == reference.rng.bit_generator.state
        assert initial != [n.to_parameters() for n in live.trainer.population]
        assert len(live.colony.nodes) == 10 and live.world.foods == []
        assert live.life.alive and all(c["energy"] > 0 for r in live.trainer.history for c in r["candidates"])
    finally:
        live.close()


def test_completed_attempts_survive_resume_and_partial_attempt_is_repeated(tmp_path):
    reference = EvolutionTrainer(config())
    reference.generation()
    live = LiveLearning(tmp_path, config(), budget_ms=100)
    advance_to(live, lambda: len(live.seed_reports) == 1)
    assert live.active_seed == 0
    live.update(0.1)  # Begin the unfinished second-seed attempt.
    live.close()
    restored = LiveLearning(tmp_path, budget_ms=100)
    try:
        assert restored.active_seed == 17 and len(restored.seed_reports) == 1
        assert restored.environment.elapsed_seconds == 0 and restored.environment.settling
        advance_to(restored, lambda: len(restored.trainer.history) == 1)
        assert restored.trainer.history == reference.history
        restored.save_progress(wait=True)
        assert EvolutionTrainer.load(tmp_path / "checkpoint.json").history == reference.history
    finally:
        restored.close()


def test_headless_continuation_updates_authoritative_live_checkpoint(tmp_path):
    from eidolon.evolve import main
    live = LiveLearning(tmp_path, config(), budget_ms=100)
    try:
        with pytest.raises(SystemExit):
            main(["resume", "--checkpoint", str(tmp_path / "checkpoint.json"), "--generations", "1"])
        advance_to(live, lambda: len(live.trainer.history) == 1)
    finally:
        live.close()
    assert main(["resume", "--checkpoint", str(tmp_path / "checkpoint.json"), "--generations", "1"]) == 0
    reference = EvolutionTrainer.load(tmp_path / "checkpoint.json")
    assert len(reference.history) == 2
    restored = LiveLearning(tmp_path)
    try:
        assert restored.trainer.checkpoint_data() == reference.checkpoint_data()
        assert restored.candidates == restored.seed_reports == []
    finally:
        restored.close()


def test_pause_stop_retry_and_new_run_preserve_or_archive_progress(tmp_path):
    live = LiveLearning(tmp_path, config(), budget_ms=100)
    try:
        advance_to(live, lambda: len(live.seed_reports) == 1)
        population = [n.to_parameters() for n in live.trainer.population]
        live.paused = True
        ticks = live.total_ticks
        live.update(1)
        assert live.total_ticks == ticks
        live.paused = False
        live.toggle_stopped()
        live.update(1)
        assert live.stopped and live.total_ticks == ticks
        live.toggle_stopped()
        live.reset()
        assert len(live.seed_reports) == 1 and [n.to_parameters() for n in live.trainer.population] == population
        assert live.environment.elapsed_seconds == 0
        live.restart_learning()
        assert live.candidates == live.seed_reports == live.trainer.history == []
        assert (tmp_path / "archive-001" / "live.json").exists()
        assert len(read_json(tmp_path / "archive-001" / "live.json")["seed_reports"]) == 1
    finally:
        live.close()


def test_frame_tick_budget_bounds_work_and_no_background_evaluations(tmp_path, monkeypatch):
    live = LiveLearning(tmp_path, config(), budget_ms=100, max_ticks=2)
    try:
        live.update(1)
        assert live.total_ticks == 2
        ticks = live.total_ticks
        times = iter([0, 1])
        monkeypatch.setattr("eidolon.live_learning.perf_counter", lambda: next(times))
        live.update(0.1)
        assert live.total_ticks == ticks
    finally:
        live.close()


def test_single_writer_lock_and_corrupt_snapshot_fail_without_overwriting(tmp_path):
    live = LiveLearning(tmp_path, config())
    try:
        with pytest.raises(ValueError, match="already open"):
            LiveLearning(tmp_path, config())
        live.save_progress(wait=True)
    finally:
        live.close()
    before = (tmp_path / "live.json").read_bytes()
    with pytest.raises(ValueError):
        write_json(tmp_path / "live.json", {"bad": float("nan")})
    assert (tmp_path / "live.json").read_bytes() == before
    data = read_json(tmp_path / "live.json")
    data["seed_reports"] = [{"seed": 99}]
    write_json(tmp_path / "live.json", data)
    with pytest.raises(ValueError):
        LiveLearning(tmp_path)
    with RunLock(tmp_path):
        pass  # Failed loads also release their lock.


def test_save_failure_is_visible_stops_training_and_releases_resources(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr("eidolon.run_store.save_snapshot", fail)
    live = LiveLearning(tmp_path, config())
    for _ in range(1000):
        live.poll_save()
        if live.save_error:
            break
        threading.Event().wait(0.001)
    assert live.stopped and "disk unavailable" in live.save_error
    assert "SAVE ERROR" in live.status
    with pytest.raises(OSError):
        live.close()
    with RunLock(tmp_path):
        pass


def test_snapshot_writer_coalesces_to_one_pending_save(tmp_path, monkeypatch):
    entered, release, saved = threading.Event(), threading.Event(), []
    def record(directory, data):
        saved.append(data)
        if data == 1:
            entered.set()
            release.wait(2)
    monkeypatch.setattr("eidolon.run_store.save_snapshot", record)
    writer = SnapshotWriter(tmp_path)
    try:
        writer.submit(1)
        assert entered.wait(1)
        writer.submit(2)
        writer.submit(3)
        release.set()
        writer.flush()
        assert saved == [1, 3]
    finally:
        release.set()
        writer.close()


def test_default_mode_and_window_mode_switches_preserve_current_attempt(tmp_path):
    app = Application(AppConfig(live_run=str(tmp_path)))
    try:
        assert app.mode == "live" and app.simulation.is_live_learning
        assert len(app.simulation.colony.nodes) == 10
        app.simulation.update(0.1)
        before = [asdict(n) for n in app.simulation.colony.nodes], app.simulation.total_ticks
        for view in ("neural", "training", "split", "simulation"):
            app._switch_view(view)
            assert ([asdict(n) for n in app.simulation.colony.nodes], app.simulation.total_ticks) == before
        session = app.simulation
        app._switch_mode("survival")
        assert not getattr(app.simulation, "is_live_learning", False)
        assert app.simulation.life.alive and len(app.simulation.world.foods) == 32
        app._switch_mode("live")
        assert app.simulation is session and app.simulation.total_ticks == before[1]
    finally:
        app.close()


def test_actual_candidate_and_muscle_indicators_visible_in_simulation_and_split(tmp_path, monkeypatch):
    live = LiveLearning(tmp_path, config(), budget_ms=100)
    pygame.font.init()
    try:
        for _ in range(100):
            live.update(1 / 120)
            if not live.environment.settling and live.environment._ticks == 1:
                break
        network, trace, status = neural_activity(live)
        assert network is live.trainer.population[live.active_candidate]
        assert trace is network.last_activations and trace is not None
        actual = tuple(live.colony.muscles.activations)
        assert any(a != t for a, t in zip(actual, live.colony.muscles.targets))
        def forbidden(*args):
            pytest.fail("drawing ran training or inference")
        monkeypatch.setattr(network, "forward", forbidden)
        chamber = ChamberRenderer()
        renderer = LaboratoryRenderer(chamber)
        for view in ("simulation", "split", "training", "neural"):
            renderer.view = view
            renderer.draw(pygame.Surface((1280, 720)), live, 1)
            assert tuple(live.colony.muscles.activations) == actual
    finally:
        live.close()
        pygame.quit()


def test_muscle_readouts_use_physical_activation_and_show_contraction_and_relaxation():
    from eidolon.simulation import Simulation
    pygame.font.init()
    try:
        simulation = Simulation(body="simple")
        simulation.colony.muscles.activations = (0.25, 0.75, 0, 0, 0, 0, 0, 0)
        simulation.colony.muscles.set_targets((0.75, 0.25, 0, 0, 0, 0, 0, 0))
        texts = []
        class ReadoutFont:
            def render(self, text, *args):
                texts.append(text)
                return pygame.Surface((20, 12))
        chamber = ChamberRenderer()
        chamber._muscle_font = ReadoutFont()
        renderer = LaboratoryRenderer(chamber)
        for view in ("simulation", "split"):
            texts.clear()
            renderer.view = view
            renderer.draw(pygame.Surface((1280, 720)), simulation, 1)
            assert texts[:2] == ["1+25", "2-75"]
            assert len([text for text in texts if text[0].isdigit()]) == 8
    finally:
        pygame.quit()


def test_plain_cli_launch_selects_live_learning_and_closes_its_writer(tmp_path, monkeypatch):
    from eidolon.__main__ import main
    apps = []
    monkeypatch.setattr(Application, "run", lambda app: apps.append(app))
    assert main(["--live-run", str(tmp_path)]) == 0
    assert apps[0].mode == "live" and len(apps[0].simulation.colony.nodes) == 10
    assert apps[0].simulation._closed
    restored = LiveLearning(tmp_path)
    restored.close()
