from dataclasses import replace
import json

import numpy as np
import pytest

from eidolon.evolution import EvolutionConfig, EvolutionTrainer, load_metrics, mutate, select_elites
from eidolon.evolve import compare, main
from eidolon.experiment_io import read_json, write_json
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.neural_network import FeedForwardNetwork
from eidolon.training import EpisodeConfig, LocomotionEnvironment


def tiny_config():
    return EvolutionConfig(population=4, elites=1, seeds=(0, 17), hidden_layers=(4,),
                           episode=EpisodeConfig(duration=0.2, settle_seconds=0.1))


def test_muscle_policy_dimensions_ranges_trace_and_parameter_immutability():
    controller = NeuralMuscleController.from_seed(17)
    parameters = controller.network.to_parameters()
    observation = LocomotionEnvironment(EpisodeConfig(duration=0.1)).observe()
    actions = controller.act(observation)
    assert controller.network.layers == (30, 16, 8)
    assert len(actions) == 8 and all(0 <= action <= 1 for action in actions)
    trace = controller.last_activations
    np.testing.assert_array_equal(trace[0], observation)
    for layer, (weight, bias) in enumerate(zip(controller.network.weights, controller.network.biases)):
        np.testing.assert_array_equal(trace[layer + 1], np.tanh(weight @ trace[layer] + bias))
    np.testing.assert_array_equal(actions, (trace[-1] + 1) / 2)
    assert all(not values.flags.writeable for values in trace)
    assert controller.network.to_parameters() == parameters
    controller.clear_activity()
    assert controller.last_activations is controller.network.last_activations is None


@pytest.mark.parametrize("layers", [(12, 16, 2), (30, 16, 2), (12, 8), (30, 33, 8), (30, 4, 4, 4, 8)])
def test_incompatible_muscle_networks_are_rejected(layers):
    with pytest.raises(ValueError):
        NeuralMuscleController(FeedForwardNetwork(layers))


def test_brain_roundtrip_validates_schema_and_preserves_legacy_network_files(tmp_path):
    controller = NeuralMuscleController.from_seed(17, (8,), tiny_config().episode)
    controller.training_seeds = (0, 17)
    path = tmp_path / "brain.json"
    controller.save(path)
    loaded = NeuralMuscleController.load(path)
    assert loaded.network.to_parameters() == controller.network.to_parameters()
    assert loaded.episode == controller.episode and loaded.training_seeds == (0, 17)
    for mutation in (lambda d: d.update(version=True), lambda d: d.update(observation_names=["wrong"] * 30),
                     lambda d: d.update(training_seeds=[0, 0]), lambda d: d.update(extra=1),
                     lambda d: d["network"].update(weights=[])):
        data = read_json(path)
        mutation(data)
        other = tmp_path / "invalid.json"
        write_json(other, data)
        with pytest.raises(ValueError):
            NeuralMuscleController.load(other)
    legacy = FeedForwardNetwork((12, 16, 2), 17)
    legacy.save(tmp_path / "legacy.json")
    assert FeedForwardNetwork.load(tmp_path / "legacy.json").to_parameters() == legacy.to_parameters()
    with pytest.raises(ValueError):
        NeuralMuscleController.load(tmp_path / "legacy.json")


def test_mutation_is_deterministic_bounded_and_does_not_modify_parent():
    parent = FeedForwardNetwork((30, 4, 8), 17)
    parameters = parent.to_parameters()
    a = mutate(parent, np.random.default_rng(123), 0.3, 1)
    b = mutate(parent, np.random.default_rng(123), 0.3, 1)
    assert a.to_parameters() == b.to_parameters() != parameters
    assert parent.to_parameters() == parameters
    assert all(np.isfinite(v).all() and abs(v).max() <= 5 for v in (*a.weights, *a.biases))
    assert select_elites([1, 3, 3, 0], 2) == (1, 2)


def test_deterministic_training_elitism_checkpoint_and_exact_resume(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("neural training used compatibility propulsion")

    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    full, split = EvolutionTrainer(tiny_config()), EvolutionTrainer(tiny_config())
    for _ in range(4):
        full.generation()
    for _ in range(2):
        split.generation()
    split.save(tmp_path)
    metrics = load_metrics(tmp_path / "metrics.json")
    assert metrics["completed_generations"] == 2
    resumed = EvolutionTrainer.load(tmp_path / "checkpoint.json")
    for _ in range(2):
        resumed.generation()
    assert resumed.history == full.history
    assert resumed.best.to_parameters() == full.best.to_parameters()
    assert resumed.initial_best.to_parameters() == full.initial_best.to_parameters()
    assert [n.to_parameters() for n in resumed.population] == [n.to_parameters() for n in full.population]
    assert resumed.rng.bit_generator.state == full.rng.bit_generator.state
    assert [row["best"] for row in full.history] == sorted(row["best"] for row in full.history)
    assert all(candidate["stable"] == 1 for row in full.history for candidate in row["candidates"])


def test_each_candidate_is_evaluated_on_all_training_seeds(monkeypatch):
    import eidolon.evolution as evolution

    seeds = []
    original = evolution.LocomotionEnvironment

    def record(config):
        seeds.append(config.seed)
        return original(config)

    monkeypatch.setattr(evolution, "LocomotionEnvironment", record)
    trainer = EvolutionTrainer(tiny_config())
    trainer.generation()
    assert seeds == [0, 17] * 4


@pytest.mark.parametrize("change", [dict(population=1), dict(elites=12), dict(seed=True), dict(seeds=(1,)),
    dict(seeds=(1, 1)), dict(mutation_std=float("nan")), dict(mutation_rate=0), dict(hidden_layers=(64,))])
def test_invalid_evolution_settings(change):
    with pytest.raises(ValueError):
        EvolutionConfig(**change)


def test_checkpoint_and_metric_corruption_fail_safely(tmp_path):
    trainer = EvolutionTrainer(tiny_config())
    trainer.generation()
    trainer.save(tmp_path)
    valid = read_json(tmp_path / "checkpoint.json")
    for change in (lambda d: d.update(population=[]), lambda d: d.update(rng={"bit_generator": "pickle"}),
                   lambda d: d.update(best=None), lambda d: d["history"][0].update(mean=999),
                   lambda d: d["config"].update(unknown=1)):
        data = json.loads(json.dumps(valid))
        change(data)
        write_json(tmp_path / "bad.json", data)
        with pytest.raises(ValueError):
            EvolutionTrainer.load(tmp_path / "bad.json")
    (tmp_path / "bad.json").write_text('{"x": NaN}')
    with pytest.raises(ValueError):
        read_json(tmp_path / "bad.json")


def test_comparison_labels_held_out_seeds_and_uses_same_episode_settings():
    controller = NeuralMuscleController.from_seed(0, (4,), tiny_config().episode)
    controller.training_seeds = (0, 17)
    report = compare(controller, [101, 102])
    assert report["held_out"]
    assert set(report["results"]) == {"trained", "idle", "random", "scripted"}
    for result in report["results"].values():
        assert [r["seed"] for r in result["episodes"]] == [101, 102]
        assert all(r["elapsed_seconds"] == 0.2 and not r["propulsion_enabled"] for r in result["episodes"])
    assert compare(controller, [0, 101])["held_out"] is False


def test_training_resume_evaluation_cli_and_overwrite_protection(tmp_path, capsys):
    directory = str(tmp_path / "run")
    assert main(["train", "--output", directory, "--generations", "1", "--population", "2", "--elites", "1",
                 "--seeds", "0", "17", "--duration", "0.1", "--hidden-layers", "4"]) == 0
    capsys.readouterr()
    assert main(["resume", "--checkpoint", directory + "/checkpoint.json", "--generations", "1"]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--brain", directory + "/best.json", "--seeds", "101", "102", "--compare"]) == 0
    assert json.loads(capsys.readouterr().out)["held_out"]
    assert load_metrics(directory + "/metrics.json")["completed_generations"] == 2
    with pytest.raises(SystemExit):
        main(["train", "--output", directory])
