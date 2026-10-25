"""Sequential elite selection and bounded Gaussian mutation of muscle brains."""

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from statistics import fmean, median

import numpy as np

from eidolon.experiment_io import episode_parameters, read_json, restore_episode, write_json
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.neural_network import FeedForwardNetwork
from eidolon.training import EpisodeConfig, LocomotionEnvironment


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    seed: int = 123
    population: int = 12
    elites: int = 3
    mutation_std: float = 0.15
    mutation_rate: float = 0.2
    hidden_layers: tuple[int, ...] = (16,)
    seeds: tuple[int, ...] = (0, 17, 42)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    def __post_init__(self):
        if type(self.seed) is not int or type(self.population) is not int or not 2 <= self.population <= 64:
            raise ValueError("seed must be integer; population must be 2–64")
        if type(self.elites) is not int or not 1 <= self.elites < self.population:
            raise ValueError("elites must be between 1 and population - 1")
        for name in ("mutation_std", "mutation_rate"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (not isinstance(self.hidden_layers, tuple) or len(self.hidden_layers) > 2
                or any(type(width) is not int or not 1 <= width <= 32 for width in self.hidden_layers)):
            raise ValueError("use at most two hidden layers of 1–32 units")
        if (not isinstance(self.seeds, tuple) or not 2 <= len(self.seeds) <= 32
                or any(type(seed) is not int for seed in self.seeds) or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("training requires 2–32 unique integer seeds")


def restore_config(data):
    if not isinstance(data, dict) or set(data) != set(EvolutionConfig.__dataclass_fields__):
        raise ValueError("invalid evolutionary configuration fields")
    try:
        values = dict(data)
        values["hidden_layers"], values["seeds"] = tuple(values["hidden_layers"]), tuple(values["seeds"])
        values["episode"] = restore_episode(values["episode"])
        return EvolutionConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid evolutionary configuration: {error}") from error


def evaluate_policy(controller, episode, seeds):
    reports = []
    for seed in seeds:
        environment = LocomotionEnvironment(replace(episode, seed=seed))
        controller.clear_activity()
        while not environment.done:
            environment.step(controller.act(environment.observe()))
        reports.append(environment.metrics())
    return dict(score=fmean(report["reward"] for report in reports),
                distance=fmean(report["distance"] for report in reports),
                energy=fmean(report["muscle_energy"] for report in reports),
                stable=fmean(report["stable"] for report in reports),
                success=fmean(report["reached_target"] for report in reports), episodes=reports)


def mutate(network, rng, std, rate):
    parameters = network.to_parameters()
    for name, arrays in (("weights", network.weights), ("biases", network.biases)):
        parameters[name] = [np.clip(array + rng.normal(0, std, array.shape)
                                     * (rng.random(array.shape) < rate), -5, 5) for array in arrays]
    return FeedForwardNetwork.from_parameters(parameters)


def select_elites(scores, count):
    return tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:count])


def validate_history(history, config):
    fields = {"generation", "best", "mean", "median", "worst", "distance", "energy", "stable", "candidates"}
    candidate_fields = {"index", "score", "distance", "energy", "stable", "success"}
    if not isinstance(history, list) or len(history) > 10000:
        raise ValueError("invalid generation history")
    for generation, row in enumerate(history, 1):
        if (not isinstance(row, dict) or set(row) != fields or type(row["generation"]) is not int
                or row["generation"] != generation or not isinstance(row["candidates"], list)
                or len(row["candidates"]) != config.population):
            raise ValueError("invalid generation metric fields/counts")
        if any(type(row[key]) not in (int, float) or not isfinite(row[key]) for key in fields - {"generation", "candidates"}):
            raise ValueError("generation metrics must be finite")
        for index, candidate in enumerate(row["candidates"]):
            if (not isinstance(candidate, dict) or set(candidate) != candidate_fields
                    or type(candidate["index"]) is not int or candidate["index"] != index
                    or any(type(candidate[key]) not in (int, float) or not isfinite(candidate[key])
                           for key in candidate_fields - {"index"})):
                raise ValueError("invalid candidate metrics")
        scores = [candidate["score"] for candidate in row["candidates"]]
        if (row["best"] != max(scores) or row["worst"] != min(scores) or row["mean"] != fmean(scores)
                or row["median"] != median(scores)):
            raise ValueError("inconsistent generation scores")
        champion = row["candidates"][select_elites(scores, 1)[0]]
        if (any(row[key] != champion[key] for key in ("distance", "energy", "stable"))
                or any(c["energy"] < 0 or not 0 <= c["stable"] <= 1 or not 0 <= c["success"] <= 1 for c in row["candidates"])
                or (generation > 1 and row["best"] < history[generation - 2]["best"])):
            raise ValueError("inconsistent champion or stability metrics")
    return history


def load_metrics(path):
    data = read_json(path, 16 * 1024 * 1024)
    if (not isinstance(data, dict) or set(data) != {"format", "version", "config", "completed_generations", "history"}
            or data["format"] != "eidolon.training-metrics" or type(data["version"]) is not int or data["version"] != 1):
        raise ValueError("invalid training metrics format")
    config = restore_config(data["config"])
    validate_history(data["history"], config)
    if type(data["completed_generations"]) is not int or data["completed_generations"] != len(data["history"]):
        raise ValueError("inconsistent completed generation count")
    return data


class EvolutionTrainer:
    def __init__(self, config=None):
        self.config = config if config is not None else EvolutionConfig()
        self.rng = np.random.default_rng(self.config.seed % (1 << 64))
        layers = (30, *self.config.hidden_layers, 8)
        self.population = [FeedForwardNetwork(layers, int(self.rng.integers(0, 1 << 63)))
                           for _ in range(self.config.population)]
        self.history = []
        self.best = self.initial_best = None

    def generation(self):
        if len(self.history) >= 10000:
            raise ValueError("generation history limit reached")
        candidates = []
        for index, network in enumerate(self.population):
            result = evaluate_policy(NeuralMuscleController(network), self.config.episode, self.config.seeds)
            candidates.append(dict(index=index, **{key: result[key] for key in ("score", "distance", "energy", "stable", "success")}))
        scores = [candidate["score"] for candidate in candidates]
        elite_indices = select_elites(scores, self.config.elites)
        self.best = self.population[elite_indices[0]]
        if self.initial_best is None:
            self.initial_best = self.best
        champion = candidates[elite_indices[0]]
        row = dict(generation=len(self.history) + 1, best=max(scores), mean=fmean(scores), median=median(scores),
                   worst=min(scores), distance=champion["distance"], energy=champion["energy"],
                   stable=champion["stable"], candidates=candidates)
        self.history.append(row)
        elites = [self.population[index] for index in elite_indices]
        self.population = elites + [mutate(elites[int(self.rng.integers(len(elites)))], self.rng,
                                           self.config.mutation_std, self.config.mutation_rate)
                                    for _ in range(self.config.population - len(elites))]
        return row

    def save(self, directory):
        directory = Path(directory)
        write_json(directory / "checkpoint.json", dict(format="eidolon.evolution-checkpoint", version=1,
            config=asdict(self.config), history=self.history, rng=self.rng.bit_generator.state,
            population=[network.to_parameters() for network in self.population],
            best=self.best.to_parameters() if self.best is not None else None,
            initial_best=self.initial_best.to_parameters() if self.initial_best is not None else None))
        if self.best is not None:
            NeuralMuscleController(self.best, self.config.episode, self.config.seeds).save(directory / "best.json")
            NeuralMuscleController(self.initial_best, self.config.episode, self.config.seeds).save(directory / "initial-best.json")
        write_json(directory / "metrics.json", dict(format="eidolon.training-metrics", version=1,
            config=asdict(self.config), completed_generations=len(self.history), history=self.history))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        fields = {"format", "version", "config", "history", "rng", "population", "best", "initial_best"}
        if (not isinstance(data, dict) or set(data) != fields or data["format"] != "eidolon.evolution-checkpoint"
                or type(data["version"]) is not int or data["version"] != 1):
            raise ValueError("invalid evolutionary checkpoint format")
        config = restore_config(data["config"])
        history = validate_history(data["history"], config)
        if not isinstance(data["population"], list) or len(data["population"]) != config.population:
            raise ValueError("checkpoint population count mismatch")
        trainer = cls.__new__(cls)
        trainer.config, trainer.history = config, history
        trainer.population = [FeedForwardNetwork.from_parameters(parameters) for parameters in data["population"]]
        trainer.best = FeedForwardNetwork.from_parameters(data["best"]) if data["best"] is not None else None
        trainer.initial_best = FeedForwardNetwork.from_parameters(data["initial_best"]) if data["initial_best"] is not None else None
        if (bool(history) != (trainer.best is not None and trainer.initial_best is not None)
                or (trainer.best is None) != (trainer.initial_best is None)):
            raise ValueError("checkpoint champion/history mismatch")
        for network in [*trainer.population, *([trainer.best, trainer.initial_best] if trainer.best is not None else [])]:
            if network.layers != (30, *config.hidden_layers, 8):
                raise ValueError("checkpoint network architecture mismatch")
        trainer.rng = np.random.default_rng()
        try:
            if not isinstance(data["rng"], dict) or data["rng"].get("bit_generator") != "PCG64":
                raise ValueError("checkpoint requires PCG64 random state")
            trainer.rng.bit_generator.state = data["rng"]
        except (TypeError, KeyError, OverflowError) as error:
            raise ValueError("invalid checkpoint random state") from error
        return trainer
