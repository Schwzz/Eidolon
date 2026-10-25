"""Independent 30-input/eight-output neural muscle policy and safe brain files."""

import numpy as np

from eidolon.experiment_io import episode_parameters, read_json, restore_episode, write_json
from eidolon.neural_network import FeedForwardNetwork
from eidolon.training import EpisodeConfig, OBSERVATION_NAMES


class NeuralMuscleController:
    name = "neural muscles"

    def __init__(self, network: FeedForwardNetwork, episode: EpisodeConfig | None = None, training_seeds=()):
        if (not isinstance(network, FeedForwardNetwork) or network.layers[0] != 30 or network.layers[-1] != 8
                or len(network.layers) > 4 or any(width > 32 for width in network.layers[1:-1])):
            raise ValueError("muscle network requires 30 inputs, 8 outputs, and at most two hidden layers of 1–32 units")
        if (not isinstance(training_seeds, (list, tuple)) or len(training_seeds) > 32
                or any(type(seed) is not int for seed in training_seeds)
                or len(set(training_seeds)) != len(training_seeds)):
            raise ValueError("training seeds must be unique integers (at most 32)")
        self.network = network
        self.episode = episode if episode is not None else EpisodeConfig()
        self.training_seeds = tuple(training_seeds)
        self.last_activations = None
        self.last_actions = None

    @classmethod
    def from_seed(cls, seed=0, hidden_layers=(16,), episode=None):
        return cls(FeedForwardNetwork((30, *hidden_layers, 8), seed), episode)

    def act(self, observations):
        outputs = self.network.forward(observations)
        self.last_activations = self.network.last_activations
        self.last_actions = tuple(float(value) for value in np.clip((outputs + 1) * 0.5, 0, 1))
        return self.last_actions

    def clear_activity(self):
        self.last_activations = self.last_actions = None
        self.network.last_activations = None

    def save(self, path):
        write_json(path, dict(format="eidolon.muscle-brain", version=1, observation_names=list(OBSERVATION_NAMES),
                              episode=episode_parameters(self.episode), training_seeds=list(self.training_seeds),
                              network=self.network.to_parameters()))

    @classmethod
    def load(cls, path):
        data = read_json(path, 8 * 1024 * 1024)
        keys = {"format", "version", "observation_names", "episode", "training_seeds", "network"}
        if (not isinstance(data, dict) or set(data) != keys or data["format"] != "eidolon.muscle-brain"
                or type(data["version"]) is not int or data["version"] != 1
                or data["observation_names"] != list(OBSERVATION_NAMES)):
            raise ValueError("invalid muscle brain format/version/observation schema")
        return cls(FeedForwardNetwork.from_parameters(data["network"]), restore_episode(data["episode"]), data["training_seeds"])
