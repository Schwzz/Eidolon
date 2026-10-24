"""Display-free timing of the default feedforward network and sensory/controller path."""

import argparse
import json
import platform
from time import perf_counter

import numpy as np

from eidolon.behavior import Senses
from eidolon.neural_controller import NeuralController, encode_senses
from eidolon.world import Food


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    controller = NeuralController.from_seed(args.seed)
    senses = Senses((600, 400), 0.5, 10, 0.7, 144, (1200, 800), Food(0, (650, 440), 18, 5), speed=25)
    inputs = encode_senses(senses)
    for _ in range(200):
        controller.decide(senses)
    started = perf_counter()
    for _ in range(args.iterations):
        controller.network.forward(inputs)
    forward_us = (perf_counter() - started) * 1_000_000 / args.iterations
    started = perf_counter()
    for _ in range(args.iterations):
        controller.decide(senses)
    decision_us = (perf_counter() - started) * 1_000_000 / args.iterations
    print(json.dumps(dict(python=platform.python_version(), platform=platform.system(), numpy=np.__version__,
                          seed=args.seed, layers=list(controller.network.layers),
                          parameters=sum(array.size for array in (*controller.network.weights, *controller.network.biases)),
                          iterations=args.iterations, forward_mean_us=round(forward_us, 3),
                          decision_mean_us=round(decision_us, 3)), indent=2))


if __name__ == "__main__":
    main()
