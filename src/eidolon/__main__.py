"""Launch with ``python -m eidolon`` or the installed ``eidolon`` command."""

import argparse
from collections.abc import Sequence
import sys

import pygame

from eidolon.app import Application
from eidolon.config import AppConfig


def main(argv: Sequence[str] | None = None) -> int:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="Open the EIDOLON simulation chamber.")
    parser.add_argument("--width", type=int, default=defaults.width, help="window width")
    parser.add_argument("--height", type=int, default=defaults.height, help="window height")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="world and appearance seed (default: 17)")
    parser.add_argument("--controller", choices=("rules", "neural"), default=defaults.controller,
                        help="decision controller (default: rules)")
    parser.add_argument("--network-seed", type=int, help="neural initialization seed (default: world seed)")
    parser.add_argument("--hidden-layers", type=int, nargs="+", metavar="WIDTH",
                        help="neural hidden layer widths (default: 16)")
    parser.add_argument("--load-network", metavar="PATH", help="load neural parameters from UTF-8 JSON")
    parser.add_argument("--save-network", metavar="PATH", help="save neural parameters to JSON before launching")
    parser.add_argument(
        "--max-fps", type=int, default=defaults.max_fps,
        help="render frame limit (default: 60; 0: uncapped)",
    )
    args = parser.parse_args(argv)
    if args.controller != "neural" and any(option is not None for option in (
            args.network_seed, args.hidden_layers, args.load_network, args.save_network)):
        parser.error("network options require --controller neural")
    if args.load_network is not None and (args.network_seed is not None or args.hidden_layers is not None):
        parser.error("--load-network cannot be combined with --network-seed or --hidden-layers")
    try:
        config = AppConfig(width=args.width, height=args.height, max_fps=args.max_fps,
                           seed=args.seed, controller=args.controller)
        controller = None
        if args.controller == "neural":
            from eidolon.neural_controller import NeuralController
            from eidolon.neural_network import FeedForwardNetwork

            if args.load_network is not None:
                controller = NeuralController(FeedForwardNetwork.load(args.load_network))
            else:
                seed = args.seed if args.network_seed is None else args.network_seed
                hidden = (16,) if args.hidden_layers is None else tuple(args.hidden_layers)
                controller = NeuralController.from_seed(seed, hidden)
            if args.save_network is not None:
                controller.network.save(args.save_network)
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        parser.error(f"could not read/write network parameters: {error}")
    try:
        Application(config, controller=controller).run()
    except KeyboardInterrupt:
        return 130
    except (pygame.error, ValueError) as error:
        print(f"EIDOLON could not run: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
