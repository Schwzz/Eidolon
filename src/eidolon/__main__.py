"""Launch with ``python -m eidolon`` or the installed ``eidolon`` command."""

import argparse
from collections.abc import Sequence
import sys

import pygame

from eidolon.app import Application
from eidolon.config import AppConfig, PhysicsConfig


def main(argv: Sequence[str] | None = None) -> int:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="Open the EIDOLON simulation chamber.")
    parser.add_argument("--width", type=int, default=defaults.width, help="window width")
    parser.add_argument("--height", type=int, default=defaults.height, help="window height")
    parser.add_argument("--seed", type=int, default=None, help="world seed (default: saved brain seed, otherwise 17)")
    parser.add_argument("--body", choices=("simple", "detailed"), default=None,
                        help="physics topology (desktop default: detailed; training default: simple)")
    parser.add_argument("--show-nodes", action="store_true", help="overlay subtle physics-node markers")
    parser.add_argument("--brain", help="run a saved 30-input muscle brain in continuous survival simulation")
    parser.add_argument("--training-run", help="saved metrics JSON or run directory for the Training tab")
    parser.add_argument("--view", choices=("simulation", "neural", "training", "split"), default="simulation")
    parser.add_argument("--controller", choices=("rules", "neural"), default=defaults.controller,
                        help="decision controller (default: rules)")
    parser.add_argument("--network-seed", type=int, help="neural initialization seed (default: world seed)")
    parser.add_argument("--hidden-layers", type=int, nargs="+", metavar="WIDTH",
                        help="neural hidden layer widths (default: 16)")
    parser.add_argument("--load-network", metavar="PATH", help="load neural parameters from UTF-8 JSON")
    parser.add_argument("--save-network", metavar="PATH", help="save neural parameters to JSON before launching")
    parser.add_argument("--gravity", type=float, default=None,
                        help="downward world units/second squared (default: 180)")
    parser.add_argument("--floor-friction", type=float, default=None,
                        help="nonnegative floor friction coefficient (default: 0.55)")
    parser.add_argument("--no-propulsion", action="store_true", help="disable compatibility drive, homing, and automatic gait")
    parser.add_argument("--muscle-test", choices=("manual", "wave"), nargs="?", const="wave",
                        help="test muscles without propulsion: manual (hold 1-8, 0 relax) or wave (default)")
    parser.add_argument(
        "--max-fps", type=int, default=defaults.max_fps,
        help="render frame limit (default: 60; 0: uncapped)",
    )
    args = parser.parse_args(argv)
    if args.brain and (args.gravity is not None or args.floor_friction is not None):
        parser.error("muscle brains use the physics settings saved in the brain")
    if args.brain and (args.controller != "rules" or args.muscle_test is not None
                      or any(value is not None for value in (args.network_seed, args.hidden_layers, args.load_network, args.save_network))):
        parser.error("--brain is separate from steering controllers and manual muscle-test options")
    if args.controller != "neural" and any(option is not None for option in (
            args.network_seed, args.hidden_layers, args.load_network, args.save_network)):
        parser.error("network options require --controller neural")
    if args.load_network is not None and (args.network_seed is not None or args.hidden_layers is not None):
        parser.error("--load-network cannot be combined with --network-seed or --hidden-layers")
    try:
        brain_body = None
        brain = None
        if args.brain:
            from eidolon.muscle_controller import NeuralMuscleController
            brain = NeuralMuscleController.load(args.brain)
            brain_body = brain.episode.body
        config = AppConfig(width=args.width, height=args.height, max_fps=args.max_fps,
                           seed=args.seed if args.seed is not None else brain.episode.seed if brain is not None else defaults.seed,
                           controller=args.controller,
                           muscle_test=args.muscle_test,
                           body=args.body or brain_body or defaults.body, show_nodes=args.show_nodes,
                           brain_path=args.brain, training_path=args.training_run, view=args.view,
                           physics=PhysicsConfig(gravity=defaults.physics.gravity if args.gravity is None else args.gravity,
                                                 floor_friction=defaults.physics.floor_friction if args.floor_friction is None else args.floor_friction,
                                                 propulsion_enabled=not (args.no_propulsion or args.muscle_test)))
        controller = None
        if args.controller == "neural":
            from eidolon.neural_controller import NeuralController
            from eidolon.neural_network import FeedForwardNetwork

            if args.load_network is not None:
                controller = NeuralController(FeedForwardNetwork.load(args.load_network))
            else:
                seed = config.seed if args.network_seed is None else args.network_seed
                hidden = (16,) if args.hidden_layers is None else tuple(args.hidden_layers)
                controller = NeuralController.from_seed(seed, hidden)
            if args.save_network is not None:
                controller.network.save(args.save_network)
        application = Application(config, controller=controller, brain=brain)
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        parser.error(f"could not read/write network parameters: {error}")
    try:
        application.run()
    except KeyboardInterrupt:
        return 130
    except (pygame.error, ValueError) as error:
        print(f"EIDOLON could not run: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
