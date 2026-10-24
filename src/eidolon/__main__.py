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
    parser.add_argument(
        "--max-fps", type=int, default=defaults.max_fps,
        help="render frame limit (default: 60; 0: uncapped)",
    )
    args = parser.parse_args(argv)
    try:
        config = AppConfig(width=args.width, height=args.height, max_fps=args.max_fps, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    try:
        Application(config).run()
    except KeyboardInterrupt:
        return 130
    except pygame.error as error:
        print(f"EIDOLON could not run: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
