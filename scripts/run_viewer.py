"""Watch one world with a fixed controller (ARQUITETURA §2.1).

Keys: space pauses, ``.`` runs one tick, ``+`` and ``-`` change the ticks per frame,
a click selects a creature, Esc clears the selection, ``q`` quits.

Usage: ``uv run python scripts/run_viewer.py --seed 1 --controller heuristic``
"""

import argparse
from pathlib import Path
from typing import cast, get_args

from genome.core.config import load_config
from genome.lab.ecology import ControllerName, make_simulation
from genome.lab.viewer import run_viewer
from genome.viz.renderer import WorldRenderer

DEFAULT_PRESET = Path(__file__).resolve().parents[1] / "configs" / "f1_ecology.yaml"


def main() -> None:
    """Open the window and run until the person quits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--controller", default="heuristic", choices=get_args(ControllerName))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.preset)
    simulation = make_simulation(config, args.seed, cast(ControllerName, args.controller))
    screen = WorldRenderer(config, "human")
    try:
        run_viewer(simulation, screen)
    finally:
        screen.close()


if __name__ == "__main__":
    main()
