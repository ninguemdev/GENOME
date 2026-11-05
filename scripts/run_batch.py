"""Run one recorded world per seed, in parallel processes (ARQUITETURA §11.7, §12.5, ADR 0011).

Usage:
    ``uv run python scripts/run_batch.py --output runs/b --seeds 1-20``
    ``uv run python scripts/run_batch.py --output runs/b --seeds 1-20 --resume``

The exit code is zero only if every seed reached its end.
"""

import argparse
import sys
from pathlib import Path
from typing import cast, get_args

from genome.core.config import load_config
from genome.lab.batch import BatchResult, parse_seeds, run_batch
from genome.lab.ecology import ControllerName

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PRESET = REPOSITORY / "configs" / "f1_ecology.yaml"


def seeds_argument(text: str) -> list[int]:
    """Turn a refused list of seeds into a usage error, instead of a traceback."""
    try:
        return parse_seeds(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def report(result: BatchResult) -> None:
    """Print how the battery went, and what stopped each seed that did not end."""
    done = len(result.entries) - len(result.unfinished)
    print(f"{done} of {len(result.entries)} seeds ended; {result.workers} workers")
    for entry in result.unfinished:
        print(f"seed {entry.seed}: {entry.status}", file=sys.stderr)
        if entry.error is not None:
            print(entry.error, file=sys.stderr)


def main() -> int:
    """Run the battery and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=seeds_argument, required=True)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--controller", default="heuristic", choices=get_args(ControllerName))
    parser.add_argument("--workers", type=int, default=None, help="by default, the ceiling")
    parser.add_argument("--resume", action="store_true", help="continue a battery cut short")
    args = parser.parse_args()
    result = run_batch(
        load_config(args.preset),
        args.seeds,
        cast(ControllerName, args.controller),
        args.output,
        REPOSITORY,
        args.workers,
        args.resume,
    )
    report(result)
    return 1 if result.unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
