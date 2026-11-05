"""Run one world recording telemetry, checkpoints and a manifest (ARQUITETURA §11).

Usage:
    ``uv run python scripts/run_recorded.py --output runs/a --seed 1``
    ``uv run python scripts/run_recorded.py --output runs/a --resume``
"""

import argparse
import sys
from pathlib import Path
from typing import cast, get_args

from genome.core.config import load_config
from genome.lab.ecology import ControllerName
from genome.lab.recorded import RecordedRun, is_resumable, resume_recorded, run_recorded

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PRESET = REPOSITORY / "configs" / "f1_ecology.yaml"


def main() -> None:
    """Start or resume a run and print how it ended."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--controller", default="heuristic", choices=get_args(ControllerName))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = parser.parse_args()
    result: RecordedRun
    if args.resume:
        result = resume_recorded(args.output, REPOSITORY)
    else:
        config = load_config(args.preset)
        controller = cast(ControllerName, args.controller)
        result = run_recorded(config, args.seed, controller, args.output, REPOSITORY)
    print(f"outcome={result.outcome} ticks={result.ticks} hash={result.final_hash}")
    if result.outcome == "interrupted" and not is_resumable(args.output):
        print(
            f"warning: no checkpoint was written, so {args.output} cannot be resumed; "
            "delete it to start again, or use a smaller lab.t_ckpt",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
