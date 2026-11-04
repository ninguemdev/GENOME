"""The state hash after 1000 ticks is the same in different processes (ARQUITETURA §2.5)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tests.determinism.hash_run"


def hash_in_a_new_process(seed: int, python_hash_seed: str, controller: str) -> str:
    """Different ``PYTHONHASHSEED`` values expose any dependence on the built-in ``hash``."""
    env = {**os.environ, "PYTHONHASHSEED": python_hash_seed, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", MODULE, str(seed), controller],
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.slow
@pytest.mark.parametrize("controller", ["heuristic", "random"])
def test_hash_is_equal_across_processes_and_differs_across_seeds(controller: str) -> None:
    """The random control also covers lateral and backward moves and the creatures stream."""
    first = hash_in_a_new_process(1, "1", controller)
    assert len(first) == 64
    assert first == hash_in_a_new_process(1, "2", controller)
    assert first != hash_in_a_new_process(2, "1", controller)
