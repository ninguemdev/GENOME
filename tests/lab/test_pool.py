"""Worlds in processes: one failure costs one world, and the ceiling holds (ADR 0011)."""

import os
import signal
import sys
import threading
from pathlib import Path

import pytest
from tests.lab import pool_targets as targets

from genome.lab.pool import (
    ABOVE_NORMAL_NICE,
    ABOVE_NORMAL_PRIORITY_CLASS,
    SINGLE_THREAD_VARIABLES,
    JobResult,
    ceiling_workers,
    run_pool,
    set_process_priority,
)

# A process spawned on Windows reports this class when nobody changed it.
NORMAL_PRIORITY_CLASS = 0x20
#: Larger than any pipe buffer of the supported systems.
LARGE_RESULT_BYTES = 2**24
#: Long enough for the first child to be running.
INTERRUPT_AFTER_SECONDS = 1.0


@pytest.mark.parametrize(
    ("cpus", "fraction", "expected"),
    [(16, 0.8, 12), (10, 0.8, 8), (16, 1.0, 16), (1, 0.8, 1), (4, 0.1, 1), (16, 0.81, 12)],
)
def test_ceiling_by_hand(cpus: int, fraction: float, expected: int) -> None:
    assert ceiling_workers(cpus, fraction) == expected


def test_normal_priority_changes_nothing() -> None:
    assert set_process_priority("normal") == "normal"


def test_results_come_in_the_order_of_the_jobs() -> None:
    results = run_pool([3, 1, 2], targets.double, 2, "normal")
    assert [result.value for result in results] == [6, 2, 4]
    assert {result.status for result in results} == {"ok"}


def test_no_jobs() -> None:
    assert run_pool([], targets.double, 2, "normal") == []


def test_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        run_pool([1], targets.double, 0, "normal")


def test_an_exception_costs_only_its_job() -> None:
    results = run_pool([1, 2, 3], targets.fail_on_two, 2, "normal")
    assert [result.status for result in results] == ["ok", "failed", "ok"]
    assert [results[0].value, results[2].value] == [1, 3]
    assert results[1].error is not None
    assert "job two fails" in results[1].error


def test_a_dead_process_costs_only_its_job() -> None:
    results = run_pool([1, 2, 3], targets.crash_on_two, 2, "normal")
    assert [result.status for result in results] == ["ok", "crashed", "ok"]
    assert results[1].error == f"exit code {targets.CRASH_CODE}"
    assert results[1].priority is None


def test_a_result_larger_than_the_pipe_does_not_block() -> None:
    result = run_pool([LARGE_RESULT_BYTES], targets.large_result, 1, "normal")[0]
    assert result.status == "ok"
    assert result.value is not None
    assert len(result.value) == LARGE_RESULT_BYTES


def test_concurrency_never_passes_the_workers(tmp_path: Path) -> None:
    jobs = [str(tmp_path / f"{index}.txt") for index in range(6)]
    results = run_pool(jobs, targets.hold_and_record, 2, "normal")
    intervals = [result.value for result in results if result.value is not None]
    assert len(intervals) == len(jobs)
    edges = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
        key=lambda edge: (edge[0], edge[1]),
    )
    level, peak = 0, 0
    for _, change in edges:
        level += change
        peak = max(peak, level)
    assert peak == 2
    assert all(Path(job).exists() for job in jobs)


def test_children_run_single_threaded_and_the_parent_keeps_its_environment() -> None:
    before = [os.environ.get(name) for name in SINGLE_THREAD_VARIABLES]
    value = run_pool([0], targets.thread_variables, 1, "normal")[0].value
    assert value == ["1"] * len(SINGLE_THREAD_VARIABLES)
    assert [os.environ.get(name) for name in SINGLE_THREAD_VARIABLES] == before


def test_each_result_reaches_the_callback_as_it_ends() -> None:
    seen: list[tuple[int, JobResult[int]]] = []
    run_pool([1, 2], targets.double, 1, "normal", on_result=lambda i, r: seen.append((i, r)))
    assert [(index, result.value) for index, result in seen] == [(0, 2), (1, 4)]


def test_normal_priority_reaches_the_child() -> None:
    result = run_pool([0], targets.real_priority, 1, "normal")[0]
    assert result.priority == "normal"
    if sys.platform == "win32":
        assert result.value == NORMAL_PRIORITY_CLASS


def test_above_normal_reaches_the_child_or_is_reported_as_refused() -> None:
    result = run_pool([0], targets.real_priority, 1, "above_normal")[0]
    if sys.platform == "win32":
        assert result.priority == "above_normal"
        assert result.value == ABOVE_NORMAL_PRIORITY_CLASS
    elif result.priority == "above_normal":
        assert result.value == ABOVE_NORMAL_NICE
    else:
        assert result.priority == "normal"


def test_interrupt_starts_nothing_new_and_ends_what_outlives_the_grace() -> None:
    # The signal reaches only this process, so the child never records anything and outlives
    # the grace, as one that hangs would.
    timer = threading.Timer(INTERRUPT_AFTER_SECONDS, signal.raise_signal, [signal.SIGINT])
    timer.start()
    try:
        results = run_pool([1, 2, 3], targets.sleep_forever, 1, "normal", grace_seconds=0.5)
    finally:
        timer.cancel()
    assert [result.status for result in results] == ["crashed", "not_started", "not_started"]
