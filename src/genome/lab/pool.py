"""Independent worlds in parallel processes, each with its own fate (ARQUITETURA §12.5, ADR 0011).

One process per job, instead of a ``ProcessPoolExecutor``: there, a process that dies abruptly
breaks the whole pool, and here it must cost only its own world. Processes share nothing, so
the number of workers and their priority never reach the result of a world (§2.5).
"""

import ctypes
import math
import multiprocessing
import os
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.context import SpawnProcess
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from genome.core.config.sections import ProcessPriority

#: A world per processor is the whole parallelism: libraries must not add threads to it (§12.5).
SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
#: How long the worlds in progress get to record an interruption before they are ended.
INTERRUPT_GRACE_SECONDS = 60.0
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
#: The pseudo handle of Windows for the calling process.
CURRENT_PROCESS = -1
#: Niceness of ``above_normal`` where processes have one; lowering it needs permission.
ABOVE_NORMAL_NICE = -5

JobStatus = Literal["ok", "failed", "crashed", "not_started"]
#: ``Connection`` on POSIX and ``PipeConnection`` on Windows, with no common public type.
Pipe = Any


@dataclass(frozen=True, slots=True)
class JobResult[R]:
    """How one job ended.

    Attributes:
        status: ``ok``, ``failed`` (an exception), ``crashed`` (the process died) or
            ``not_started`` (the batch was interrupted first).
        value: What the target returned, when ``ok``.
        error: The traceback or the exit code, otherwise.
        priority: The priority the process really had, unknown if it died first.
    """

    status: JobStatus
    value: R | None = None
    error: str | None = None
    priority: "ProcessPriority | None" = None


@dataclass(slots=True)
class _Running:
    index: int
    process: SpawnProcess
    pipe: Pipe
    message: tuple[JobStatus, object, str | None, "ProcessPriority"] | None = None


def ceiling_workers(cpus: int, fraction: float) -> int:
    """Most processes allowed: ``floor(cpus · fraction)``, and always at least one (§12.5)."""
    return max(1, math.floor(cpus * fraction))


def set_process_priority(priority: "ProcessPriority") -> "ProcessPriority":
    """Set the priority of this process, returning the one it really got (ADR 0011).

    It only decides when the process runs, never what it computes. Where the system refuses,
    the process stays at ``normal``.
    """
    if priority == "normal":
        return "normal"
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32")
        handle = ctypes.c_void_p(CURRENT_PROCESS)
        accepted = kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)
        return "above_normal" if accepted else "normal"
    try:
        os.nice(ABOVE_NORMAL_NICE)
    except PermissionError:
        return "normal"
    return "above_normal"


def run_pool[J, R](
    jobs: Sequence[J],
    target: Callable[[J], R],
    workers: int,
    priority: "ProcessPriority",
    on_result: Callable[[int, JobResult[R]], None] | None = None,
    grace_seconds: float = INTERRUPT_GRACE_SECONDS,
) -> list[JobResult[R]]:
    """Run every job in its own process, at most ``workers`` at a time, started in order.

    A failure costs only its own job. On Ctrl+C no new job starts, and the ones in progress
    get ``grace_seconds`` to record the interruption before they are ended.

    Args:
        jobs: The jobs; they and the results must be picklable.
        target: A module-level function, which the new process imports.
        workers: Most processes at the same time.
        priority: Priority of the processes.
        on_result: Called in the parent as each job ends, with its index and result.
        grace_seconds: Time for a job to end after Ctrl+C.

    Returns:
        One result per job, in the order of ``jobs``.

    Raises:
        ValueError: If ``workers`` is not positive.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    results: list[JobResult[R] | None] = [None] * len(jobs)

    def finish(index: int, result: JobResult[R]) -> None:
        results[index] = result
        if on_result is not None:
            on_result(index, result)

    context = multiprocessing.get_context("spawn")
    pending = deque(range(len(jobs)))
    running: list[_Running] = []
    deadline: float | None = None
    with _single_threaded_children():
        try:
            while pending or running:
                try:
                    while pending and len(running) < workers:
                        index = pending.popleft()
                        running.append(_launch(context, index, jobs[index], target, priority))
                    ended = _wait_for_ends(running, deadline)
                    if ended is None:
                        _end_all(running)
                        ended = list(running)
                except KeyboardInterrupt:
                    while pending:
                        finish(pending.popleft(), JobResult("not_started"))
                    if deadline is not None:
                        _end_all(running)
                    deadline = time.monotonic() + grace_seconds
                    continue
                for child in ended:
                    running.remove(child)
                    finish(child.index, _result_of(child))
        finally:
            _end_all(running)
    return [result for result in results if result is not None]


def _launch[J, R](
    context: multiprocessing.context.SpawnContext,
    index: int,
    job: J,
    target: Callable[[J], R],
    priority: "ProcessPriority",
) -> _Running:
    parent_end, child_end = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(target, job, priority, child_end), daemon=True)
    process.start()
    # Only the child holds this end now, so its death reads as the end of the pipe.
    child_end.close()
    return _Running(index, process, parent_end)


def _child[J, R](target: Callable[[J], R], job: J, priority: "ProcessPriority", pipe: Pipe) -> None:
    applied = set_process_priority(priority)
    message: tuple[JobStatus, object, str | None, ProcessPriority]
    try:
        message = ("ok", target(job), None, applied)
    except Exception:
        message = ("failed", None, traceback.format_exc(), applied)
    pipe.send(message)
    pipe.close()


def _wait_for_ends(running: list[_Running], deadline: float | None) -> list[_Running] | None:
    """The children that ended, reading what they sent; ``None`` when the deadline passed.

    Pipes are watched too: a child blocked on a result that fills its pipe never ends otherwise.
    """
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
    watched: list[Pipe] = []
    for child in running:
        watched.append(child.process.sentinel)
        if child.message is None and not child.pipe.closed:
            watched.append(child.pipe)
    ready = wait(watched, timeout)
    if not ready:
        return None
    for child in running:
        if child.pipe in ready:
            _read_message(child)
    return [child for child in running if child.process.sentinel in ready]


def _read_message(child: _Running) -> None:
    """Keep what the child sent; a child that died shows as an end of file or a broken pipe."""
    try:
        child.message = child.pipe.recv()
    except (EOFError, OSError):
        child.pipe.close()


def _result_of[R](child: _Running) -> JobResult[R]:
    child.process.join()
    if child.message is None and not child.pipe.closed:
        _read_message(child)
    if not child.pipe.closed:
        child.pipe.close()
    if child.message is None:
        return JobResult("crashed", error=f"exit code {child.process.exitcode}")
    status, value, error, applied = child.message
    # The value came from ``target``, which returns R.
    return JobResult(status, value, error, applied)  # type: ignore[arg-type]


def _end_all(running: list[_Running]) -> None:
    for child in running:
        if child.process.is_alive():
            child.process.terminate()


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Children inherit the environment when they start, so set it only around the pool."""
    saved = {name: os.environ.get(name) for name in SINGLE_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(SINGLE_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
