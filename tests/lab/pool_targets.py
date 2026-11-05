"""Module-level targets of the pool tests: a new process must be able to import them."""

import os
import sys
import time
from pathlib import Path

from genome.lab.pool import SINGLE_THREAD_VARIABLES

#: How long each job of the concurrency test holds its process.
HOLD_SECONDS = 0.5
CRASH_CODE = 3


def double(value: int) -> int:
    return 2 * value


def fail_on_two(value: int) -> int:
    if value == 2:
        raise ValueError("job two fails")
    return value


def crash_on_two(value: int) -> int:
    if value == 2:
        os._exit(CRASH_CODE)
    return value


def large_result(size: int) -> bytes:
    """More than a pipe holds, so sending it blocks until the parent reads."""
    return b"x" * size


def hold_and_record(path: str) -> tuple[float, float]:
    """The interval this process was running in."""
    start = time.monotonic()
    time.sleep(HOLD_SECONDS)
    Path(path).write_text(str(os.getpid()), encoding="utf-8")
    return start, time.monotonic()


def sleep_forever(_: int) -> int:
    while True:
        time.sleep(1.0)


def thread_variables(_: int) -> list[str | None]:
    return [os.environ.get(name) for name in SINGLE_THREAD_VARIABLES]


def real_priority(_: int) -> int:
    """What the system says of this process: its class on Windows, its niceness elsewhere."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        return int(kernel32.GetPriorityClass(ctypes.c_void_p(-1)))
    return os.nice(0)
