"""Numba compilation policy of the core (ARQUITETURA §2.5, PADROES §5).

Every kernel is compiled with ``cache=True`` and ``fastmath=False``, so that
floating-point results are bit-identical to the reference semantics. Modules
keep the plain Python function next to the compiled one; tests call the plain
version too, which keeps the kernel bodies under coverage.
"""

from collections.abc import Callable
from typing import Any, cast

from numba import njit


def jit[F: Callable[..., Any]](func: F) -> F:
    """Compile ``func`` with the core's Numba policy, keeping its type for mypy."""
    return cast(F, njit(cache=True, fastmath=False)(func))
