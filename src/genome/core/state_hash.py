"""Bit-exact hash of simulation state, for determinism tests (ARQUITETURA §2.5, PADROES §7).

Two states hash equal only if every named array has the same name, dtype,
shape and bytes. The hash never depends on the order in which arrays are given.
"""

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

NamedArray = tuple[str, npt.NDArray[np.generic]]

# Separates fields so that ("ab", "c") and ("a", "bc") never collide.
FIELD_SEPARATOR = b"\x00"


def state_hash(arrays: Sequence[NamedArray]) -> str:
    """Return the SHA-256 hex digest of a set of named arrays.

    Args:
        arrays: ``(name, array)`` pairs. Names must be unique.

    Returns:
        Hex digest that changes with any name, dtype, shape or bit of data.

    Raises:
        ValueError: On duplicate names or arrays of ``object`` dtype, whose
            bytes are memory addresses rather than values.
    """
    ordered = sorted(arrays, key=lambda pair: pair[0])
    _require_unique_names(ordered)
    digest = hashlib.sha256()
    for name, array in ordered:
        _update(digest, name, np.asarray(array))
    return digest.hexdigest()


def _update(digest: "hashlib._Hash", name: str, array: npt.NDArray[np.generic]) -> None:
    if array.dtype.hasobject:
        raise ValueError(f"array {name!r} has object dtype and cannot be hashed")
    contiguous = np.ascontiguousarray(array)
    header = f"{name}|{contiguous.dtype.str}|{contiguous.shape}".encode()
    digest.update(header + FIELD_SEPARATOR)
    digest.update(contiguous.tobytes())
    digest.update(FIELD_SEPARATOR)


def _require_unique_names(ordered: Sequence[NamedArray]) -> None:
    duplicated = sorted({a[0] for a, b in itertools.pairwise(ordered) if a[0] == b[0]})
    if duplicated:
        raise ValueError(f"duplicate array names: {duplicated}")
