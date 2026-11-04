"""What a checkpoint holds of a world, in memory (ARQUITETURA §2.1, §11.6, ADR 0009).

A snapshot carries the same named arrays that ``state_hash`` reads, so the hash
proves that a state came back whole. Writing it to disk belongs to ``genome.io``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.rng import GeneratorState
from genome.core.state_hash import NamedArray

SavedArrays = Mapping[str, npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class SimulationSnapshot[G]:
    """A complete copy of one world at the start of a tick (§11.6).

    Attributes:
        arrays: Copy of every array of ``Simulation.named_arrays``.
        streams: State of the four streams of the world (§2.5).
        genomes: ``(slot, genome)`` of each live creature, in ``uid`` order.
    """

    arrays: tuple[NamedArray, ...]
    streams: dict[str, GeneratorState]
    genomes: tuple[tuple[int, G], ...]


def copy_arrays(targets: Sequence[NamedArray], saved: SavedArrays) -> None:
    """Write the saved values into the arrays that hold the live state, in place."""
    for name, target in targets:
        target[...] = saved[name]


def require_same_arrays(current: Sequence[NamedArray], saved: SavedArrays) -> None:
    """Refuse a snapshot whose fixed geography differs from this world (§11.6).

    Raises:
        ValueError: Naming the first array that differs.
    """
    for name, array in current:
        if not np.array_equal(array, saved[name]):
            raise ValueError(f"the snapshot belongs to another world: {name} differs")
