"""Checkpoints on disk: arrays, genomes and random streams (ARQUITETURA §11.1, §11.6, ADR 0009).

Each checkpoint is one directory, written aside and renamed at the end, so that a
crash never leaves half of one under the final name.
"""

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig, config_hash
from genome.core.snapshot import SimulationSnapshot
from genome.core.state_hash import NamedArray

#: Version of this layout; a checkpoint of any other version is refused (§11.4).
FORMAT_VERSION = 1
CHECKPOINT_PREFIX = "checkpoint_"
GENOMES_FILE = "genomes.json"
RNG_FILE = "rng_state.json"
# Prefixes inside the ``.npz``, next to the arrays of the snapshot.
META_VERSION = "meta.format_version"
META_CONFIG_HASH = "meta.config_hash"
EXTRA_PREFIX = "extra."

JsonValue = Any  # What ``json`` reads and writes; each codec knows its own structure.


class GenomeCodec[G](Protocol):
    """Turns genomes into JSON and back; ``genome.lab`` injects one (§11.6, ADR 0009)."""

    def encode(self, genome: G) -> JsonValue:
        """The JSON form of ``genome``."""
        ...

    def decode(self, data: JsonValue) -> G:
        """The genome that ``encode`` wrote as ``data``."""
        ...


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint[G]:
    """What was read from a checkpoint directory.

    Attributes:
        snapshot: The world, ready for ``Simulation.restore``.
        extras: The arrays the writer asked to keep, under the names it gave them.
    """

    snapshot: SimulationSnapshot[G]
    extras: dict[str, npt.NDArray[np.generic]]


def checkpoint_path(root: Path, tick: int) -> Path:
    """The directory of the checkpoint taken before ``tick`` runs (§11.1, ADR 0009)."""
    return root / f"{CHECKPOINT_PREFIX}{tick}"


def latest_checkpoint(root: Path) -> Path | None:
    """The checkpoint directory of the highest tick under ``root``, if there is any."""
    if not root.is_dir():
        return None
    ticks = [
        int(entry.name.removeprefix(CHECKPOINT_PREFIX))
        for entry in sorted(root.iterdir())
        if entry.is_dir() and entry.name.removeprefix(CHECKPOINT_PREFIX).isdecimal()
    ]
    return checkpoint_path(root, max(ticks)) if ticks else None


def save_checkpoint[G](
    root: Path,
    snapshot: SimulationSnapshot[G],
    config: GenomeConfig,
    codec: GenomeCodec[G],
    extras: Sequence[NamedArray] = (),
) -> Path:
    """Write a checkpoint under ``root`` and return its directory (§11.1, §11.6).

    Args:
        root: Directory that holds the checkpoints of a run.
        snapshot: The world.
        config: Its configuration, whose hash guards the restore.
        codec: How to write the genomes.
        extras: Arrays of whoever runs the world, returned as they are by the load.
    """
    tick = int(dict(snapshot.arrays)["tick"][0])
    final = checkpoint_path(root, tick)
    temporary = root / f".{final.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    # The stubs fear a key named ``allow_pickle``; the names here are the ones of the state.
    arrays: dict[str, Any] = _npz_arrays(snapshot.arrays, config, extras)
    np.savez_compressed(temporary / f"{final.name}.npz", **arrays)
    genomes = [{"slot": slot, "genome": codec.encode(g)} for slot, g in snapshot.genomes]
    (temporary / GENOMES_FILE).write_text(json.dumps(genomes), encoding="utf-8")
    (temporary / RNG_FILE).write_text(json.dumps(snapshot.streams), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    return final


def load_checkpoint[G](
    path: Path, config: GenomeConfig, codec: GenomeCodec[G]
) -> LoadedCheckpoint[G]:
    """Read a checkpoint directory (§11.6).

    Raises:
        ValueError: If it has another format version or belongs to another configuration.
    """
    with np.load(path / f"{path.name}.npz", allow_pickle=False) as stored:
        loaded = {name: stored[name] for name in sorted(stored.files)}
    _require_compatible(loaded, config)
    arrays = tuple(
        (name, array)
        for name, array in sorted(loaded.items())
        if not name.startswith(("meta.", EXTRA_PREFIX))
    )
    extras = {
        name.removeprefix(EXTRA_PREFIX): array
        for name, array in sorted(loaded.items())
        if name.startswith(EXTRA_PREFIX)
    }
    entries = json.loads((path / GENOMES_FILE).read_text(encoding="utf-8"))
    genomes = tuple((int(entry["slot"]), codec.decode(entry["genome"])) for entry in entries)
    streams = json.loads((path / RNG_FILE).read_text(encoding="utf-8"))
    return LoadedCheckpoint(SimulationSnapshot(arrays, streams, genomes), extras)


def _npz_arrays(
    arrays: Sequence[NamedArray], config: GenomeConfig, extras: Sequence[NamedArray]
) -> dict[str, npt.NDArray[np.generic]]:
    stored = dict(arrays)
    stored[META_VERSION] = np.array([FORMAT_VERSION], dtype=np.int64)
    stored[META_CONFIG_HASH] = np.array(config_hash(config))
    for name, array in extras:
        stored[f"{EXTRA_PREFIX}{name}"] = array
    return stored


def _require_compatible(stored: dict[str, npt.NDArray[np.generic]], config: GenomeConfig) -> None:
    version = int(stored[META_VERSION][0])
    if version != FORMAT_VERSION:
        raise ValueError(f"checkpoint format {version} is not {FORMAT_VERSION} (§11.4)")
    if str(stored[META_CONFIG_HASH]) != config_hash(config):
        raise ValueError("the checkpoint belongs to another configuration")
