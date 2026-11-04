import dataclasses
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from genome.core.config import GenomeConfig
from genome.core.state_hash import state_hash
from genome.io.checkpoints import (
    FORMAT_VERSION,
    GENOMES_FILE,
    RNG_FILE,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from genome.lab.clonal import FixedGenomeCodec
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation

CODEC = FixedGenomeCodec()


def small(config: GenomeConfig) -> GenomeConfig:
    run = dataclasses.replace(config.run, max_population=30, max_ticks=20)
    world = dataclasses.replace(config.world, initial_population=6)
    training = dataclasses.replace(config.training, n=6, t_max=20)
    return dataclasses.replace(config, run=run, world=world, training=training)


def test_file_layout_follows_the_specification(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = small(config)
    simulation = make_simulation(chosen, 2, "random")
    for _ in range(4):
        simulation.step_batch()
    path = save_checkpoint(tmp_path, simulation.snapshot(), chosen, CODEC)
    assert path == checkpoint_path(tmp_path, 4)
    assert sorted(entry.name for entry in path.iterdir()) == sorted(
        ["checkpoint_4.npz", GENOMES_FILE, RNG_FILE]
    )


def test_a_run_resumed_from_disk_equals_the_direct_one(
    config: GenomeConfig, tmp_path: Path
) -> None:
    chosen = small(config)
    direct = make_simulation(chosen, 2, "random")
    for _ in range(5):
        direct.step_batch()
    path = save_checkpoint(tmp_path, direct.snapshot(), chosen, CODEC)
    loaded = load_checkpoint(path, chosen, CODEC)
    resumed = make_simulation(chosen, 2, "random")
    resumed.restore(loaded.snapshot)
    assert loaded.snapshot.streams == direct.snapshot().streams
    assert state_hash(loaded.snapshot.arrays) == state_hash(direct.named_arrays())
    for _ in range(10):
        direct.step_batch()
        resumed.step_batch()
    assert state_hash(resumed.named_arrays()) == state_hash(direct.named_arrays())


def test_genomes_and_extras_come_back(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = small(config)
    snapshot = make_simulation(chosen, 2, "immobile").snapshot()
    energy = np.array([1.5, 2.5], dtype=np.float64)
    path = save_checkpoint(tmp_path, snapshot, chosen, CODEC, extras=[("energy", energy)])
    loaded = load_checkpoint(path, chosen, CODEC)
    assert loaded.snapshot.genomes == tuple((slot, FixedGenome()) for slot in range(6))
    assert_array_equal(loaded.extras["energy"], energy)


def test_another_configuration_is_refused(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = small(config)
    path = save_checkpoint(
        tmp_path, make_simulation(chosen, 2, "immobile").snapshot(), chosen, CODEC
    )
    with pytest.raises(ValueError, match="another configuration"):
        load_checkpoint(path, config, CODEC)


def test_another_format_version_is_refused(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = small(config)
    path = save_checkpoint(
        tmp_path, make_simulation(chosen, 2, "immobile").snapshot(), chosen, CODEC
    )
    npz = path / f"{path.name}.npz"
    with np.load(npz) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["meta.format_version"] = np.array([FORMAT_VERSION + 1], dtype=np.int64)
    np.savez_compressed(npz, **arrays)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path, chosen, CODEC)


def test_saving_the_same_tick_replaces_it_and_leaves_no_temporary(
    config: GenomeConfig, tmp_path: Path
) -> None:
    chosen = small(config)
    snapshot = make_simulation(chosen, 2, "immobile").snapshot()
    (tmp_path / ".checkpoint_0.tmp").mkdir()
    save_checkpoint(tmp_path, snapshot, chosen, CODEC)
    save_checkpoint(tmp_path, snapshot, chosen, CODEC)
    assert [entry.name for entry in tmp_path.iterdir()] == ["checkpoint_0"]


def test_latest_checkpoint_orders_by_tick_and_skips_the_rest(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path / "missing") is None
    assert latest_checkpoint(tmp_path) is None
    for name in ("checkpoint_9", "checkpoint_10", ".checkpoint_11.tmp", "checkpoint_x"):
        (tmp_path / name).mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint_10"
