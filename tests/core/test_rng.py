import json
from pathlib import Path

import numpy as np
import pytest

from genome.core import rng
from genome.core.config import GenomeConfig, load_config
from genome.core.rng import Domain, Stream, WorldStreams

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "training.yaml"
DRAWS = 16


@pytest.fixture(scope="module")
def config() -> GenomeConfig:
    return load_config(CONFIG)


def draws(generator: np.random.Generator) -> list[float]:
    return [float(x) for x in generator.random(DRAWS)]


def stream_draws(streams: WorldStreams) -> tuple[list[float], ...]:
    return (
        draws(streams.world),
        draws(streams.climate),
        draws(streams.plants),
        draws(streams.creatures),
    )


def test_key_layout_matches_seed_sequence(config: GenomeConfig) -> None:
    """§2.5: SeedSequence(entropy, spawn_key=(DOMAIN, g, k, STREAM)) with PCG64."""
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(42, spawn_key=(1, 7, 2, 3)))
    )
    actual = rng.training_world_streams(42, 7, 2, config.training).plants
    assert draws(actual) == draws(expected)


def test_same_key_same_sequence(config: GenomeConfig) -> None:
    a = rng.training_world_streams(123, 4, 1, config.training)
    b = rng.training_world_streams(123, 4, 1, config.training)
    assert stream_draws(a) == stream_draws(b)


def test_streams_of_one_world_are_independent(config: GenomeConfig) -> None:
    values = stream_draws(rng.training_world_streams(123, 0, 0, config.training))
    assert len({tuple(v) for v in values}) == len(values)


@pytest.mark.parametrize(
    ("other_seed", "other_g", "other_k"),
    [(124, 4, 1), (123, 5, 1), (123, 4, 2)],
)
def test_each_key_element_changes_the_sequence(
    config: GenomeConfig, other_seed: int, other_g: int, other_k: int
) -> None:
    base = rng.training_world_streams(123, 4, 1, config.training)
    other = rng.training_world_streams(other_seed, other_g, other_k, config.training)
    assert draws(base.world) != draws(other.world)


def test_domains_never_share_sequences(config: GenomeConfig) -> None:
    seed = config.lab.lab_seed  # same entropy on purpose: only the domain differs
    sequences = [
        draws(rng.training_world_streams(seed, 0, 0, config.training).world),
        draws(rng.continuous_world_streams(seed).world),
        draws(rng.evolution_stream(seed)),
        draws(rng.validation_world_streams(config.lab, 0).world),
        draws(rng.final_test_world_streams(config.lab, 0).world),
    ]
    assert len({tuple(s) for s in sequences}) == len(sequences)


def test_validation_and_test_sets_are_disjoint(config: GenomeConfig) -> None:
    val = {tuple(draws(rng.validation_world_streams(config.lab, i).world)) for i in range(3)}
    test = {tuple(draws(rng.final_test_world_streams(config.lab, i).world)) for i in range(10)}
    assert len(val) == config.lab.n_val
    assert len(test) == config.lab.n_test
    assert val.isdisjoint(test)


@pytest.mark.parametrize("bad_k", [-1, 3])
def test_training_world_index_is_checked(config: GenomeConfig, bad_k: int) -> None:
    with pytest.raises(ValueError, match=r"k must be in \[0, 3\)"):
        rng.training_world_streams(1, 0, bad_k, config.training)


def test_validation_index_is_checked(config: GenomeConfig) -> None:
    with pytest.raises(ValueError, match=r"i must be in \[0, 3\)"):
        rng.validation_world_streams(config.lab, 3)


def test_test_index_is_checked(config: GenomeConfig) -> None:
    with pytest.raises(ValueError, match=r"i must be in \[0, 10\)"):
        rng.final_test_world_streams(config.lab, 10)


@pytest.mark.parametrize(
    ("entropy", "g", "name"),
    [(-1, 0, "entropy"), (1, -2, "g")],
)
def test_negative_keys_are_rejected(entropy: int, g: int, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        rng.make_generator(entropy, Domain.TRAINING, g, 0, Stream.WORLD)


def test_enum_values_are_frozen() -> None:
    """Renumbering would silently change every stream of every saved run."""
    assert [int(d) for d in Domain] == [1, 2, 3, 4, 5]
    assert [int(s) for s in rng.WORLD_STREAMS] == [1, 2, 3, 4]


def test_save_and_restore_generator() -> None:
    generator = rng.evolution_stream(99)
    generator.random(5)
    state = rng.generator_state(generator)
    expected = draws(generator)
    assert draws(rng.restore_generator(state)) == expected


def test_save_and_restore_world_streams(config: GenomeConfig) -> None:
    streams = rng.training_world_streams(7, 3, 2, config.training)
    for generator in (streams.world, streams.climate, streams.plants, streams.creatures):
        generator.normal(size=11)
    state = rng.streams_state(streams)
    expected = stream_draws(streams)
    assert stream_draws(rng.restore_streams(state)) == expected


def test_state_survives_json_round_trip(config: GenomeConfig) -> None:
    streams = rng.continuous_world_streams(2026)
    streams.plants.integers(0, 100, size=37)
    text = json.dumps(rng.streams_state(streams), sort_keys=True)
    restored = rng.restore_streams(json.loads(text))
    assert stream_draws(restored) == stream_draws(streams)


def test_restore_rejects_other_bit_generators() -> None:
    with pytest.raises(ValueError, match="PCG64"):
        rng.restore_generator({"bit_generator": "MT19937", "state": {}})


def test_restore_requires_all_streams(config: GenomeConfig) -> None:
    state = rng.streams_state(rng.continuous_world_streams(1))
    del state["plants"]
    with pytest.raises(ValueError, match="plants"):
        rng.restore_streams(state)
