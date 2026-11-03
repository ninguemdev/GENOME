"""Seed hierarchy and independent random streams (ARQUITETURA §2.5).

Every random number in GENOME comes from a generator built here. Each stream is
``default_rng(SeedSequence(entropy, spawn_key=(domain, g, k, stream)))`` (PCG64),
with a fixed four-element key and zeros in positions a domain does not use
(ADR 0002). This is the only module allowed to create generators.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast

import numpy as np

from genome.core.config import LabConfig, TrainingConfig

# Unused positions of the spawn key are filled with this value (ADR 0002).
UNUSED_KEY = 0


class Domain(IntEnum):
    """Top level of the seed hierarchy (§2.5). Values are part of the format: never renumber."""

    TRAINING = 1
    CONTINUOUS = 2
    EVOLUTION = 3
    VALIDATION = 4
    TEST = 5


class Stream(IntEnum):
    """The four streams of every world (§2.5). Values are part of the format: never renumber."""

    WORLD = 1
    CLIMATE = 2
    PLANTS = 3
    CREATURES = 4


WORLD_STREAMS: tuple[Stream, ...] = (Stream.WORLD, Stream.CLIMATE, Stream.PLANTS, Stream.CREATURES)

# JSON-compatible form of ``bit_generator.state``; ``Any`` because NumPy types it so.
GeneratorState = dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorldStreams:
    """The four independent generators of one world (§2.5).

    Attributes:
        world: Rivers, lakes, obstacles and puddle positions.
        climate: Rain sequence.
        plants: Initial positions and growth draws.
        creatures: Initial positions and orientations, in-world mutation and crossover.
    """

    world: np.random.Generator
    climate: np.random.Generator
    plants: np.random.Generator
    creatures: np.random.Generator


def make_generator(
    entropy: int, domain: Domain, g: int, k: int, stream: int
) -> np.random.Generator:
    """Create the PCG64 generator of one node of the seed hierarchy (§2.5).

    Args:
        entropy: Root seed of the run, or laboratory seed of the protocol.
        domain: Top-level domain.
        g: Generation index (0 when unused).
        k: World index (0 when unused).
        stream: Stream identifier (0 when unused).

    Raises:
        ValueError: If any key element is negative.
    """
    _require_non_negative((("entropy", entropy), ("g", g), ("k", k), ("stream", stream)))
    sequence = np.random.SeedSequence(entropy, spawn_key=(int(domain), g, k, stream))
    return np.random.default_rng(sequence)


def training_world_streams(
    root_seed: int, g: int, k: int, training: TrainingConfig
) -> WorldStreams:
    """Streams of world ``k`` of training generation ``g``: key ``(TRAINING, g, k, s)``.

    Raises:
        ValueError: If ``k`` is not a valid world index of the generation.
    """
    _require_index("k", k, training.worlds_per_generation)
    return _world_streams(root_seed, Domain.TRAINING, g, k)


def continuous_world_streams(root_seed: int) -> WorldStreams:
    """Streams of the continuous-mode world: key ``(CONTINUOUS, 0, 0, s)``."""
    return _world_streams(root_seed, Domain.CONTINUOUS, UNUSED_KEY, UNUSED_KEY)


def evolution_stream(root_seed: int) -> np.random.Generator:
    """Stream of the NEAT operators between generations: key ``(EVOLUTION, 0, 0, 0)``."""
    return make_generator(root_seed, Domain.EVOLUTION, UNUSED_KEY, UNUSED_KEY, UNUSED_KEY)


def validation_world_streams(lab: LabConfig, i: int) -> WorldStreams:
    """Streams of validation world ``i``, from the laboratory seed: ``(VALIDATION, 0, i, s)``.

    Raises:
        ValueError: If ``i`` is not in ``[0, n_val)``.
    """
    _require_index("i", i, lab.n_val)
    return _world_streams(lab.lab_seed, Domain.VALIDATION, UNUSED_KEY, i)


def final_test_world_streams(lab: LabConfig, i: int) -> WorldStreams:
    """Streams of test-set world ``i``, from the laboratory seed: ``(TEST, 0, i, s)``.

    The test set is used only for the final evaluation (§10.3). The name avoids
    the ``test_`` prefix so that pytest never collects it.

    Raises:
        ValueError: If ``i`` is not in ``[0, n_test)``.
    """
    _require_index("i", i, lab.n_test)
    return _world_streams(lab.lab_seed, Domain.TEST, UNUSED_KEY, i)


def generator_state(generator: np.random.Generator) -> GeneratorState:
    """Return a JSON-serialisable copy of the generator state (§11.6)."""
    return dict(generator.bit_generator.state)


def restore_generator(state: GeneratorState) -> np.random.Generator:
    """Rebuild a generator from ``generator_state`` output (§11.6).

    Raises:
        ValueError: If the state is not a PCG64 state.
    """
    if state.get("bit_generator") != "PCG64":
        raise ValueError(f"expected a PCG64 state, got {state.get('bit_generator')!r}")
    bit_generator = np.random.PCG64()
    # NumPy types the setter with a private TypedDict; the PCG64 check above guards it.
    bit_generator.state = cast(Any, state)
    return np.random.Generator(bit_generator)


def streams_state(streams: WorldStreams) -> dict[str, GeneratorState]:
    """Return the state of the four world streams, keyed by stream name (§11.6)."""
    return {
        "world": generator_state(streams.world),
        "climate": generator_state(streams.climate),
        "plants": generator_state(streams.plants),
        "creatures": generator_state(streams.creatures),
    }


def restore_streams(state: dict[str, GeneratorState]) -> WorldStreams:
    """Rebuild the four world streams from ``streams_state`` output (§11.6).

    Raises:
        ValueError: If a stream is missing or a state is invalid.
    """
    expected = ("world", "climate", "plants", "creatures")
    missing = [name for name in expected if name not in state]
    if missing:
        raise ValueError(f"missing stream states: {missing}")
    return WorldStreams(
        world=restore_generator(state["world"]),
        climate=restore_generator(state["climate"]),
        plants=restore_generator(state["plants"]),
        creatures=restore_generator(state["creatures"]),
    )


def _world_streams(entropy: int, domain: Domain, g: int, k: int) -> WorldStreams:
    return WorldStreams(
        world=make_generator(entropy, domain, g, k, Stream.WORLD),
        climate=make_generator(entropy, domain, g, k, Stream.CLIMATE),
        plants=make_generator(entropy, domain, g, k, Stream.PLANTS),
        creatures=make_generator(entropy, domain, g, k, Stream.CREATURES),
    )


def _require_non_negative(values: tuple[tuple[str, int], ...]) -> None:
    negative = [name for name, value in values if value < 0]
    if negative:
        raise ValueError(f"seed hierarchy keys must be >= 0: {negative}")


def _require_index(name: str, value: int, size: int) -> None:
    if not 0 <= value < size:
        raise ValueError(f"{name} must be in [0, {size}), got {value}")
