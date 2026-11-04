import dataclasses
import math

import numpy as np
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.envs.rewards import base_increment, bonus_increment, descendants
from tests.core.scenes import make_creatures


def fitness_terms(config: GenomeConfig) -> GenomeConfig:
    """The values of the tables of §9.2: t_max = 2000, w_S = w_B = 0.5, k_repr = 0.5, D_sat = 5."""
    training = dataclasses.replace(
        config.training, t_max=2000, w_s=0.5, w_b=0.5, k_repr=0.5, d_sat=5.0
    )
    return dataclasses.replace(config, training=training)


def test_base_increment_computed_by_hand(config: GenomeConfig) -> None:
    """E = 70 of 100 and H = 30 of 60: (0.5 + 0.5 · 0.5) / 2000 = 3.75e-4 (§9.2)."""
    chosen = fitness_terms(config)
    state = make_creatures(chosen, [(1.0, 1.0)])
    state.energy[0], state.hydration[0] = 70.0, 30.0
    slots = np.array([0], dtype=np.int64)
    result = base_increment(state, slots, np.array([True]), chosen.training)
    assert_allclose(result, [3.75e-4])


def test_a_tick_not_lived_adds_nothing_to_the_base(config: GenomeConfig) -> None:
    chosen = fitness_terms(config)
    state = make_creatures(chosen, [(1.0, 1.0)])
    slots = np.array([0], dtype=np.int64)
    assert base_increment(state, slots, np.array([False]), chosen.training)[0] == 0.0


def test_a_whole_life_gives_the_row_of_the_table(config: GenomeConfig) -> None:
    """2000 ticks with the worst reserve at 70%: F_base = 0.85 (§9.2)."""
    chosen = fitness_terms(config)
    state = make_creatures(chosen, [(1.0, 1.0)])
    state.energy[0], state.hydration[0] = 70.0, 60.0
    slots = np.array([0], dtype=np.int64)
    one_tick = base_increment(state, slots, np.array([True]), chosen.training)[0]
    assert_allclose(2000 * one_tick, 0.85)


def test_bonus_follows_the_table(config: GenomeConfig) -> None:
    """B_repr = 0.09, 0.32 and 0.43 for D = 1, 5 and 10 (§9.2), and increments add up."""
    training = fitness_terms(config).training
    steps = np.array([0.0, 1.0, 5.0, 10.0])
    increments = bonus_increment(steps[:-1], steps[1:], training)
    assert_allclose(np.cumsum(increments), [0.0906, 0.3161, 0.4323], atol=1e-4)
    assert_allclose(np.sum(increments), 0.5 * (1.0 - math.exp(-2.0)))


def test_descendants_weigh_grandchildren_by_lambda(config: GenomeConfig) -> None:
    training = dataclasses.replace(config.training, lambda_grandchildren=0.5)
    state = make_creatures(config, [(1.0, 1.0)])
    state.n_children[0], state.n_grandchildren[0] = 2, 3
    assert_allclose(descendants(state, np.array([0], dtype=np.int64), training), [3.5])
