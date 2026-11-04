import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.body import Bodies, build_bodies
from genome.core.config import GenomeConfig
from genome.core.state import NO_ID, CreatureState, NewCreatures


def one(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float64)


def bodies(config: GenomeConfig, mass: float) -> Bodies:
    return build_bodies(one(mass), one(1.0), one(0.5), config.creatures)


def test_reference_creature_matches_section_5_1(config: GenomeConfig) -> None:
    b = bodies(config, 1.0)
    assert_allclose(b.body_radius, 1.0)
    assert_allclose(b.touch_radius, 1.5)
    assert_allclose((b.e_max, b.h_max), ((100.0,), (60.0,)))
    assert_allclose((b.t_vida, b.t_mat), ((3000.0,), (300.0,)))
    assert_allclose(b.inertia_decay, math.exp(-1 / 3))
    assert_allclose(b.rotation_decay, math.exp(-1 / 2))


def test_heavy_creature_computed_by_hand(config: GenomeConfig) -> None:
    b = bodies(config, 2.0)
    assert_allclose(b.body_radius, math.sqrt(2))
    assert_allclose((b.e_max, b.h_max), ((200.0,), (120.0,)))
    assert_allclose(b.t_vida, 3567.6, atol=0.05)
    assert_allclose(b.t_mat, 356.76, atol=0.005)
    assert_allclose(b.inertia_decay, math.exp(-1 / 6))
    assert_allclose(b.rotation_decay, math.exp(-1 / 4))


def test_lightest_creature_outlives_a_generation(config: GenomeConfig) -> None:
    """§5.1: the minimum life expectancy, about 2523, exceeds t_max = 2000."""
    b = bodies(config, 0.5)
    assert_allclose(b.t_vida, 2522.7, atol=0.05)
    assert b.t_vida[0] > config.training.t_max


@pytest.mark.parametrize(
    ("mass", "v_max", "color", "message"),
    [
        (0.4, 1.0, 0.5, "mass outside"),
        (2.1, 1.0, 0.5, "mass outside"),
        (1.0, 0.4, 0.5, "v_max outside"),
        (1.0, 1.6, 0.5, "v_max outside"),
        (1.0, 1.0, 1.0, "color outside"),
        (1.0, 1.0, -0.1, "color outside"),
    ],
)
def test_genes_out_of_range_are_rejected(
    config: GenomeConfig, mass: float, v_max: float, color: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_bodies(one(mass), one(v_max), one(color), config.creatures)


def newborn(config: GenomeConfig, mass: float = 1.0, parent: int = NO_ID) -> NewCreatures:
    return NewCreatures(
        bodies=bodies(config, mass),
        position=np.array([[1.0, 2.0]]),
        heading=one(0.5),
        energy=one(60.0),
        hydration=one(48.0),
        parent_uid=np.array([parent], dtype=np.int64),
        founder_id=np.array([NO_ID], dtype=np.int64),
        birth_tick=7,
    )


def test_add_stores_body_and_makes_founder_its_own_lineage(config: GenomeConfig) -> None:
    state = CreatureState(3)
    slot = state.add(newborn(config, mass=2.0))[0]
    assert state.founder_id[slot] == state.pool.ids[slot] == 0
    assert state.parent_uid[slot] == NO_ID
    assert state.birth_tick[slot] == 7
    assert state.mass[slot] == 2.0
    assert_allclose(state.t_mat[slot], 356.76, atol=0.005)
    assert_allclose(state.position[slot], [1.0, 2.0])
    assert state.energy[slot] == 60.0


def test_reused_slot_does_not_inherit_the_dead_creature(config: GenomeConfig) -> None:
    state = CreatureState(1)
    slot = state.add(newborn(config))[0]
    state.velocity[slot] = (1.0, 1.0)
    state.angular_velocity[slot] = 0.1
    state.age[slot] = 99
    state.memory[slot] = 0.7
    state.last_reproduction_tick[slot] = 5
    state.n_children[slot] = 2
    state.remove(np.array([slot], dtype=np.int64))
    assert state.energy[slot] == 0.0
    assert state.hydration[slot] == 0.0

    again = state.add(newborn(config))[0]
    assert again == slot
    assert state.pool.ids[slot] == 1  # the uid keeps growing (§2.5)
    assert_allclose(state.velocity[slot], [0.0, 0.0])
    assert state.angular_velocity[slot] == 0.0
    assert state.age[slot] == 0
    assert state.memory[slot] == 0.0
    assert state.last_reproduction_tick[slot] == NO_ID
    assert state.n_children[slot] == 0


def test_child_inherits_the_lineage_it_is_given(config: GenomeConfig) -> None:
    state = CreatureState(2)
    founder = state.add(newborn(config))[0]
    child = newborn(config, parent=int(state.pool.ids[founder]))
    inherited = NewCreatures(
        bodies=child.bodies,
        position=child.position,
        heading=child.heading,
        energy=child.energy,
        hydration=child.hydration,
        parent_uid=child.parent_uid,
        founder_id=state.founder_id[np.array([founder])],
        birth_tick=child.birth_tick,
    )
    slot = state.add(inherited)[0]
    assert state.founder_id[slot] == 0
    assert state.parent_uid[slot] == 0
    assert state.pool.ids[slot] == 1
