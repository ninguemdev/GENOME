import numpy as np
import pytest

from genome.core.state import NO_ID, UNIRRIGATED, CreatureState, PlantState, SlotPool
from genome.core.state_hash import state_hash


def test_pool_allocates_lowest_free_slots_with_growing_ids() -> None:
    pool = SlotPool(4)
    np.testing.assert_array_equal(pool.allocate(3), [0, 1, 2])
    pool.release(np.array([1], dtype=np.int64))
    assert pool.count == 2
    # Slot 1 is reused, but the identifier keeps growing (§2.5).
    np.testing.assert_array_equal(pool.allocate(2), [1, 3])
    np.testing.assert_array_equal(pool.ids, [0, 3, 2, 4])
    assert pool.next_id == 5


def test_active_slots_are_sorted_by_id_not_by_slot() -> None:
    pool = SlotPool(3)
    pool.allocate(3)
    pool.release(np.array([0], dtype=np.int64))
    pool.allocate(1)  # slot 0 now holds id 3
    np.testing.assert_array_equal(pool.active(), [1, 2, 0])


def test_pool_errors() -> None:
    with pytest.raises(ValueError, match="capacity"):
        SlotPool(0)
    pool = SlotPool(2)
    with pytest.raises(ValueError, match="only 2 free"):
        pool.allocate(3)
    pool.allocate(1)
    with pytest.raises(ValueError, match="distinct occupied"):
        pool.release(np.array([1], dtype=np.int64))
    with pytest.raises(ValueError, match="distinct occupied"):
        pool.release(np.array([0, 0], dtype=np.int64))
    assert pool.capacity == 2


def test_plants_add_and_remove() -> None:
    plants = PlantState(3)
    slots = plants.add(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 3.0]), np.array([0, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(slots, [0, 1])
    np.testing.assert_array_equal(plants.position[1], [3.0, 4.0])
    plants.remove(np.array([0], dtype=np.int64))
    assert plants.biomass[0] == 0.0
    assert plants.hops[0] == UNIRRIGATED
    assert plants.pool.ids[0] == NO_ID


def test_state_hash_covers_plants_and_creatures() -> None:
    plants, creatures = PlantState(2), CreatureState(2)
    before = state_hash([*plants.named_arrays(), *creatures.named_arrays()])
    creatures.energy[0] = 1.0
    after = state_hash([*plants.named_arrays(), *creatures.named_arrays()])
    assert before != after
    names = [name for name, _ in creatures.named_arrays()]
    assert "creatures.next_id" in names
    assert "creatures.velocity" in names
