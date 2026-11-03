import numpy as np
import pytest

from genome.core.state_hash import state_hash


def base() -> list[tuple[str, np.ndarray]]:
    return [
        ("energy", np.array([1.0, 2.5, 3.25])),
        ("position", np.array([[0.0, 1.0], [2.0, 3.0]])),
        ("alive", np.array([True, False, True])),
    ]


def test_equal_states_hash_equal() -> None:
    assert state_hash(base()) == state_hash(base())
    assert len(state_hash(base())) == 64


def test_independent_of_pair_order() -> None:
    assert state_hash(base()) == state_hash(list(reversed(base())))


def test_sensitive_to_one_ulp() -> None:
    changed = base()
    energy = changed[0][1].copy()
    energy[1] = np.nextafter(energy[1], np.inf)
    changed[0] = ("energy", energy)
    assert state_hash(changed) != state_hash(base())


def test_sensitive_to_element_order() -> None:
    changed = base()
    changed[0] = ("energy", changed[0][1][::-1].copy())
    assert state_hash(changed) != state_hash(base())


def test_sensitive_to_dtype() -> None:
    a = [("x", np.array([1, 2, 3], dtype=np.int64))]
    b = [("x", np.array([1, 2, 3], dtype=np.int32))]
    assert state_hash(a) != state_hash(b)


def test_sensitive_to_shape() -> None:
    a = [("x", np.arange(6.0).reshape(2, 3))]
    b = [("x", np.arange(6.0).reshape(3, 2))]
    assert state_hash(a) != state_hash(b)


def test_sensitive_to_names() -> None:
    a = [("ab", np.array([1.0])), ("c", np.array([2.0]))]
    b = [("a", np.array([1.0])), ("bc", np.array([2.0]))]
    assert state_hash(a) != state_hash(b)


def test_non_contiguous_views_hash_like_copies() -> None:
    matrix = np.arange(12.0).reshape(3, 4)
    view = matrix[:, ::2]
    assert state_hash([("m", view)]) == state_hash([("m", view.copy())])


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate array names"):
        state_hash([("x", np.zeros(1)), ("x", np.ones(1))])


def test_object_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="object dtype"):
        state_hash([("x", np.array([1, "a"], dtype=object))])
