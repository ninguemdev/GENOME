import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from genome.core import space

W, H = 256.0, 128.0
ATOL = 1e-9

coord_x = st.floats(0.0, W, exclude_max=True, allow_nan=False)
coord_y = st.floats(0.0, H, exclude_max=True, allow_nan=False)
points = st.tuples(coord_x, coord_y)
anything = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
shifts = st.integers(-5, 5)


# ---------------------------------------------------------------- hand-computed values


def test_minimum_image_hand_values() -> None:
    deltas = np.array([0.0, 10.0, -10.0, 200.0, -200.0, 127.0, 129.0, 128.0, -128.0])
    expected = np.array([0.0, 10.0, -10.0, -56.0, 56.0, 127.0, -127.0, -128.0, -128.0])
    assert_array_equal(space.minimum_image(deltas, W), expected)


def test_displacement_across_the_border() -> None:
    a = np.array([[250.0, 2.0]])
    b = np.array([[4.0, 126.0]])
    assert_array_equal(space.displacement(a, b, W, H), [[10.0, -4.0]])
    assert_array_equal(space.displacement(b, a, W, H), [[-10.0, 4.0]])


def test_distance_across_the_corner() -> None:
    # (1, 1) and (W - 2, H - 3): offsets 3 and 4 through the corner.
    assert_allclose(space.distance([1.0, 1.0], [W - 2.0, H - 3.0], W, H), 5.0)


def test_distance_broadcasts_one_to_many() -> None:
    many = np.array([[0.0, 0.0], [3.0, 4.0], [W - 3.0, H - 4.0]])
    assert_allclose(space.distance([0.0, 0.0], many, W, H), [0.0, 5.0, 5.0])


def test_wrap_fixes_rounding_to_size() -> None:
    """§3.1: x mod W may round to exactly W; it must become 0."""
    tiny = -1e-17
    assert np.mod(tiny, W) == W  # the floating-point trap itself
    assert space.wrap_coordinate(tiny, W) == 0.0
    wrapped = space.wrap_positions([[tiny, tiny]], W, H)
    assert_array_equal(wrapped, [[0.0, 0.0]])


def test_wrap_positions_hand_values() -> None:
    p = np.array([[-1.0, 130.0], [256.0, 128.0], [512.5, -0.5]])
    expected = np.array([[255.0, 2.0], [0.0, 0.0], [0.5, 127.5]])
    assert_array_equal(space.wrap_positions(p, W, H), expected)


def test_wrap_positions_does_not_modify_input() -> None:
    p = np.array([[-1.0, -1.0]])
    space.wrap_positions(p, W, H)
    assert_array_equal(p, [[-1.0, -1.0]])


def test_wrap_angle_hand_values() -> None:
    angles = np.array([0.0, math.pi, -math.pi, 3 * math.pi / 2, -3 * math.pi / 2, 4 * math.pi])
    expected = np.array([0.0, -math.pi, -math.pi, -math.pi / 2, math.pi / 2, 0.0])
    assert_allclose(space.wrap_angle(angles), expected, atol=1e-12)


def test_results_are_float64() -> None:
    assert space.distance([0, 0], [1, 1], W, H).dtype == np.float64
    assert space.wrap_positions([[1, 2]], W, H).dtype == np.float64


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_invalid_size_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        space.wrap_coordinate(1.0, bad)


@pytest.mark.parametrize("shape", [(3,), (2, 3), ()])
def test_points_must_have_two_coordinates(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
        space.wrap_positions(np.zeros(shape), W, H)


# ---------------------------------------------------------------- properties


@given(points, points)
def test_distance_is_symmetric(a: tuple[float, float], b: tuple[float, float]) -> None:
    assert_allclose(space.distance(a, b, W, H), space.distance(b, a, W, H), atol=ATOL)


@given(points)
def test_distance_to_self_is_zero(a: tuple[float, float]) -> None:
    assert space.distance(a, a, W, H) == 0.0


@given(points, points)
def test_components_lie_in_half_open_box(a: tuple[float, float], b: tuple[float, float]) -> None:
    d = space.displacement(a, b, W, H)
    assert -W / 2 <= d[0] < W / 2
    assert -H / 2 <= d[1] < H / 2


@given(points, points)
def test_distance_is_bounded_by_half_diagonal(
    a: tuple[float, float], b: tuple[float, float]
) -> None:
    assert space.distance(a, b, W, H) <= math.hypot(W / 2, H / 2) + ATOL


@given(points, points, shifts, shifts)
def test_distance_invariant_to_whole_turns(
    a: tuple[float, float], b: tuple[float, float], i: int, j: int
) -> None:
    moved = (b[0] + i * W, b[1] + j * H)
    assert_allclose(space.distance(a, moved, W, H), space.distance(a, b, W, H), atol=ATOL)


@given(points, points, points)
def test_triangle_inequality(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> None:
    ab = space.distance(a, b, W, H)
    bc = space.distance(b, c, W, H)
    ac = space.distance(a, c, W, H)
    assert ac <= ab + bc + ATOL


@given(points, points)
def test_displacement_reaches_target(a: tuple[float, float], b: tuple[float, float]) -> None:
    landed = space.wrap_positions(np.asarray(a) + space.displacement(a, b, W, H), W, H)
    error = space.distance(landed, b, W, H)
    assert error <= ATOL


@given(anything, anything)
def test_wrap_lands_in_range_and_is_idempotent(x: float, y: float) -> None:
    once = space.wrap_positions([x, y], W, H)
    assert 0.0 <= once[0] < W
    assert 0.0 <= once[1] < H
    assert_array_equal(space.wrap_positions(once, W, H), once)


@given(anything)
def test_wrap_angle_lands_in_range(phi: float) -> None:
    wrapped = float(space.wrap_angle(phi))
    assert -math.pi <= wrapped < math.pi
    assert_allclose(math.cos(wrapped), math.cos(phi), atol=1e-6)
    assert_allclose(math.sin(wrapped), math.sin(phi), atol=1e-6)
