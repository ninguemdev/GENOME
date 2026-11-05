import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from genome.core.config import GenomeConfig
from genome.core.rng import continuous_world_streams
from genome.core.space import displacement, minimum_image_scalar_py
from genome.core.world.rivers import (
    NEAREST_VERTICES,
    Rivers,
    generate_rivers,
    nearest_segments_py,
)
from tests.core.worlds import SIDE, straight_river


def test_current_profile_by_hand() -> None:
    rivers = straight_river()
    points = np.array([[10.0, 52.0], [10.0, 50.0], [10.0, 54.5], [10.3, 47.0]])
    current = rivers.current(points)
    # 0.3 · (1 − (2·2/8)²) = 0.225; centre 0.3; outside 0; d_c = 3 → 0.3 · (1 − 0.5625).
    np.testing.assert_allclose(current[:, 0], [0.225, 0.3, 0.0, 0.13125])
    np.testing.assert_allclose(current[:, 1], 0.0)


def test_margin_distance_and_contains() -> None:
    rivers = straight_river()
    points = np.array([[10.0, 60.0], [10.0, 51.0], [10.0, 40.0]])
    np.testing.assert_allclose(rivers.margin_distance(points), [6.0, 0.0, 6.0])
    np.testing.assert_array_equal(rivers.contains(points), [False, True, False])


def test_distance_across_the_seam() -> None:
    rivers = straight_river()
    distance, tangent = rivers.centerline(np.array([[99.5, 50.0], [0.2, 98.0]]))
    np.testing.assert_allclose(distance, [0.0, 48.0])
    np.testing.assert_allclose(tangent, [[1.0, 0.0], [1.0, 0.0]])


def test_crossing_rivers_use_the_nearest_of_both(config: GenomeConfig) -> None:
    worst = 0.0
    for seed in range(10):
        rivers = generate_rivers(config.water, config.world, continuous_world_streams(seed).world)
        distance, _ = rivers.centerline(rivers.samples)
        worst = max(worst, float(distance.max()))
    # Every sample lies on a centre line, crossings included (ADR 0004).
    assert worst < 1e-9


def _centerline_reference(rivers: Rivers, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The array form of ADR 0004 that the kernel replaced, for bitwise comparison."""
    w, h = rivers.size
    n, k = points.shape[0], NEAREST_VERTICES
    _, nearest = rivers._vertex_tree.query(points, k=k)
    starts = np.concatenate((rivers.prev_vertex[nearest], nearest), axis=1).ravel()
    ends = np.concatenate((nearest, rivers.next_vertex[nearest]), axis=1).ravel()
    a = rivers.vertices[starts]
    seg = displacement(a, rivers.vertices[ends], w, h)
    length = np.hypot(seg[:, 0], seg[:, 1])
    unit = seg / length[:, None]
    rel = displacement(a, np.repeat(points, 2 * k, axis=0), w, h)
    along = np.clip(np.einsum("ij,ij->i", rel, unit), 0.0, length)
    offset = rel - along[:, None] * unit
    gap = np.hypot(offset[:, 0], offset[:, 1]).reshape(n, 2 * k)
    best = np.argmin(gap, axis=1)
    rows = np.arange(n)
    return gap[rows, best], unit.reshape(n, 2 * k, 2)[rows, best]


@pytest.mark.parametrize("seed", range(5))
def test_kernel_is_bitwise_equal_to_the_array_form(config: GenomeConfig, seed: int) -> None:
    streams = continuous_world_streams(seed)
    rivers = generate_rivers(config.water, config.world, streams.world)
    side = np.array(rivers.size)
    points = np.vstack(
        (
            streams.world.uniform(0.0, 1.0, (4000, 2)) * side,
            rivers.vertices,
            rivers.samples,
            np.array([[0.0, 0.0], [side[0] - 1e-12, side[1] - 1e-12]]),
        )
    )
    distance, tangent = rivers.centerline(points)
    expected_distance, expected_tangent = _centerline_reference(rivers, points)
    np.testing.assert_array_equal(distance, expected_distance)
    np.testing.assert_array_equal(tangent, expected_tangent)


def test_plain_kernel_by_hand() -> None:
    rivers = straight_river()
    points = np.array([[10.0, 52.0], [99.5, 50.0]])
    _, nearest = rivers._vertex_tree.query(points, k=NEAREST_VERTICES)
    distance, tangent = np.empty(2), np.empty((2, 2))
    nearest_segments_py(
        points,
        nearest,
        rivers.vertices,
        rivers.prev_vertex,
        rivers.next_vertex,
        SIDE,
        SIDE,
        distance,
        tangent,
    )
    np.testing.assert_allclose(distance, [2.0, 0.0])
    np.testing.assert_allclose(tangent, [[1.0, 0.0], [1.0, 0.0]])


def test_minimum_image_scalar_by_hand() -> None:
    assert minimum_image_scalar_py(90.0, 100.0) == -10.0
    assert minimum_image_scalar_py(-90.0, 100.0) == 10.0
    assert minimum_image_scalar_py(50.0, 100.0) == -50.0
    assert minimum_image_scalar_py(-50.0, 100.0) == -50.0


def test_no_rivers() -> None:
    empty = Rivers(
        vertices=np.zeros((0, 2)),
        next_vertex=np.zeros(0, dtype=np.int64),
        prev_vertex=np.zeros(0, dtype=np.int64),
        samples=np.zeros((0, 2)),
        width=8.0,
        v_corr_max=0.3,
        size=(SIDE, SIDE),
    )
    distance, _ = empty.centerline(np.array([[1.0, 1.0]]))
    assert distance[0] == math.inf
    np.testing.assert_array_equal(empty.current(np.array([[1.0, 1.0]])), [[0.0, 0.0]])
    assert empty.sample_tree.n == 0


@pytest.fixture(scope="module")
def generated(config: GenomeConfig) -> Rivers:
    return generate_rivers(config.water, config.world, continuous_world_streams(42).world)


def test_generation_is_deterministic(config: GenomeConfig, generated: Rivers) -> None:
    again = generate_rivers(config.water, config.world, continuous_world_streams(42).world)
    np.testing.assert_array_equal(again.vertices, generated.vertices)
    other = generate_rivers(config.water, config.world, continuous_world_streams(43).world)
    assert not np.array_equal(other.vertices, generated.vertices)


def test_rivers_close_smoothly_at_the_seam(config: GenomeConfig, generated: Rivers) -> None:
    w, h = config.world.width, config.world.height
    step = displacement(generated.vertices, generated.vertices[generated.next_vertex], w, h)
    lengths = np.hypot(step[:, 0], step[:, 1])
    # Every segment, including the one across the seam, is short: no jump.
    assert lengths.max() < 3 * config.water.river_vertex_step
    assert generated.count_vertices == config.world.n_rivers * 256
    assert np.all((generated.vertices >= 0) & (generated.vertices < 256))


def test_samples_are_spaced_by_s_rio(config: GenomeConfig, generated: Rivers) -> None:
    w, h = config.world.width, config.world.height
    step = displacement(generated.samples[:-1], generated.samples[1:], w, h)
    lengths = np.hypot(step[:, 0], step[:, 1])
    # Within one river consecutive samples are ≤ s_rio apart (chords of arcs of s_rio).
    within = lengths < 2 * config.water.s_rio
    assert np.all(lengths[within] <= config.water.s_rio + 1e-9)
    assert np.median(lengths[within]) > 0.95 * config.water.s_rio


def brute_force_distance(rivers: Rivers, point: np.ndarray) -> float:
    w, h = rivers.size
    a = rivers.vertices
    seg = displacement(a, rivers.vertices[rivers.next_vertex], w, h)
    rel = displacement(a, np.broadcast_to(point, a.shape), w, h)
    length2 = np.einsum("ij,ij->i", seg, seg)
    t = np.clip(np.einsum("ij,ij->i", rel, seg) / length2, 0.0, 1.0)
    off = rel - t[:, None] * seg
    return float(np.hypot(off[:, 0], off[:, 1]).min())


@given(
    x=st.floats(0, 255.999, allow_nan=False),
    y=st.floats(0, 255.999, allow_nan=False),
)
def test_nearest_segment_matches_brute_force_near_rivers(
    generated: Rivers, x: float, y: float
) -> None:
    point = np.array([x, y])
    distance, _ = generated.centerline(point[None, :])
    exact = brute_force_distance(generated, point)
    # ADR 0004: exact wherever it matters (inside rivers and within irrigation reach).
    if exact < 20.0:
        assert math.isclose(distance[0], exact, abs_tol=1e-9)
    assert distance[0] >= exact - 1e-9
