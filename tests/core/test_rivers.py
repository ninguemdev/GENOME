import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from genome.core.config import GenomeConfig
from genome.core.rng import continuous_world_streams
from genome.core.space import displacement
from genome.core.world.rivers import Rivers, generate_rivers

SIDE = 100.0


def straight_river(width: float = 8.0, v_corr_max: float = 0.3) -> Rivers:
    """A horizontal river along y = 50, flowing toward +x, on a 100 × 100 torus."""
    x = np.arange(100, dtype=np.float64)
    vertices = np.column_stack((x, np.full(100, 50.0)))
    index = np.arange(100, dtype=np.int64)
    return Rivers(
        vertices=vertices,
        next_vertex=np.roll(index, -1),
        prev_vertex=np.roll(index, 1),
        samples=vertices[::4],
        width=width,
        v_corr_max=v_corr_max,
        size=(SIDE, SIDE),
    )


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
