"""Rivers: periodic meander geometry, current field and perception samples (ARQUITETURA §3.2).

Generation follows ADR 0004: each river runs along one axis of the torus and
its perpendicular offset is a sum of sines whose periods divide the side, so the
polyline closes exactly at the seam. Vertex order is the direction of the flow.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from genome.core.config import WaterConfig, WorldConfig
from genome.core.space import displacement, wrap_positions

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

HORIZONTAL = 0
# Vertices whose adjacent segments are tested per query point. More than one,
# so that near a crossing the vertices of both rivers are considered (ADR 0004).
NEAREST_VERTICES = 4


@dataclass(frozen=True, eq=False)
class Rivers:
    """Static river network of one world (§3.2).

    Attributes:
        vertices: Centre-line vertices of every river, shape ``(V, 2)``.
        next_vertex: Index of the following vertex of the same (closed) river.
        prev_vertex: Index of the preceding vertex of the same river.
        samples: Perception points every ``s_rio`` along the centre line, shape ``(P, 2)``.
        width: River width ``w_rio``.
        v_corr_max: Current at the centre line.
        size: Torus sides ``(W, H)``.
    """

    vertices: FloatArray
    next_vertex: IntArray
    prev_vertex: IntArray
    samples: FloatArray
    width: float
    v_corr_max: float
    size: tuple[float, float]
    _vertex_tree: cKDTree = field(init=False, repr=False)
    sample_tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the static spatial trees (§12.5)."""
        box = np.array(self.size)
        object.__setattr__(self, "_vertex_tree", _tree(self.vertices, box))
        object.__setattr__(self, "sample_tree", _tree(self.samples, box))

    @property
    def count_vertices(self) -> int:
        """Number of centre-line vertices of all rivers."""
        return int(self.vertices.shape[0])

    def centerline(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Distance d_c to the nearest centre line and the flow tangent t̂ there (§3.2, ADR 0004).

        Returns:
            ``(distance, tangent)`` with shapes ``(n,)`` and ``(n, 2)``. Without
            rivers the distance is infinite and the tangent zero.
        """
        n = points.shape[0]
        if self.count_vertices == 0:
            return np.full(n, np.inf), np.zeros((n, 2))
        k = min(NEAREST_VERTICES, self.count_vertices)
        _, nearest = self._vertex_tree.query(points, k=k)
        nearest = np.asarray(nearest, dtype=np.int64).reshape(n, k)
        starts = np.concatenate((self.prev_vertex[nearest], nearest), axis=1)
        ends = np.concatenate((nearest, self.next_vertex[nearest]), axis=1)
        repeated = np.repeat(points, 2 * k, axis=0)
        d, t = self._segment_distance(repeated, starts.ravel(), ends.ravel())
        d, t = d.reshape(n, 2 * k), t.reshape(n, 2 * k, 2)
        # First minimum along the fixed candidate order: deterministic ties.
        best = np.argmin(d, axis=1)
        rows = np.arange(n)
        return d[rows, best], t[rows, best]

    def current(self, points: FloatArray) -> FloatArray:
        """Current v_corr = v_corr_max · (1 − (2 d_c / w_rio)²) · t̂ inside rivers (§3.2)."""
        distance, tangent = self.centerline(points)
        profile = 1.0 - (2.0 * distance / self.width) ** 2
        strength = np.where(distance < self.width / 2.0, self.v_corr_max * profile, 0.0)
        return strength[:, None] * tangent

    def margin_distance(self, points: FloatArray) -> FloatArray:
        """Distance to the nearest river bank, zero inside a river (§3.2)."""
        distance, _ = self.centerline(points)
        return np.maximum(0.0, distance - self.width / 2.0)

    def contains(self, points: FloatArray) -> BoolArray:
        """Whether each point lies inside a river (§3.2, §4.4)."""
        distance, _ = self.centerline(points)
        return distance < self.width / 2.0

    def _segment_distance(
        self, points: FloatArray, start: IntArray, end: IntArray
    ) -> tuple[FloatArray, FloatArray]:
        w, h = self.size
        a = self.vertices[start]
        seg = displacement(a, self.vertices[end], w, h)
        length = np.hypot(seg[:, 0], seg[:, 1])
        unit = seg / length[:, None]
        rel = displacement(a, points, w, h)
        along = np.clip(np.einsum("ij,ij->i", rel, unit), 0.0, length)
        offset = rel - along[:, None] * unit
        return np.hypot(offset[:, 0], offset[:, 1]), unit


def generate_rivers(water: WaterConfig, world: WorldConfig, rng: np.random.Generator) -> Rivers:
    """Draw the rivers of a world from the world stream (§3.2, ADR 0004).

    Draw order per river: axis, flow direction, perpendicular offset, then the
    amplitudes and the phases of the harmonics.
    """
    polylines = [_draw_polyline(water, world, rng) for _ in range(world.n_rivers)]
    vertices, next_vertex, prev_vertex = _concatenate(polylines)
    samples = _sample_centerlines(polylines, water.s_rio, world)
    return Rivers(
        vertices=vertices,
        next_vertex=next_vertex,
        prev_vertex=prev_vertex,
        samples=samples,
        width=water.w_rio,
        v_corr_max=water.v_corr_max,
        size=(world.width, world.height),
    )


def _draw_polyline(water: WaterConfig, world: WorldConfig, rng: np.random.Generator) -> FloatArray:
    axis = int(rng.integers(2))
    flow = int(rng.integers(2))
    along_side, across_side = (
        (world.width, world.height) if axis == HORIZONTAL else (world.height, world.width)
    )
    offset = rng.uniform(0.0, across_side)
    harmonics = np.arange(1, water.river_harmonics + 1, dtype=np.float64)
    amplitudes = rng.uniform(0.0, water.river_amplitude / harmonics)
    phases = rng.uniform(0.0, 2.0 * math.pi, harmonics.size)
    n_vertices = max(1, round(along_side / water.river_vertex_step))
    s = np.arange(n_vertices, dtype=np.float64) * (along_side / n_vertices)
    waves = np.sin(2.0 * math.pi * np.outer(s, harmonics) / along_side + phases)
    across = offset + waves @ amplitudes
    points = np.column_stack((s, across) if axis == HORIZONTAL else (across, s))
    if flow == 1:
        points = points[::-1]
    return wrap_positions(points, world.width, world.height)


def _concatenate(polylines: list[FloatArray]) -> tuple[FloatArray, IntArray, IntArray]:
    if not polylines:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 2)), empty, empty
    next_parts, prev_parts, start = [], [], 0
    for line in polylines:
        index = np.arange(start, start + line.shape[0], dtype=np.int64)
        next_parts.append(np.roll(index, -1))
        prev_parts.append(np.roll(index, 1))
        start += line.shape[0]
    return np.concatenate(polylines), np.concatenate(next_parts), np.concatenate(prev_parts)


def _sample_centerlines(polylines: list[FloatArray], step: float, world: WorldConfig) -> FloatArray:
    """Points every ``step`` of arc length along each closed centre line (§3.2)."""
    parts = [np.zeros((0, 2))]
    for line in polylines:
        seg = displacement(line, np.roll(line, -1, axis=0), world.width, world.height)
        lengths = np.hypot(seg[:, 0], seg[:, 1])
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        arc = np.arange(0.0, cumulative[-1], step)
        index = np.searchsorted(cumulative, arc, side="right") - 1
        fraction = (arc - cumulative[index]) / lengths[index]
        points = line[index] + fraction[:, None] * seg[index]
        parts.append(wrap_positions(points, world.width, world.height))
    return np.concatenate(parts)


def _tree(points: FloatArray, box: FloatArray) -> cKDTree:
    return cKDTree(points if points.size else np.zeros((0, 2)), boxsize=box)
