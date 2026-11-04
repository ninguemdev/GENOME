"""Vision: dynamic radius, six-receptor retina and four channels (ARQUITETURA §5.2, ADR 0005)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig, SensorConfig
from genome.core.kernels import jit
from genome.core.sensors.pairs import CreatureView, Pairs, Scene, find_pairs
from genome.core.space import TWO_PI, displacement
from genome.core.world.rivers import Rivers
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Receptors per visual channel (§5.2).
RECEPTORS = 6
#: Visual channels: food, water, density and affinity (§5.5).
CHANNELS = 4


def max_by_row_py(out: FloatArray, rows: IntArray, values: FloatArray) -> None:
    """Raise each receptor of ``out[rows[i]]`` to ``values[i]``; independent of pair order."""
    for i in range(rows.size):
        for j in range(values.shape[1]):
            if values[i, j] > out[rows[i], j]:
                out[rows[i], j] = values[i, j]


def add_by_row_py(out: FloatArray, rows: IntArray, values: FloatArray) -> None:
    """Add ``values[i]`` to each receptor of ``out[rows[i]]``, in the given order."""
    for i in range(rows.size):
        for j in range(values.shape[1]):
            out[rows[i], j] += values[i, j]


def retina_signals_py(
    observer: IntArray,
    dx: FloatArray,
    dy: FloatArray,
    distance: FloatArray,
    heading: FloatArray,
    view_limits: FloatArray,
    mu: FloatArray,
    sigma: FloatArray,
) -> tuple[IntArray, FloatArray]:
    """``f(d) · g_j(θ)`` of the pairs in sight, rejecting the others early (§5.2).

    Args:
        observer: Row of the observer of each pair.
        dx: First component of the vector from observer to target.
        dy: Second component of that vector.
        distance: Distance used by the falloff; the bank's distance for water.
        heading: Orientation φ of each row.
        view_limits: ``(radius, fov, edge)``, the reach of the tick and the angles.
        mu: Centre of each receptor, in radians.
        sigma: Width of each receptor, in radians.

    Returns:
        ``(kept, signals)``: index of each pair in sight, and its signal per receptor.
    """
    radius, fov, edge = view_limits[0], view_limits[1], view_limits[2]
    kept = np.empty(observer.size, dtype=np.int64)
    signals = np.empty((observer.size, mu.size), dtype=np.float64)
    count = 0
    for p in range(observer.size):
        if distance[p] >= radius:
            continue
        if dx[p] == 0.0 and dy[p] == 0.0:
            # A zero vector has no direction; it counts as straight ahead (ADR 0005).
            theta = 0.0
        else:
            # Same wrap as space.wrap_angle, including its rounding fix.
            wrapped = (np.arctan2(dy[p], dx[p]) - heading[observer[p]] + np.pi) % TWO_PI
            theta = (0.0 if wrapped >= TWO_PI else wrapped) - np.pi
        if abs(theta) > fov:
            continue
        scale = min(1.0, (fov - abs(theta)) / edge) * (1.0 - distance[p] / radius)
        for j in range(mu.size):
            offset = (theta - mu[j]) / sigma[j]
            signals[count, j] = scale * np.exp(-0.5 * offset * offset)
        kept[count] = p
        count += 1
    return kept[:count], signals[:count]


max_by_row = jit(max_by_row_py)
add_by_row = jit(add_by_row_py)
retina_signals = jit(retina_signals_py)


@dataclass(frozen=True, slots=True)
class Retina:
    """Receptor layout and the moment's reach (§5.2).

    Attributes:
        mu, sigma: Centre and width of each receptor, in radians.
        fov: Half field of view, in radians.
        edge: Width of the window that fades receptors at the border, in radians.
        radius: The vision radius of this tick.
    """

    mu: FloatArray
    sigma: FloatArray
    fov: float
    edge: float
    radius: float


def vision_radius(light: float, sensors: SensorConfig) -> float:
    """r_vis(t) = r_vis · (m_noite + (1 − m_noite) · L(t)) (§5.2)."""
    return sensors.r_vis * (sensors.m_noite + (1.0 - sensors.m_noite) * light)


def make_retina(light: float, sensors: SensorConfig) -> Retina:
    """The retina of a tick with light ``light`` (§5.2)."""
    return Retina(
        mu=np.array(sensors.receptor_mu_rad, dtype=np.float64),
        sigma=np.array(sensors.receptor_sigma_rad, dtype=np.float64),
        fov=sensors.fov_half_rad,
        edge=sensors.edge_window_rad,
        radius=vision_radius(light, sensors),
    )


def color_similarity(a: FloatArray, b: FloatArray) -> FloatArray:
    """sim(c₁, c₂) = 1 − 2·Δc, with Δc the circular distance between colours (§5.2)."""
    delta = np.abs(a - b)
    return 1.0 - 2.0 * np.minimum(delta, 1.0 - delta)


def vision_channels(scene: Scene, water: Water, light: float, config: GenomeConfig) -> FloatArray:
    """The 24 visual inputs: food, water, density and affinity (§5.2, §5.5).

    Returns:
        Shape ``(n, CHANNELS * RECEPTORS)``, in the order of §5.5.
    """
    view = scene.view
    retina = make_retina(light, config.sensors)
    food = np.zeros((view.count, RECEPTORS))
    _raise_channel(food, view, scene.plant_pairs, _plant_weights(scene, config), retina)
    density, affinity = _creature_channels(scene, retina)
    drink = _water_channel(view, water, retina)
    return np.concatenate((food, drink, density, affinity), axis=1)


def _plant_weights(scene: Scene, config: GenomeConfig) -> FloatArray:
    weights: FloatArray = scene.plant_biomass[scene.plant_pairs.target] / config.plants.b_max
    return weights


def _signals(view: CreatureView, pairs: Pairs, retina: Retina) -> tuple[IntArray, FloatArray]:
    """Index of the pairs in sight and their ``f(d) · g_j(θ)`` (§5.2)."""
    limits = np.array((retina.radius, retina.fov, retina.edge), dtype=np.float64)
    kept, signals = retina_signals(
        pairs.observer,
        pairs.dx,
        pairs.dy,
        pairs.distance,
        view.heading,
        limits,
        retina.mu,
        retina.sigma,
    )
    return kept, signals


def _raise_channel(
    out: FloatArray, view: CreatureView, pairs: Pairs, weights: FloatArray, retina: Retina
) -> None:
    """Raise ``out`` to the best ``W · f(d) · g_j(θ)`` per receptor (§5.2)."""
    kept, signals = _signals(view, pairs, retina)
    max_by_row(out, pairs.observer[kept], weights[kept, None] * signals)


def _creature_channels(scene: Scene, retina: Retina) -> tuple[FloatArray, FloatArray]:
    """Saturated density and best affinity; the pairs already exclude self (§5.2)."""
    view, pairs = scene.view, scene.creature_pairs
    kept, signals = _signals(view, pairs, retina)
    rows = pairs.observer[kept]
    total = np.zeros((view.count, RECEPTORS))
    add_by_row(total, rows, signals)
    similarity = color_similarity(view.color[rows], view.color[pairs.target[kept]])
    affinity = np.zeros((view.count, RECEPTORS))
    max_by_row(affinity, rows, similarity[:, None] * signals)
    return 1.0 - np.exp(-total), affinity


def _water_channel(view: CreatureView, water: Water, retina: Retina) -> FloatArray:
    """Best source per receptor among river samples, lakes and puddles (§5.2)."""
    out = np.zeros((view.count, RECEPTORS))
    sources = (
        _river_pairs(view, water.rivers, retina),
        _disk_pairs(view, water.lakes.centers, water.lakes.radius(), water.lakes.fill(), retina),
        _disk_pairs(
            view, water.puddles.sites, water.puddles.radius(), water.puddles.fill(), retina
        ),
    )
    for pairs, weights in sources:
        _raise_channel(out, view, pairs, weights, retina)
    return out


def _river_pairs(view: CreatureView, rivers: Rivers, retina: Retina) -> tuple[Pairs, FloatArray]:
    """River samples seen at their distance to the bank, with weight one (§3.2, §5.2)."""
    half_width = rivers.width / 2.0
    pairs = find_pairs(view, rivers.sample_tree, rivers.samples, retina.radius + half_width)
    margin = np.maximum(0.0, pairs.distance - half_width)
    banked = Pairs(pairs.observer, pairs.target, pairs.dx, pairs.dy, margin)
    return banked, np.ones(pairs.observer.size)


def _disk_pairs(
    view: CreatureView,
    centers: FloatArray,
    radii: FloatArray,
    fill: FloatArray,
    retina: Retina,
) -> tuple[Pairs, FloatArray]:
    """Lakes or puddles seen at ``max(0, d − r)``, towards their centre (§5.2).

    Bodies without water are dropped first: most puddles are dry most of the time.
    """
    wet = np.flatnonzero(fill > 0.0)
    width, height = view.size
    vector = displacement(view.position[:, None, :], centers[wet][None, :, :], width, height)
    margin = np.maximum(0.0, np.hypot(vector[..., 0], vector[..., 1]) - radii[wet])
    rows, bodies = np.nonzero(margin < retina.radius)
    observer = rows.astype(np.int64)
    body = bodies.astype(np.int64)
    pairs = Pairs(
        observer,
        wet[body],
        vector[observer, body, 0],
        vector[observer, body, 1],
        margin[observer, body],
    )
    return pairs, fill[wet[body]]
