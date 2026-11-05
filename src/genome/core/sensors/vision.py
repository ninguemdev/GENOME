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
#: Scale of a target out of sight; a target in sight never has a negative one (§5.2).
OUT_OF_SIGHT = -1.0


def color_similarity_py(a: float, b: float) -> float:
    """sim(c₁, c₂) = 1 − 2·Δc, with Δc the circular distance between colours (§5.2)."""
    delta = abs(a - b)
    return 1.0 - 2.0 * min(delta, 1.0 - delta)


color_similarity = jit(color_similarity_py)


def sight_py(
    dx: float, dy: float, distance: float, heading: float, view_limits: FloatArray
) -> tuple[float, float]:
    """Angle θ of a target and its ``w(θ) · f(d)``, rejecting it early when out of sight (§5.2).

    Args:
        dx: First component of the vector from observer to target.
        dy: Second component of that vector.
        distance: Distance used by the falloff; the bank's distance for water.
        heading: Orientation φ of the observer.
        view_limits: ``(radius, fov, edge)``, the reach of the tick and the angles.

    Returns:
        ``(theta, scale)``; ``scale`` is ``OUT_OF_SIGHT`` for a target out of sight.
    """
    radius, fov, edge = view_limits[0], view_limits[1], view_limits[2]
    if distance >= radius:
        return 0.0, OUT_OF_SIGHT
    if dx == 0.0 and dy == 0.0:
        # A zero vector has no direction; it counts as straight ahead (ADR 0005).
        theta = 0.0
    else:
        # Same wrap as space.wrap_angle, including its rounding fix.
        wrapped = (np.arctan2(dy, dx) - heading + np.pi) % TWO_PI
        theta = (0.0 if wrapped >= TWO_PI else wrapped) - np.pi
    if abs(theta) > fov:
        return 0.0, OUT_OF_SIGHT
    return theta, min(1.0, (fov - abs(theta)) / edge) * (1.0 - distance / radius)


sight = jit(sight_py)


def raise_retina_py(
    out: FloatArray,
    observer: IntArray,
    dx: FloatArray,
    dy: FloatArray,
    distance: FloatArray,
    weights: FloatArray,
    heading: FloatArray,
    view_limits: FloatArray,
    mu: FloatArray,
    sigma: FloatArray,
) -> None:
    """Raise ``out`` to the best ``W · f(d) · g_j(θ)`` per receptor; free of pair order (§5.2)."""
    for p in range(observer.size):
        row = observer[p]
        theta, scale = sight(dx[p], dy[p], distance[p], heading[row], view_limits)
        if scale == OUT_OF_SIGHT:
            continue
        for j in range(mu.size):
            offset = (theta - mu[j]) / sigma[j]
            value = weights[p] * (scale * np.exp(-0.5 * offset * offset))
            if value > out[row, j]:
                out[row, j] = value


def creature_retina_py(
    total: FloatArray,
    affinity: FloatArray,
    observer: IntArray,
    target: IntArray,
    dx: FloatArray,
    dy: FloatArray,
    distance: FloatArray,
    color: FloatArray,
    heading: FloatArray,
    view_limits: FloatArray,
    mu: FloatArray,
    sigma: FloatArray,
) -> None:
    """Sum of the signals per receptor, in pair order, and the best affinity (§2.5, §5.2)."""
    for p in range(observer.size):
        row = observer[p]
        theta, scale = sight(dx[p], dy[p], distance[p], heading[row], view_limits)
        if scale == OUT_OF_SIGHT:
            continue
        similarity = color_similarity(color[row], color[target[p]])
        for j in range(mu.size):
            offset = (theta - mu[j]) / sigma[j]
            signal = scale * np.exp(-0.5 * offset * offset)
            total[row, j] += signal
            value = similarity * signal
            if value > affinity[row, j]:
                affinity[row, j] = value


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
    kept = np.empty(observer.size, dtype=np.int64)
    signals = np.empty((observer.size, mu.size), dtype=np.float64)
    count = 0
    for p in range(observer.size):
        theta, scale = sight(dx[p], dy[p], distance[p], heading[observer[p]], view_limits)
        if scale == OUT_OF_SIGHT:
            continue
        for j in range(mu.size):
            offset = (theta - mu[j]) / sigma[j]
            signals[count, j] = scale * np.exp(-0.5 * offset * offset)
        kept[count] = p
        count += 1
    return kept[:count], signals[:count]


raise_retina = jit(raise_retina_py)
creature_retina = jit(creature_retina_py)
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


def make_retina(radius: float, sensors: SensorConfig) -> Retina:
    """The retina of a tick whose vision radius is ``radius`` (§5.2)."""
    return Retina(
        mu=np.array(sensors.receptor_mu_rad, dtype=np.float64),
        sigma=np.array(sensors.receptor_sigma_rad, dtype=np.float64),
        fov=sensors.fov_half_rad,
        edge=sensors.edge_window_rad,
        radius=radius,
    )


def vision_channels(scene: Scene, water: Water, config: GenomeConfig) -> FloatArray:
    """The 24 visual inputs: food, water, density and affinity (§5.2, §5.5).

    Returns:
        Shape ``(n, CHANNELS * RECEPTORS)``, in the order of §5.5.
    """
    view = scene.view
    retina = make_retina(scene.vision_radius, config.sensors)
    food = np.zeros((view.count, RECEPTORS))
    _raise_channel(food, view, scene.plant_pairs, _plant_weights(scene, config), retina)
    density, affinity = _creature_channels(scene, retina)
    drink = _water_channel(view, water, retina)
    return np.concatenate((food, drink, density, affinity), axis=1)


def _plant_weights(scene: Scene, config: GenomeConfig) -> FloatArray:
    """Weight W_i / W_max of the plant of each pair, b / b_max (§5.2)."""
    weights: FloatArray = scene.plant_biomass[scene.plant_pairs.target] / config.plants.b_max
    return weights


def _view_limits(retina: Retina) -> FloatArray:
    return np.array((retina.radius, retina.fov, retina.edge), dtype=np.float64)


def _raise_channel(
    out: FloatArray, view: CreatureView, pairs: Pairs, weights: FloatArray, retina: Retina
) -> None:
    """Raise ``out`` to the best ``W · f(d) · g_j(θ)`` per receptor (§5.2)."""
    raise_retina(
        out,
        pairs.observer,
        pairs.dx,
        pairs.dy,
        pairs.distance,
        weights,
        view.heading,
        _view_limits(retina),
        retina.mu,
        retina.sigma,
    )


def _creature_channels(scene: Scene, retina: Retina) -> tuple[FloatArray, FloatArray]:
    """Saturated density and best affinity; the pairs already exclude self (§5.2)."""
    view, pairs = scene.view, scene.creature_pairs
    total = np.zeros((view.count, RECEPTORS))
    affinity = np.zeros((view.count, RECEPTORS))
    creature_retina(
        total,
        affinity,
        pairs.observer,
        pairs.target,
        pairs.dx,
        pairs.dy,
        pairs.distance,
        view.color,
        view.heading,
        _view_limits(retina),
        retina.mu,
        retina.sigma,
    )
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
    pairs = find_pairs(view, rivers.samples, retina.radius + half_width, view.buffers.rivers)
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
