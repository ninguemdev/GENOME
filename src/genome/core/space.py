"""Toroidal geometry: minimum image, distance and wrap-around (ARQUITETURA §3.1).

All functions are vectorised, broadcast over leading dimensions and work in
``float64``. Positions have shape ``(..., 2)``: the last axis holds ``(x, y)``.
"""

import math

import numpy as np
import numpy.typing as npt

from genome.core.kernels import jit

FloatArray = npt.NDArray[np.float64]

TWO_PI = 2.0 * math.pi


def wrap_coordinate(values: npt.ArrayLike, size: float) -> FloatArray:
    """Map values into ``[0, size)``, fixing the rounding case ``value == size``.

    Implements ARQUITETURA §3.1: ``x ← x mod W`` and, if ``x == W`` by rounding,
    ``x ← 0``. (For example, ``-1e-17 mod 256`` rounds to exactly ``256``.)

    Raises:
        ValueError: If ``size`` is not positive and finite.
    """
    _require_size(size)
    wrapped = np.mod(np.asarray(values, dtype=np.float64), size)
    return np.where(wrapped >= size, 0.0, wrapped)


def wrap_positions(positions: npt.ArrayLike, width: float, height: float) -> FloatArray:
    """Wrap positions of shape ``(..., 2)`` into ``[0, W) × [0, H)`` (§3.1, §6.3)."""
    p = _as_points(positions)
    out = np.empty_like(p)
    out[..., 0] = wrap_coordinate(p[..., 0], width)
    out[..., 1] = wrap_coordinate(p[..., 1], height)
    return out


def minimum_image(delta: npt.ArrayLike, size: float) -> FloatArray:
    """Shortest signed offset on a circle of length ``size``, in ``[-size/2, size/2)``.

    Implements ARQUITETURA §3.1: ``Δ = ((Δ_raw + size/2) mod size) − size/2``.
    """
    half = size / 2.0
    return wrap_coordinate(np.asarray(delta, dtype=np.float64) + half, size) - half


def displacement(a: npt.ArrayLike, b: npt.ArrayLike, width: float, height: float) -> FloatArray:
    """Minimum-image vector from ``a`` to ``b``, shape ``(..., 2)`` (§3.1)."""
    pa, pb = _as_points(a), _as_points(b)
    raw = pb - pa
    out = np.empty_like(raw)
    out[..., 0] = minimum_image(raw[..., 0], width)
    out[..., 1] = minimum_image(raw[..., 1], height)
    return out


def distance(a: npt.ArrayLike, b: npt.ArrayLike, width: float, height: float) -> FloatArray:
    """Toroidal Euclidean distance between ``a`` and ``b``, shape ``(...)`` (§3.1)."""
    d = displacement(a, b, width, height)
    return np.hypot(d[..., 0], d[..., 1])


def minimum_image_scalar_py(delta: float, size: float) -> float:
    """Minimum-image offset of one number (§3.1), bit-identical to ``minimum_image``."""
    half = size / 2.0
    wrapped = (delta + half) % size
    return (0.0 if wrapped >= size else wrapped) - half


minimum_image_scalar = jit(minimum_image_scalar_py)


def nearest_disk_margin_py(
    points: FloatArray,
    centers: FloatArray,
    radii: FloatArray,
    width: float,
    height: float,
    out: FloatArray,
) -> None:
    """Distance from each point to the nearest disk, zero inside one, infinite without disks.

    Implements ``min_j max(0, d(p, c_j) − r_j)`` with the toroidal distance of §3.1.
    """
    for i in range(points.shape[0]):
        best = np.inf
        for j in range(centers.shape[0]):
            dx = minimum_image_scalar(centers[j, 0] - points[i, 0], width)
            dy = minimum_image_scalar(centers[j, 1] - points[i, 1], height)
            margin = max(0.0, np.hypot(dx, dy) - radii[j])
            if margin < best:
                best = margin
        out[i] = best


nearest_disk_margin = jit(nearest_disk_margin_py)


def wrap_angle(angles: npt.ArrayLike) -> FloatArray:
    """Map angles into ``[-π, π)`` (§6.3), with the same rounding fix as positions."""
    shifted = np.asarray(angles, dtype=np.float64) + math.pi
    return wrap_coordinate(shifted, TWO_PI) - math.pi


def _as_points(points: npt.ArrayLike) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError(f"expected points with shape (..., 2), got {array.shape}")
    return array


def _require_size(size: float) -> None:
    if not (math.isfinite(size) and size > 0):
        raise ValueError(f"torus side must be positive and finite, got {size!r}")
