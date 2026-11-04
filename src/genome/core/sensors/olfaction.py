"""Olfaction: smoothed fields of food and water, and two nostrils (ARQUITETURA §5.3, ADR 0005).

Plant biomass enters whole in the cell that holds the plant. Water coverage is
the mean, over regular sub-points of each cell, of the largest weight among the
water bodies that cover the point. The value of a cell sits at its centre.
"""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from genome.core.config import GenomeConfig, SensorConfig
from genome.core.kernels import jit
from genome.core.sensors.pairs import CreatureView
from genome.core.space import wrap_positions
from genome.core.state import PlantState
from genome.core.state_hash import NamedArray
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]

#: Olfactory inputs: intensity and direction of food, then of water (§5.5).
OLFACTORY_INPUTS = 4


def stamp_disk_py(
    grid: FloatArray, cx: float, cy: float, radius: float, weight: float, step: float
) -> None:
    """Raise to ``weight`` every sub-point of ``grid`` within ``radius``, on the torus.

    Only the window around the disk is visited, so the cost follows the area of
    the water body and not the size of the world (ADR 0005).
    """
    nx, ny = grid.shape
    radius_sq = radius * radius
    for ix in range(int(np.floor((cx - radius) / step)), int(np.ceil((cx + radius) / step)) + 1):
        dx = (ix + 0.5) * step - cx
        for iy in range(
            int(np.floor((cy - radius) / step)), int(np.ceil((cy + radius) / step)) + 1
        ):
            dy = (iy + 0.5) * step - cy
            if dx * dx + dy * dy < radius_sq and weight > grid[ix % nx, iy % ny]:
                grid[ix % nx, iy % ny] = weight


def bilinear_py(field: FloatArray, points: FloatArray, cell: float) -> FloatArray:
    """Read ``field`` at ``points`` by bilinear interpolation between cell centres (§5.3)."""
    nx, ny = field.shape
    out = np.empty(points.shape[0], dtype=np.float64)
    for k in range(points.shape[0]):
        u = points[k, 0] / cell - 0.5
        v = points[k, 1] / cell - 0.5
        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        fu = u - i0
        fv = v - j0
        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
        i0 = i0 % nx
        j0 = j0 % ny
        low = field[i0, j0] * (1.0 - fu) + field[i1, j0] * fu
        high = field[i0, j1] * (1.0 - fu) + field[i1, j1] * fu
        out[k] = low * (1.0 - fv) + high * fv
    return out


stamp_disk = jit(stamp_disk_py)
bilinear = jit(bilinear_py)


class SmellFields:
    """Smoothed fields of one world, recomputed every ``T_olf`` ticks (§5.3).

    Attributes:
        food: Smoothed plant biomass per cell.
        water: Smoothed fraction of each cell covered by water.
    """

    def __init__(self, water: Water, config: GenomeConfig) -> None:
        """Rasterise the static rivers once; the fields start at zero until ``update``."""
        sensors, world = config.sensors, config.world
        self._cell = sensors.cell_olf
        self._sigma = sensors.sigma_olf
        self._subsamples = sensors.olf_water_subsamples
        self._nx = round(world.width / sensors.cell_olf)
        self._ny = round(world.height / sensors.cell_olf)
        self._step = sensors.cell_olf / sensors.olf_water_subsamples
        self._river_cover = self._rasterise_rivers(water)
        self.food: FloatArray = np.zeros((self._nx, self._ny), dtype=np.float64)
        self.water: FloatArray = np.zeros((self._nx, self._ny), dtype=np.float64)

    def update(self, plants: PlantState, water: Water) -> None:
        """Recompute both fields from the current plants and water (§5.3)."""
        self.food = self._smooth(self.food_deposit(plants))
        self.water = self._smooth(self.water_coverage(water))

    def food_deposit(self, plants: PlantState) -> FloatArray:
        """Biomass of the plants deposited in each cell, before smoothing (§5.3)."""
        slots = plants.pool.active()
        cells = np.floor(plants.position[slots] / self._cell).astype(np.int64)
        index = cells[:, 0] % self._nx * self._ny + cells[:, 1] % self._ny
        deposit = np.bincount(index, weights=plants.biomass[slots], minlength=self._nx * self._ny)
        return deposit.reshape(self._nx, self._ny)

    def water_coverage(self, water: Water) -> FloatArray:
        """Fraction of each cell covered by water, weighted by fill, before smoothing (§5.3)."""
        grid = self._river_cover.copy()
        for bodies in (
            (water.lakes.centers, water.lakes.radius(), water.lakes.fill()),
            (water.puddles.sites, water.puddles.radius(), water.puddles.fill()),
        ):
            self._stamp(grid, *bodies)
        s = self._subsamples
        by_cell = grid.reshape(self._nx, s, self._ny, s)
        coverage: FloatArray = by_cell.mean(axis=(1, 3))
        return coverage

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the fields, for ``state_hash``."""
        return [("smell.food", self.food), ("smell.water", self.water)]

    def _stamp(
        self, grid: FloatArray, centers: FloatArray, radii: FloatArray, fill: FloatArray
    ) -> None:
        for i in np.flatnonzero(fill > 0.0):
            stamp_disk(
                grid,
                float(centers[i, 0]),
                float(centers[i, 1]),
                float(radii[i]),
                float(fill[i]),
                self._step,
            )

    def _rasterise_rivers(self, water: Water) -> FloatArray:
        nx, ny = self._nx * self._subsamples, self._ny * self._subsamples
        x = (np.arange(nx, dtype=np.float64) + 0.5) * self._step
        y = (np.arange(ny, dtype=np.float64) + 0.5) * self._step
        points = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(nx * ny, 2)
        return water.rivers.contains(points).astype(np.float64).reshape(nx, ny)

    def _smooth(self, field: FloatArray) -> FloatArray:
        """Gaussian filter with toroidal borders (§5.3)."""
        smoothed: FloatArray = gaussian_filter(field, sigma=self._sigma, mode="wrap")
        return smoothed


def _nostril_positions(view: CreatureView, side: float, sensors: SensorConfig) -> FloatArray:
    """Nostrils at ``δ = 2·r_c + 1`` ahead, at ``side · alpha_olf`` from the heading (§5.3)."""
    reach = sensors.nostril_body_factor * view.body_radius + sensors.nostril_offset
    angle = view.heading + side * sensors.alpha_olf_rad
    offset = np.column_stack((reach * np.cos(angle), reach * np.sin(angle)))
    return wrap_positions(view.position + offset, view.size[0], view.size[1])


def _read_field(
    field: FloatArray,
    left: FloatArray,
    right: FloatArray,
    k_olf: float,
    sensors: SensorConfig,
) -> tuple[FloatArray, FloatArray]:
    """Intensity and direction of a field; positive direction means more to the left (§5.3)."""
    c_left = bilinear(field, left, sensors.cell_olf)
    c_right = bilinear(field, right, sensors.cell_olf)
    total = c_left + c_right
    return 1.0 - np.exp(-k_olf * total / 2.0), (c_left - c_right) / (total + sensors.eps_olf)


def olfaction_channels(
    view: CreatureView, fields: SmellFields, sensors: SensorConfig
) -> FloatArray:
    """The 4 olfactory inputs, shape ``(n, OLFACTORY_INPUTS)``, in the order of §5.5."""
    # Angles grow to the right (§1.3), so the left nostril is at −alpha_olf.
    left = _nostril_positions(view, -1.0, sensors)
    right = _nostril_positions(view, 1.0, sensors)
    food = _read_field(fields.food, left, right, sensors.k_olf_food, sensors)
    drink = _read_field(fields.water, left, right, sensors.k_olf_water, sensors)
    return np.column_stack((*food, *drink))
