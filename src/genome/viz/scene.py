"""What one frame shows, computed from a snapshot without any graphics library (ARQUITETURA §2.1).

The input is the mapping of named arrays that ``GenomeParallelEnv.state`` returns. Nothing
here writes to it, so the viewer cannot change the world.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.state import active_slots
from genome.core.world.climate import light
from genome.core.world.lakes import water_radius

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ByteArray = npt.NDArray[np.uint8]
WorldArrays = Mapping[str, npt.NDArray[np.generic]]

#: Saturation and value of the lineage colours, whose hue is the colour gene (§5.1).
LINEAGE_SATURATION = 0.85
LINEAGE_VALUE = 1.0
HUE_SECTORS = 6
MAX_CHANNEL = 255


@dataclass(frozen=True, slots=True)
class Frame:
    """The world as one frame draws it, in world units.

    Attributes:
        tick: The next tick to run.
        size: ``(W, H)`` of the world.
        light: Light L(t) (§3.6).
        raining: Rain state of the current tick.
        humidity: Air humidity U.
        river_points: Centre lines of the rivers, shape ``(n, 2)``.
        river_radius: Half of the width of a river.
        lake_centers, lake_radii: Lakes, at their current sizes (§3.3).
        puddle_centers, puddle_radii: Puddles that hold water (§3.4).
        plant_positions: Live plants, shape ``(p, 2)``.
        plant_stacks: The stack ⌈b⌉ of each plant (§4.1).
        uids: Live creatures, in ``uid`` order.
        positions, headings, body_radii: Their bodies.
        colors: Their lineage colours, RGB bytes of shape ``(c, 3)``.
    """

    tick: int
    size: tuple[float, float]
    light: float
    raining: bool
    humidity: float
    river_points: FloatArray
    river_radius: float
    lake_centers: FloatArray
    lake_radii: FloatArray
    puddle_centers: FloatArray
    puddle_radii: FloatArray
    plant_positions: FloatArray
    plant_stacks: IntArray
    uids: IntArray
    positions: FloatArray
    headings: FloatArray
    body_radii: FloatArray
    colors: ByteArray


def hue_to_rgb(hue: FloatArray) -> ByteArray:
    """The colour of each gene value in ``[0, 1)``, used as an HSV hue (§5.1).

    Returns:
        RGB bytes, shape ``(n, 3)``.
    """
    sector = np.mod(hue, 1.0) * HUE_SECTORS
    # Standard HSV conversion: each channel follows the same ramp, shifted by a third of the circle.
    offsets = np.array([5.0, 3.0, 1.0])
    k = np.mod(sector[:, np.newaxis] + offsets, HUE_SECTORS)
    ramp = np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)
    channels = LINEAGE_VALUE * (1.0 - LINEAGE_SATURATION * ramp)
    rgb: ByteArray = np.round(channels * MAX_CHANNEL).astype(np.uint8)
    return rgb


def live_slots(arrays: WorldArrays) -> IntArray:
    """Slots of the live creatures of a snapshot, in ``uid`` order (§2.5)."""
    return active_slots(arrays["creatures.alive"], arrays["creatures.ids"])


def build_frame(arrays: WorldArrays, config: GenomeConfig) -> Frame:
    """Read one frame from a snapshot (§2.1)."""
    tick = int(arrays["tick"][0])
    water = config.water
    rain = arrays["climate.rain"]
    lake_volume = _floats(arrays, "lakes.volume")
    puddle_volume = _floats(arrays, "puddles.volume")
    wet = puddle_volume > 0.0
    plants = np.flatnonzero(arrays["plants.alive"])
    slots = live_slots(arrays)
    return Frame(
        tick=tick,
        size=(float(config.world.width), float(config.world.height)),
        light=light(tick, config.climate),
        # The rain of the next tick may not be drawn yet (§2.5): show the latest one known.
        raining=bool(rain[min(tick, rain.size - 1)]),
        humidity=float(arrays["climate.humidity"][0]),
        river_points=_floats(arrays, "rivers.vertices"),
        river_radius=water.w_rio / 2.0,
        lake_centers=_floats(arrays, "lakes.centers"),
        lake_radii=water_radius(
            water.k_r_max * np.sqrt(_floats(arrays, "lakes.v_max")),
            lake_volume,
            _floats(arrays, "lakes.v_max"),
        ),
        puddle_centers=_floats(arrays, "puddles.sites")[wet],
        puddle_radii=water_radius(water.r_p_max, puddle_volume[wet], water.v_p),
        plant_positions=_floats(arrays, "plants.position")[plants],
        plant_stacks=np.ceil(_floats(arrays, "plants.biomass")[plants]).astype(np.int64),
        uids=arrays["creatures.ids"][slots].astype(np.int64),
        positions=_floats(arrays, "creatures.position")[slots],
        headings=_floats(arrays, "creatures.heading")[slots],
        body_radii=_floats(arrays, "creatures.body_radius")[slots],
        colors=hue_to_rgb(_floats(arrays, "creatures.color")[slots]),
    )


def _floats(arrays: WorldArrays, name: str) -> FloatArray:
    return np.asarray(arrays[name], dtype=np.float64)
