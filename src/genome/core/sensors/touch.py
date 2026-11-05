"""Touch: food, water and partner within reach (ARQUITETURA §5.4)."""

import numpy as np
import numpy.typing as npt

from genome.core.kernels import jit
from genome.core.sensors.pairs import Scene
from genome.core.sensors.vision import color_similarity
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Touch inputs: food, water and partner (§5.5).
TOUCH_INPUTS = 3
FOOD, WATER, PARTNER = 0, 1, 2


def mark_touch_py(
    out: FloatArray,
    plant_observer: IntArray,
    plant_distance: FloatArray,
    other_observer: IntArray,
    other_target: IntArray,
    other_distance: FloatArray,
    touch_radius: FloatArray,
    body_radius: FloatArray,
    age: IntArray,
    t_mat: FloatArray,
    color: FloatArray,
    s_min: float,
) -> None:
    """Mark the rows that touch a plant or a mature partner of a similar colour (§5.4)."""
    for p in range(plant_observer.size):
        if plant_distance[p] <= touch_radius[plant_observer[p]]:
            out[plant_observer[p], FOOD] = 1.0
    for p in range(other_observer.size):
        a, b = other_observer[p], other_target[p]
        if (
            other_distance[p] <= touch_radius[a] + body_radius[b]
            and age[b] >= t_mat[b]
            and color_similarity(color[a], color[b]) >= s_min
        ):
            out[a, PARTNER] = 1.0


mark_touch = jit(mark_touch_py)


def touch_channels(scene: Scene, water: Water, s_min: float) -> FloatArray:
    """The 3 touch inputs, each 0 or 1, shape ``(n, TOUCH_INPUTS)`` (§5.4)."""
    view = scene.view
    out = np.zeros((view.count, TOUCH_INPUTS), dtype=np.float64)
    plants, others = scene.plant_pairs, scene.creature_pairs
    mark_touch(
        out,
        plants.observer,
        plants.distance,
        others.observer,
        others.target,
        others.distance,
        view.touch_radius,
        view.body_radius,
        view.age,
        view.t_mat,
        view.color,
        s_min,
    )
    # The margin distance is zero inside water, so this also covers being in it.
    out[water.margin_distance(view.position) <= view.touch_radius, WATER] = 1.0
    return out
