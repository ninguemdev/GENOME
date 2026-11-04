"""Touch: food, water and partner within reach (ARQUITETURA §5.4)."""

import numpy as np
import numpy.typing as npt

from genome.core.sensors.pairs import Scene
from genome.core.sensors.vision import color_similarity
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]

#: Touch inputs: food, water and partner (§5.5).
TOUCH_INPUTS = 3


def touch_channels(scene: Scene, water: Water, s_min: float) -> FloatArray:
    """The 3 touch inputs, each 0 or 1, shape ``(n, TOUCH_INPUTS)`` (§5.4)."""
    view = scene.view
    out = np.zeros((view.count, TOUCH_INPUTS), dtype=np.float64)
    plants = scene.plant_pairs
    out[plants.observer[plants.distance <= view.touch_radius[plants.observer]], 0] = 1.0
    # The margin distance is zero inside water, so this also covers being in it.
    out[water.margin_distance(view.position) <= view.touch_radius, 1] = 1.0
    others = scene.creature_pairs
    reach = view.touch_radius[others.observer] + view.body_radius[others.target]
    mature = view.age[others.target] >= view.t_mat[others.target]
    similar = color_similarity(view.color[others.observer], view.color[others.target]) >= s_min
    out[others.observer[(others.distance <= reach) & mature & similar], 2] = 1.0
    return out
