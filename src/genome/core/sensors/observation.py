"""The input vector of 42 values, ``sensor_model = "genome-v1"`` (ARQUITETURA §5.5).

Observations use the state and the climate of tick ``t`` (§2.4 step 1). Rows
follow ``uid`` order, as ``Brain.evaluate`` expects.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

from genome.core.config import CreatureConfig
from genome.core.environment import Environment
from genome.core.protocols import NUM_INPUTS
from genome.core.sensors.olfaction import olfaction_channels
from genome.core.sensors.pairs import CreatureView, build_scene, view_creatures
from genome.core.sensors.touch import touch_channels
from genome.core.sensors.vision import vision_channels, vision_radius
from genome.core.state import CreatureState

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# First index of each block of the layout (§5.5).
INPUT_VISION_FOOD: Final = 0
INPUT_VISION_WATER: Final = 6
INPUT_VISION_DENSITY: Final = 12
INPUT_VISION_AFFINITY: Final = 18
INPUT_SMELL_FOOD_INTENSITY: Final = 24
INPUT_SMELL_FOOD_DIRECTION: Final = 25
INPUT_SMELL_WATER_INTENSITY: Final = 26
INPUT_SMELL_WATER_DIRECTION: Final = 27
INPUT_TOUCH_FOOD: Final = 28
INPUT_TOUCH_WATER: Final = 29
INPUT_TOUCH_PARTNER: Final = 30
INPUT_ENERGY: Final = 31
INPUT_HYDRATION: Final = 32
INPUT_SPEED_FORWARD: Final = 33
INPUT_SPEED_LATERAL: Final = 34
INPUT_SPEED_ANGULAR: Final = 35
INPUT_MATURITY: Final = 36
INPUT_AGE: Final = 37
INPUT_MEMORY: Final = 38
INPUT_LIGHT: Final = 39
INPUT_RAIN: Final = 40
INPUT_HUMIDITY: Final = 41


def build_observations(
    state: CreatureState, environment: Environment, t: int
) -> tuple[FloatArray, IntArray]:
    """Compute the inputs of every live creature at tick ``t`` (§2.4 step 1, §5.5).

    Returns:
        ``(observations, slots)``: shape ``(n, NUM_INPUTS)``, and the slot of each row.
    """
    config = environment.config
    light = environment.climate.light(t)
    view = view_creatures(state, environment.size)
    scene = build_scene(view, environment.plants, vision_radius(light, config.sensors))
    observations = np.column_stack(
        (
            vision_channels(scene, environment.water, config),
            olfaction_channels(view, environment.smell, config.sensors),
            touch_channels(scene, environment.water, config.sensors.s_min),
            _internal_inputs(view, config.creatures),
            _climate_inputs(view.count, environment, t),
        )
    )
    return observations.reshape(view.count, NUM_INPUTS), view.slots


def _internal_inputs(view: CreatureView, creatures: CreatureConfig) -> FloatArray:
    """Reserves, proprioception, maturity, age and memory: inputs 31 to 38 (§5.5, §6.3)."""
    cos_phi, sin_phi = np.cos(view.heading), np.sin(view.heading)
    forward = view.velocity[:, 0] * cos_phi + view.velocity[:, 1] * sin_phi
    lateral = -view.velocity[:, 0] * sin_phi + view.velocity[:, 1] * cos_phi
    age = view.age.astype(np.float64)
    return np.column_stack(
        (
            view.energy / view.e_max,
            view.hydration / view.h_max,
            np.clip(forward / view.v_max, -1.0, 1.0),
            np.clip(lateral / view.v_max, -1.0, 1.0),
            np.clip(view.angular_velocity / creatures.omega_max, -1.0, 1.0),
            np.minimum(1.0, age / view.t_mat),
            # Sensors run before deaths (§2.4), so age may have just passed T_vida.
            np.minimum(1.0, age / view.t_vida),
            view.memory,
        )
    )


def _climate_inputs(n: int, environment: Environment, t: int) -> FloatArray:
    """Light, rain and humidity of tick ``t``: inputs 39 to 41 (§5.5)."""
    climate = environment.climate
    values = (climate.light(t), float(climate.is_raining(t)), climate.humidity)
    return np.tile(np.array(values, dtype=np.float64), (n, 1))
