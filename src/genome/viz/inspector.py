"""What the inspector shows of the selected creature (ARQUITETURA §2.1, §5.2, §5.3, §5.5).

The geometry comes from the functions the sensors themselves use, so the lobes and the
nostrils on screen are the ones the creature senses with. No graphics library here.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig, SensorConfig
from genome.core.protocols import NUM_INPUTS
from genome.core.sensors.olfaction import nostril_offsets
from genome.core.sensors.vision import make_retina, retina_signals, vision_radius
from genome.core.space import distance
from genome.viz.scene import Frame, WorldArrays, live_slots

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Points along each lobe: one per degree, so that every centre μ_j falls on a point (§5.2).
LOBE_SAMPLES = 181
#: How far from a body, in world units, a click still selects it.
PICK_REACH = 6.0
LEFT, RIGHT = -1.0, 1.0
RECEPTORS = 6
#: Blocks of the 42 inputs, in the order of §5.5: (label, first index, length).
INPUT_BLOCKS = (
    ("vision food", 0, RECEPTORS),
    ("vision water", 6, RECEPTORS),
    ("vision density", 12, RECEPTORS),
    ("vision affinity", 18, RECEPTORS),
    ("smell food", 24, 2),
    ("smell water", 26, 2),
    ("touch", 28, 3),
    ("reserves", 31, 2),
    ("speed", 33, 3),
    ("maturity, age", 36, 2),
    ("memory", 38, 1),
    ("climate", 39, 3),
)


@dataclass(frozen=True, slots=True)
class Inspection:
    """Everything the inspector knows of one creature.

    Attributes:
        uid, founder_id: Identity and lineage (§8.6).
        position, heading, body_radius: Its body.
        age, energy, e_max, hydration, h_max: Its state (§7.1, §7.2).
        mass, v_max, color: Its genes (§5.1).
        lobes: Outline of each receptor, in world units, shape ``(6, LOBE_SAMPLES, 2)``.
        nostrils: Left and right nostrils, in world units, shape ``(2, 2)``.
        inputs: Its 42 inputs of the next tick, when the caller has them (§5.5).
    """

    uid: int
    founder_id: int
    position: FloatArray
    heading: float
    body_radius: float
    age: int
    energy: float
    e_max: float
    hydration: float
    h_max: float
    mass: float
    v_max: float
    color: float
    lobes: FloatArray
    nostrils: FloatArray
    inputs: FloatArray | None


def pick_creature(frame: Frame, point: FloatArray) -> int | None:
    """The ``uid`` of the creature nearest to ``point``, measured on the torus (§3.1).

    Returns:
        The ``uid``, or ``None`` if no body is within ``PICK_REACH`` of the point.
    """
    if frame.uids.size == 0:
        return None
    gaps = distance(frame.positions, point, frame.size[0], frame.size[1]) - frame.body_radii
    nearest = int(np.argmin(gaps))
    return int(frame.uids[nearest]) if gaps[nearest] <= PICK_REACH else None


def retina_lobes(
    position: FloatArray, heading: float, light: float, sensors: SensorConfig
) -> FloatArray:
    """Outline of the sensitivity g_j(θ) of each receptor, scaled by r_vis(t) (§5.2).

    The point of receptor ``j`` at angle θ lies at ``r_vis(t) · g_j(θ)`` from the body, so
    a lobe reaches the vision radius at its centre μ_j and closes at the edge of the field.
    The values come from the kernel of the sensors, with the falloff f(d) at its maximum.

    Returns:
        Points in world units, not wrapped, shape ``(6, LOBE_SAMPLES, 2)``.
    """
    radius = vision_radius(light, sensors)
    retina = make_retina(radius, sensors)
    theta = np.linspace(-retina.fov, retina.fov, LOBE_SAMPLES)
    kept, signals = retina_signals(
        np.zeros(LOBE_SAMPLES, dtype=np.int64),
        np.cos(theta),
        np.sin(theta),
        np.zeros(LOBE_SAMPLES, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.array([radius, retina.fov, retina.edge], dtype=np.float64),
        retina.mu,
        retina.sigma,
    )
    # The kernel may reject the two ends by rounding; they are zero anyway, by the window w(θ).
    gain = np.zeros((LOBE_SAMPLES, retina.mu.size), dtype=np.float64)
    gain[kept] = signals
    directions = np.column_stack((np.cos(heading + theta), np.sin(heading + theta)))
    outlines: FloatArray = position + radius * gain[:, :, np.newaxis] * directions[:, np.newaxis]
    return np.swapaxes(outlines, 0, 1)


def inspect_creature(
    arrays: WorldArrays,
    frame: Frame,
    uid: int,
    config: GenomeConfig,
    observed: tuple[FloatArray, IntArray] | None = None,
) -> Inspection | None:
    """The inspector of the creature ``uid``, or ``None`` if it is no longer alive (§5).

    Args:
        arrays: The snapshot.
        frame: The frame read from it.
        uid: The creature.
        config: Run configuration.
        observed: ``Simulation.observe()``, to show the inputs next to the body.
    """
    rows = np.flatnonzero(frame.uids == uid)
    if rows.size == 0:
        return None
    row = int(rows[0])
    slot = int(live_slots(arrays)[row])
    position = frame.positions[row]
    heading = float(frame.headings[row])
    body_radius = float(frame.body_radii[row])
    return Inspection(
        uid=uid,
        founder_id=_integer(arrays, "creatures.founder_id", slot),
        position=position,
        heading=heading,
        body_radius=body_radius,
        age=_integer(arrays, "creatures.age", slot),
        energy=_number(arrays, "creatures.energy", slot),
        e_max=_number(arrays, "creatures.e_max", slot),
        hydration=_number(arrays, "creatures.hydration", slot),
        h_max=_number(arrays, "creatures.h_max", slot),
        mass=_number(arrays, "creatures.mass", slot),
        v_max=_number(arrays, "creatures.v_max", slot),
        color=_number(arrays, "creatures.color", slot),
        lobes=retina_lobes(position, heading, frame.light, config.sensors),
        nostrils=_nostrils(position, heading, body_radius, config.sensors),
        inputs=_inputs_of(slot, observed),
    )


def panel_lines(inspection: Inspection) -> list[str]:
    """The text of the inspector, one line per entry."""
    i = inspection
    lines = [
        f"uid {i.uid}   founder {i.founder_id}",
        f"age {i.age}",
        f"energy {i.energy:.1f} / {i.e_max:.1f}",
        f"hydration {i.hydration:.1f} / {i.h_max:.1f}",
        f"mass {i.mass:.2f}   v_max {i.v_max:.2f}",
        f"color {i.color:.3f}",
    ]
    if i.inputs is None:
        return lines
    lines.append("")
    for label, first, length in INPUT_BLOCKS:
        values = " ".join(f"{value:+.2f}" for value in i.inputs[first : first + length].tolist())
        lines.append(f"{label}: {values}")
    return lines


def _nostrils(
    position: FloatArray, heading: float, body_radius: float, sensors: SensorConfig
) -> FloatArray:
    headings = np.array([heading], dtype=np.float64)
    radii = np.array([body_radius], dtype=np.float64)
    # Angles grow to the right (§1.3), so the left nostril is at −alpha_olf.
    return np.vstack(
        (
            position + nostril_offsets(headings, radii, LEFT, sensors),
            position + nostril_offsets(headings, radii, RIGHT, sensors),
        )
    )


def _inputs_of(slot: int, observed: tuple[FloatArray, IntArray] | None) -> FloatArray | None:
    if observed is None:
        return None
    observations, slots = observed
    rows = np.flatnonzero(slots == slot)
    if rows.size == 0:
        return None
    inputs: FloatArray = observations[int(rows[0]), :NUM_INPUTS].copy()
    return inputs


def _number(arrays: WorldArrays, name: str, slot: int) -> float:
    return float(arrays[name][slot])


def _integer(arrays: WorldArrays, name: str, slot: int) -> int:
    return int(arrays[name][slot])
