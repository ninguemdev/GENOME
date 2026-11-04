"""Body genes and the quantities derived from them by allometry (ARQUITETURA §5.1, §6.3)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import CreatureConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Bodies:
    """Genes and derived quantities of a batch of creatures (§5.1).

    Attributes:
        mass, v_max, color: Body genes.
        body_radius: r_c = M^(1/2).
        touch_radius: r_t = r_c + touch_margin.
        e_max, h_max: Maximum reserves, proportional to mass.
        t_vida: Life expectancy, ``t_vida_ref · M^(1/4)``.
        t_mat: Maturity age, a fraction of ``t_vida``.
        inertia_decay: e^(−1/τ) with τ = ``k_inertia`` · M, fixed at birth (§6.3).
        rotation_decay: e^(−1/τ_rot) with τ_rot = ``k_inertia_rot`` · M (§6.3).
    """

    mass: FloatArray
    v_max: FloatArray
    color: FloatArray
    body_radius: FloatArray
    touch_radius: FloatArray
    e_max: FloatArray
    h_max: FloatArray
    t_vida: FloatArray
    t_mat: FloatArray
    inertia_decay: FloatArray
    rotation_decay: FloatArray


def build_bodies(
    mass: FloatArray, v_max: FloatArray, color: FloatArray, creatures: CreatureConfig
) -> Bodies:
    """Derive the body of each creature from its genes (§5.1, §6.3).

    Raises:
        ValueError: If a gene falls outside its range.
    """
    _require_genes(mass, v_max, color, creatures)
    body_radius = mass**creatures.body_radius_exponent
    t_vida = creatures.t_vida_ref * mass**creatures.t_vida_exponent
    return Bodies(
        mass=mass,
        v_max=v_max,
        color=color,
        body_radius=body_radius,
        touch_radius=body_radius + creatures.touch_margin,
        e_max=creatures.e_max_per_mass * mass,
        h_max=creatures.h_max_per_mass * mass,
        t_vida=t_vida,
        t_mat=creatures.t_mat_fraction * t_vida,
        inertia_decay=np.exp(-1.0 / (creatures.k_inertia * mass)),
        rotation_decay=np.exp(-1.0 / (creatures.k_inertia_rot * mass)),
    )


def _require_genes(
    mass: FloatArray, v_max: FloatArray, color: FloatArray, creatures: CreatureConfig
) -> None:
    """Check the ranges of §5.1; written so that NaN fails every test."""
    if not np.all((mass >= creatures.mass_min) & (mass <= creatures.mass_max)):
        raise ValueError(f"mass outside [{creatures.mass_min}, {creatures.mass_max}]")
    if not np.all((v_max >= creatures.v_max_min) & (v_max <= creatures.v_max_max)):
        raise ValueError(f"v_max outside [{creatures.v_max_min}, {creatures.v_max_max}]")
    if not np.all((color >= 0.0) & (color < 1.0)):
        raise ValueError("color outside [0, 1)")
