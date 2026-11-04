"""Immutable configuration sections (ARQUITETURA §2.3).

Field names follow the parameter tables of ARQUITETURA in lower snake case
(``rho_L`` becomes ``rho_l``, ``T_ciclo`` becomes ``t_ciclo``). Values that the
specification states in running text, rather than in a table, are also fields,
so that no number is hard-coded in the simulation (ADR 0002).
"""

import math
from dataclasses import dataclass
from typing import Literal

RunMode = Literal["training", "continuous"]
ReproductionMode = Literal["mixed", "asexual", "sexual"]
TopologyGenesis = Literal["dense_starter", "sparse_starter"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Execution mode and limits (ARQUITETURA §2.3, §12.1)."""

    mode: RunMode
    max_ticks: int | None
    terminate_on_extinction: bool
    max_population: int


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """World size and element counts of the preset (ARQUITETURA §3, §12.1)."""

    width: float
    height: float
    initial_population: int
    n_rivers: int
    n_lakes: int
    n_puddle_sites: int
    n_obstacles: int


@dataclass(frozen=True, slots=True)
class ClimateConfig:
    """Day/night cycle, Markov rain and air humidity (ARQUITETURA §3.6, §3.7, §2.5)."""

    t_ciclo: int
    t_dia: int
    t_crep: int
    p_sc: float
    p_cs: float
    tau_u: float
    u_chuva: float
    u_seco: float
    rain_block_ticks: int


@dataclass(frozen=True, slots=True)
class WaterConfig:
    """Rivers, lakes and puddles (ARQUITETURA §3.2 to §3.4, §3.7)."""

    w_rio: float
    v_corr_max: float
    s_rio: float
    river_harmonics: int
    river_amplitude: float
    river_vertex_step: float
    v_max_min: float
    v_max_max: float
    k_r_max: float
    rho_l: float
    gamma_chuva: float
    v_p: float
    r_p_max: float
    eps_p: float


@dataclass(frozen=True, slots=True)
class PlantConfig:
    """Plant biomass, irrigation, growth and propagation (ARQUITETURA §4, §12.1)."""

    e_stack: float
    b_max: int
    d_p: float
    r_irr: float
    k_max: int
    t_irr: int
    p_sol: float
    p_chuva: float
    t_murcha: float
    irrigation_hop_factor: float
    min_spacing_factor: float
    n_plantas_0: int
    max_plants: int


@dataclass(frozen=True, slots=True)
class CreatureConfig:
    """Body genes, allometry, locomotion and initial state (ARQUITETURA §5.1, §5.6, §6.3, §6.4)."""

    mass_min: float
    mass_max: float
    v_max_min: float
    v_max_max: float
    body_radius_exponent: float
    e_max_per_mass: float
    h_max_per_mass: float
    t_vida_ref: float
    t_vida_exponent: float
    t_mat_fraction: float
    touch_margin: float
    omega_max: float
    k_back: float
    k_lat: float
    k_inertia: float
    k_inertia_rot: float
    founder_energy_fraction: float
    founder_hydration_fraction: float
    d_min_agua: float


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Vision retina, olfaction and touch (ARQUITETURA §5.2 to §5.4, §5.7).

    Angles are stored in degrees, as written in the specification; the
    ``*_rad`` properties give the radian values used by the simulation.
    """

    r_vis: float
    m_noite: float
    fov_half_deg: float
    edge_window_deg: float
    receptor_mu_deg: tuple[float, ...]
    receptor_sigma_deg: tuple[float, ...]
    cell_olf: float
    sigma_olf: float
    t_olf: int
    olf_water_subsamples: int
    alpha_olf_deg: float
    nostril_body_factor: float
    nostril_offset: float
    k_olf_food: float
    k_olf_water: float
    eps_olf: float
    s_min: float

    @property
    def fov_half_rad(self) -> float:
        """Half field of view in radians (§5.2)."""
        return math.radians(self.fov_half_deg)

    @property
    def edge_window_rad(self) -> float:
        """Width of the edge window w(θ) in radians (§5.2)."""
        return math.radians(self.edge_window_deg)

    @property
    def receptor_mu_rad(self) -> tuple[float, ...]:
        """Receptor centres μ_j in radians (§5.2)."""
        return tuple(math.radians(v) for v in self.receptor_mu_deg)

    @property
    def receptor_sigma_rad(self) -> tuple[float, ...]:
        """Receptor widths σ_j in radians (§5.2)."""
        return tuple(math.radians(v) for v in self.receptor_sigma_deg)

    @property
    def alpha_olf_rad(self) -> float:
        """Nostril angle in radians (§5.3)."""
        return math.radians(self.alpha_olf_deg)


@dataclass(frozen=True, slots=True)
class MetabolismConfig:
    """Energy, water, actions and reproduction (ARQUITETURA §6.2, §7.1 to §7.4, §7.7)."""

    b_0: float
    b_v: float
    kleiber_exponent: float
    surface_exponent: float
    k_sen: float
    f_sen: float
    c_m: float
    c_g: float
    c_a: float
    rho_h: float
    chi: float
    h_0: float
    kappa: float
    eta: float
    q_eat: float
    q_drink: float
    eat_threshold: float
    drink_threshold: float
    reproduce_threshold: float
    repr_min_energy_fraction: float
    repr_min_hydration_fraction: float
    child_energy_fraction: float
    child_hydration_fraction: float
    t_rec: int
    eps_repr: float
    reproduction_mode: ReproductionMode


@dataclass(frozen=True, slots=True)
class NEATConfig:
    """Genome genesis, mutation, crossover, distance and speciation (ARQUITETURA §6.1, §8)."""

    topology_genesis: TopologyGenesis
    sparse_min_connections: int
    sparse_max_connections: int
    init_weight_sigma: float
    reproduce_bias_init: float
    founder_mass: float
    founder_v_max: float
    p_mutate_weights: float
    p_mutate_bias: float
    p_weight_perturb: float
    weight_perturb_sigma: float
    weight_reset_sigma: float
    weight_limit: float
    p_add_connection: float
    add_connection_attempts: int
    p_add_node: float
    p_toggle_connection: float
    mass_mutation_sigma: float
    v_max_mutation_sigma: float
    color_mutation_sigma: float
    crossover_disabled_prob: float
    c_1: float
    c_3: float
    c_b: float
    delta_0: float
    delta_increase: float
    delta_decrease: float
    s_alvo: int
    m_min: int
    a_prot: int
    g_estag: int
    protected_top_species: int
    survival_threshold: float
    elitism_min_size: int
    p_crossover: float
    p_interspecies: float
    in_world_mutation_rate: float
    import_mutation_rate: float
    t_tax: int
    delta_tax: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Generational training and fitness (ARQUITETURA §8.6, §8.8, §9.2, §9.6)."""

    n: int
    worlds_per_generation: int
    t_max: int
    w_s: float
    w_b: float
    k_repr: float
    d_sat: float
    lambda_grandchildren: float


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Curriculum phases, advance criterion and transition (ARQUITETURA §9.3 to §9.6)."""

    f_avanco: float
    g_janela: int
    g_max_fase: int
    g_trans: int
    g_plateau: int
    phase1_plant_multiplier: float
    phase1_p_sol_multiplier: float
    phase1_cost_multiplier: float
    phase1_patch_count: int
    phase1_patch_radius: float
    phase1_patch_gap: float


@dataclass(frozen=True, slots=True)
class LabConfig:
    """Experimental protocol, telemetry and checkpoints (ARQUITETURA §7.6, §10, §11.8)."""

    lab_seed: int
    n_seeds: int
    n_seeds_min: int
    n_val: int
    n_test: int
    g_val: int
    alpha: float
    n_bootstrap: int
    immobile_death_fraction: float
    ledger_tolerance: float
    ledger_debug: bool
    t_log: int
    t_ckpt: int
    g_ckpt: int
    flush_rows: int
    max_cap_fraction: float
    heuristic_thirst_threshold: float
    heuristic_hunger_threshold: float
    heuristic_drink_until: float
    heuristic_reproduce_energy: float
    heuristic_reproduce_hydration: float
    heuristic_crowding_limit: float
    heuristic_forward: float
    heuristic_turn_gain: float
    heuristic_explore_turn: float
    heuristic_memory_step: float


@dataclass(frozen=True, slots=True)
class GenomeConfig:
    """Complete configuration of a run: one field per section (ARQUITETURA §2.3)."""

    run: RunConfig
    world: WorldConfig
    climate: ClimateConfig
    water: WaterConfig
    plants: PlantConfig
    creatures: CreatureConfig
    sensors: SensorConfig
    metabolism: MetabolismConfig
    neat: NEATConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    lab: LabConfig
