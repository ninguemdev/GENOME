"""Semantic validation of a built configuration (ARQUITETURA §2.3).

All rule violations are collected and reported together in one ``ConfigError``.
"""

import math
from collections.abc import Callable

from genome.core.config.errors import ConfigError
from genome.core.config.sections import (
    ClimateConfig,
    CreatureConfig,
    CurriculumConfig,
    GenomeConfig,
    LabConfig,
    MetabolismConfig,
    NEATConfig,
    PlantConfig,
    RunConfig,
    SensorConfig,
    TrainingConfig,
    WaterConfig,
    WorldConfig,
)

# Weights must sum to one up to float rounding (§9.2).
WEIGHT_SUM_TOLERANCE = 1e-12
# The retina has six receptors (§5.2).
RETINA_RECEPTORS = 6

Errors = list[str]


def validate(config: GenomeConfig) -> None:
    """Validate every section and the rules that link sections.

    Raises:
        ConfigError: Listing every violated rule.
    """
    errors: Errors = []
    _check_run(config.run, errors)
    _check_world(config.world, errors)
    _check_climate(config.climate, errors)
    _check_water(config.water, errors)
    _check_plants(config.plants, errors)
    _check_creatures(config.creatures, errors)
    _check_sensors(config.sensors, errors)
    _check_metabolism(config.metabolism, errors)
    _check_neat(config.neat, errors)
    _check_training(config.training, errors)
    _check_curriculum(config.curriculum, errors)
    _check_lab(config.lab, errors)
    _check_cross_section(config, errors)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def _require(errors: Errors, condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _fields_where(
    section: object, prefix: str, names: tuple[str, ...], ok: Callable[[float], bool], rule: str
) -> Errors:
    found: Errors = []
    for name in names:
        value = getattr(section, name)
        if not ok(value):
            found.append(f"{prefix}.{name} must be {rule} (got {value!r})")
    return found


def _positive(section: object, prefix: str, *names: str) -> Errors:
    return _fields_where(section, prefix, names, lambda v: v > 0, "> 0")


def _non_negative(section: object, prefix: str, *names: str) -> Errors:
    return _fields_where(section, prefix, names, lambda v: v >= 0, ">= 0")


def _unit_interval(section: object, prefix: str, *names: str) -> Errors:
    return _fields_where(section, prefix, names, lambda v: 0 <= v <= 1, "in [0, 1]")


def _check_run(run: RunConfig, errors: Errors) -> None:
    errors += _positive(run, "run", "max_population")
    _require(
        errors,
        run.max_ticks is None or run.max_ticks > 0,
        "run.max_ticks must be > 0 or null",
    )
    _require(
        errors,
        run.mode != "training" or run.max_ticks is not None,
        "run.max_ticks is required in training mode",
    )


def _check_world(world: WorldConfig, errors: Errors) -> None:
    errors += _positive(world, "world", "width", "height", "initial_population")
    errors += _non_negative(world, "world", "n_rivers", "n_lakes", "n_puddle_sites")
    _require(errors, world.n_obstacles == 0, "world.n_obstacles must be 0 in v1 (§3.5)")


def _check_climate(climate: ClimateConfig, errors: Errors) -> None:
    errors += _positive(climate, "climate", "t_ciclo", "t_dia", "t_crep", "tau_u")
    errors += _positive(climate, "climate", "rain_block_ticks")
    errors += _unit_interval(climate, "climate", "p_sc", "p_cs", "u_chuva", "u_seco")
    _require(errors, climate.t_dia <= climate.t_ciclo, "climate.t_dia must be <= t_ciclo")
    _require(errors, 2 * climate.t_crep <= climate.t_dia, "climate.t_crep must be <= t_dia / 2")


def _check_water(water: WaterConfig, errors: Errors) -> None:
    errors += _positive(water, "water", "w_rio", "s_rio", "v_max_min", "k_r_max", "v_p")
    errors += _positive(water, "water", "r_p_max", "gamma_chuva")
    errors += _positive(water, "water", "river_harmonics", "river_vertex_step")
    errors += _non_negative(water, "water", "v_corr_max", "rho_l", "eps_p", "river_amplitude")
    _require(errors, water.v_max_min <= water.v_max_max, "water.v_max_min must be <= v_max_max")
    _require(errors, water.s_rio <= water.w_rio / 2, "water.s_rio must be <= w_rio / 2 (§3.2)")
    # ADR 0004: the polyline must be finer than the perception sampling.
    _require(
        errors,
        water.river_vertex_step <= water.s_rio,
        "water.river_vertex_step must be <= s_rio (ADR 0004)",
    )


def _check_plants(plants: PlantConfig, errors: Errors) -> None:
    errors += _positive(plants, "plants", "e_stack", "b_max", "d_p", "r_irr", "t_irr")
    errors += _positive(plants, "plants", "t_murcha", "irrigation_hop_factor", "max_plants")
    errors += _positive(plants, "plants", "min_spacing_factor")
    errors += _non_negative(plants, "plants", "k_max", "n_plantas_0")
    errors += _unit_interval(plants, "plants", "p_sol", "p_chuva")
    _require(
        errors, plants.n_plantas_0 <= plants.max_plants, "plants.n_plantas_0 must be <= max_plants"
    )


def _check_creatures(creatures: CreatureConfig, errors: Errors) -> None:
    errors += _positive(creatures, "creatures", "mass_min", "v_max_min", "e_max_per_mass")
    errors += _positive(creatures, "creatures", "h_max_per_mass", "t_vida_ref", "omega_max")
    errors += _positive(creatures, "creatures", "k_inertia", "k_inertia_rot", "t_mat_fraction")
    errors += _non_negative(creatures, "creatures", "touch_margin", "d_min_agua")
    errors += _unit_interval(creatures, "creatures", "k_back", "k_lat")
    errors += _unit_interval(
        creatures, "creatures", "founder_energy_fraction", "founder_hydration_fraction"
    )
    _require(
        errors, creatures.mass_min <= creatures.mass_max, "creatures.mass_min must be <= mass_max"
    )
    _require(
        errors,
        creatures.v_max_min <= creatures.v_max_max,
        "creatures.v_max_min must be <= v_max_max",
    )


def _check_sensors(sensors: SensorConfig, errors: Errors) -> None:
    errors += _positive(sensors, "sensors", "r_vis", "fov_half_deg", "edge_window_deg")
    errors += _positive(sensors, "sensors", "cell_olf", "sigma_olf", "t_olf", "eps_olf")
    errors += _positive(sensors, "sensors", "k_olf_food", "k_olf_water", "olf_water_subsamples")
    errors += _unit_interval(sensors, "sensors", "m_noite", "s_min")
    _require(
        errors,
        len(sensors.receptor_mu_deg) == RETINA_RECEPTORS
        and len(sensors.receptor_sigma_deg) == RETINA_RECEPTORS,
        f"sensors.receptor_mu_deg and receptor_sigma_deg need {RETINA_RECEPTORS} values",
    )
    _require(
        errors,
        all(s > 0 for s in sensors.receptor_sigma_deg),
        "sensors.receptor_sigma_deg must be > 0",
    )
    _require(
        errors,
        list(sensors.receptor_mu_deg) == sorted(sensors.receptor_mu_deg),
        "sensors.receptor_mu_deg must be sorted",
    )


def _check_metabolism(metabolism: MetabolismConfig, errors: Errors) -> None:
    prefix = "metabolism"
    errors += _positive(metabolism, prefix, "b_0", "q_eat", "q_drink", "t_rec", "eps_repr")
    errors += _non_negative(metabolism, prefix, "b_v", "k_sen", "c_m", "c_g", "c_a", "chi")
    errors += _non_negative(metabolism, prefix, "h_0", "kappa", "eta")
    errors += _unit_interval(metabolism, prefix, "eat_threshold", "drink_threshold")
    errors += _unit_interval(metabolism, prefix, "reproduce_threshold", "eps_repr")
    errors += _unit_interval(
        metabolism, prefix, "repr_min_energy_fraction", "repr_min_hydration_fraction"
    )
    errors += _unit_interval(
        metabolism, prefix, "child_energy_fraction", "child_hydration_fraction"
    )
    # §7.1: water alone must never sustain a creature, so R_H < C_basal.
    _require(errors, 0 <= metabolism.rho_h < 1, "metabolism.rho_h must be in [0, 1) (§7.1)")
    _require(errors, 0 <= metabolism.f_sen < 1, "metabolism.f_sen must be in [0, 1)")


def _check_neat(neat: NEATConfig, errors: Errors) -> None:
    prefix = "neat"
    errors += _unit_interval(neat, prefix, "p_mutate_weights", "p_mutate_bias", "p_weight_perturb")
    errors += _unit_interval(neat, prefix, "p_add_connection", "p_add_node", "p_toggle_connection")
    errors += _unit_interval(neat, prefix, "crossover_disabled_prob", "survival_threshold")
    errors += _unit_interval(neat, prefix, "p_crossover", "p_interspecies")
    errors += _unit_interval(neat, prefix, "in_world_mutation_rate", "import_mutation_rate")
    errors += _positive(neat, prefix, "init_weight_sigma", "weight_perturb_sigma", "weight_limit")
    errors += _positive(neat, prefix, "weight_reset_sigma", "add_connection_attempts", "delta_0")
    errors += _positive(neat, prefix, "founder_mass", "founder_v_max", "s_alvo", "t_tax")
    errors += _positive(neat, prefix, "delta_tax", "elitism_min_size", "g_estag")
    errors += _non_negative(neat, prefix, "mass_mutation_sigma", "v_max_mutation_sigma")
    errors += _non_negative(neat, prefix, "color_mutation_sigma", "c_1", "c_3", "c_b")
    errors += _non_negative(neat, prefix, "m_min", "a_prot", "protected_top_species")
    _require(
        errors,
        0 < neat.sparse_min_connections <= neat.sparse_max_connections,
        "neat.sparse_min_connections must be in (0, sparse_max_connections]",
    )
    _require(errors, neat.delta_increase > 1, "neat.delta_increase must be > 1 (§8.5)")
    _require(errors, 0 < neat.delta_decrease < 1, "neat.delta_decrease must be in (0, 1) (§8.5)")


def _check_training(training: TrainingConfig, errors: Errors) -> None:
    errors += _positive(training, "training", "n", "worlds_per_generation", "t_max", "d_sat")
    errors += _unit_interval(training, "training", "w_s", "w_b")
    errors += _non_negative(training, "training", "k_repr", "lambda_grandchildren")
    _require(
        errors,
        math.isclose(training.w_s + training.w_b, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE),
        "training.w_s + training.w_b must equal 1 (§9.2)",
    )


def _check_curriculum(curriculum: CurriculumConfig, errors: Errors) -> None:
    prefix = "curriculum"
    errors += _unit_interval(curriculum, prefix, "f_avanco")
    errors += _positive(curriculum, prefix, "g_janela", "g_max_fase", "g_plateau")
    errors += _positive(curriculum, prefix, "phase1_plant_multiplier", "phase1_p_sol_multiplier")
    errors += _positive(curriculum, prefix, "phase1_cost_multiplier")
    errors += _positive(curriculum, prefix, "phase1_patch_count", "phase1_patch_radius")
    errors += _non_negative(curriculum, prefix, "g_trans", "phase1_patch_gap")


def _check_lab(lab: LabConfig, errors: Errors) -> None:
    errors += _non_negative(lab, "lab", "lab_seed")
    errors += _positive(lab, "lab", "n_seeds", "n_seeds_min", "n_val", "n_test", "g_val")
    errors += _positive(lab, "lab", "n_bootstrap", "ledger_tolerance", "t_log", "t_ckpt")
    errors += _positive(lab, "lab", "g_ckpt", "flush_rows")
    errors += _unit_interval(lab, "lab", "alpha", "immobile_death_fraction")
    _require(errors, lab.n_seeds >= lab.n_seeds_min, "lab.n_seeds must be >= n_seeds_min (§10.3)")


def _check_cross_section(config: GenomeConfig, errors: Errors) -> None:
    world, sensors, training = config.world, config.sensors, config.training
    half_side = min(world.width, world.height) / 2
    # §3.1: no perception radius may reach half the torus.
    _require(errors, sensors.r_vis < half_side, "sensors.r_vis must be < min(width, height) / 2")
    _require(
        errors,
        _divides(sensors.cell_olf, world.width) and _divides(sensors.cell_olf, world.height),
        "world.width and world.height must be multiples of sensors.cell_olf (§5.3)",
    )
    curriculum = config.curriculum
    # ADR 0003: a patch must fit inside the torus without seeing itself around it.
    _require(
        errors,
        curriculum.phase1_patch_radius < half_side,
        "curriculum.phase1_patch_radius must be < min(width, height) / 2",
    )
    _require(
        errors,
        world.initial_population <= config.run.max_population,
        "world.initial_population must be <= run.max_population",
    )
    if config.run.mode == "training":
        _require(
            errors,
            config.run.max_ticks == training.t_max,
            "run.max_ticks must equal training.t_max in training mode (§12.1)",
        )
        _require(
            errors,
            world.initial_population == training.n,
            "world.initial_population must equal training.n in training mode (§12.1)",
        )


def _divides(step: float, length: float) -> bool:
    ratio = length / step
    return math.isclose(ratio, round(ratio))
