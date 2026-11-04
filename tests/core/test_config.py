import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from genome.core.config import (
    ConfigError,
    GenomeConfig,
    canonical_json,
    config_from_mapping,
    config_hash,
    load_config,
)
from genome.core.config.loader import build_dataclass

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TRAINING = CONFIGS / "training.yaml"
CONTINUOUS = CONFIGS / "continuous.yaml"
F1_ECOLOGY = CONFIGS / "f1_ecology.yaml"


def raw(path: Path = TRAINING) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def with_value(section: str, key: str, value: object, base: Path = TRAINING) -> dict[str, Any]:
    data = copy.deepcopy(raw(base))
    data[section][key] = value
    return data


@pytest.fixture(scope="module")
def training() -> GenomeConfig:
    return load_config(TRAINING)


@pytest.fixture(scope="module")
def continuous() -> GenomeConfig:
    return load_config(CONTINUOUS)


# ---------------------------------------------------------------- presets


def test_presets_load(training: GenomeConfig, continuous: GenomeConfig) -> None:
    assert training.run.mode == "training"
    assert continuous.run.mode == "continuous"


def test_training_preset_matches_section_12_1(training: GenomeConfig) -> None:
    assert (training.world.width, training.world.height) == (256.0, 256.0)
    assert training.world.initial_population == 200
    assert training.run.max_population == 600
    assert (training.world.n_rivers, training.world.n_lakes) == (2, 3)
    assert training.world.n_puddle_sites == 20
    assert (training.plants.n_plantas_0, training.plants.max_plants) == (600, 10000)
    assert training.run.max_ticks == 2000
    assert training.run.terminate_on_extinction is True
    # Olfaction grid 64 × 64 (§12.1) follows from W / cell_olf.
    assert training.world.width / training.sensors.cell_olf == 64


def test_continuous_preset_matches_section_12_1(continuous: GenomeConfig) -> None:
    assert (continuous.world.width, continuous.world.height) == (512.0, 512.0)
    assert continuous.world.initial_population == 1000
    assert continuous.run.max_population == 1200
    assert (continuous.world.n_rivers, continuous.world.n_lakes) == (4, 10)
    assert continuous.world.n_puddle_sites == 80
    assert (continuous.plants.n_plantas_0, continuous.plants.max_plants) == (2500, 40000)
    assert continuous.run.max_ticks is None
    assert continuous.world.width / continuous.sensors.cell_olf == 128


def test_f1_preset_is_training_world_for_ten_thousand_ticks(training: GenomeConfig) -> None:
    f1 = load_config(F1_ECOLOGY)
    assert f1.run.mode == "training"
    assert f1.run.max_ticks == f1.training.t_max == 10_000
    assert f1.neat.in_world_mutation_rate == 0.0
    # ADR 0004: everything else is the training preset.
    assert dataclasses.replace(f1.run, max_ticks=2000) == training.run
    assert dataclasses.replace(f1.training, t_max=2000) == training.training
    for field in dataclasses.fields(GenomeConfig):
        if field.name not in ("run", "training"):
            assert getattr(f1, field.name) == getattr(training, field.name), field.name


def test_presets_differ_only_in_section_12_1(
    training: GenomeConfig, continuous: GenomeConfig
) -> None:
    for name in ("climate", "water", "creatures", "sensors", "metabolism", "neat"):
        assert getattr(training, name) == getattr(continuous, name), name
    for name in ("training", "curriculum", "lab"):
        assert getattr(training, name) == getattr(continuous, name), name


def test_table_values_sample(training: GenomeConfig) -> None:
    c = training
    assert math.isclose(c.climate.p_sc, 1 / 600)
    assert math.isclose(c.climate.p_cs, 1 / 200)
    assert (c.climate.t_ciclo, c.climate.t_dia, c.climate.t_crep) == (1000, 600, 50)
    assert (c.water.w_rio, c.water.rho_l, c.water.gamma_chuva) == (8.0, 0.0005, 3.0)
    assert (c.plants.e_stack, c.plants.p_sol, c.plants.p_chuva) == (25.0, 0.004, 0.0015)
    assert (c.creatures.omega_max, c.creatures.k_inertia) == (0.2, 3.0)
    assert (c.sensors.r_vis, c.sensors.k_olf_water, c.sensors.s_min) == (40.0, 2.0, 0.8)
    assert (c.metabolism.b_0, c.metabolism.rho_h, c.metabolism.q_drink) == (0.06, 0.3, 4.0)
    assert c.metabolism.reproduction_mode == "mixed"
    assert (c.neat.c_1, c.neat.c_3, c.neat.c_b, c.neat.delta_0) == (1.0, 0.4, 0.5, 3.0)
    assert (c.neat.t_tax, c.neat.delta_tax) == (1000, 3.0)
    assert (c.training.n, c.training.t_max, c.training.d_sat) == (200, 2000, 5.0)
    assert (c.curriculum.f_avanco, c.curriculum.g_max_fase) == (0.6, 200)
    assert c.curriculum.phase1_patch_count == 16
    assert (c.curriculum.phase1_patch_radius, c.curriculum.phase1_patch_gap) == (22.0, 24.0)
    assert (c.lab.n_seeds, c.lab.n_val, c.lab.n_test, c.lab.n_bootstrap) == (20, 3, 10, 10000)
    assert (c.lab.t_log, c.lab.t_ckpt, c.lab.flush_rows) == (10, 50000, 10000)


def test_reference_creature_calibration(training: GenomeConfig) -> None:
    """§12.2: idle cost of the reference creature (M = 1, v_max = 1, full water)."""
    m = training.metabolism
    c_basal = m.b_0 + m.b_v
    r_h = m.rho_h * m.b_0
    assert math.isclose(c_basal, 0.08)
    assert math.isclose(r_h, 0.018)
    assert math.isclose(c_basal - r_h, 0.062)


def test_angles_converted_to_radians(training: GenomeConfig) -> None:
    s = training.sensors
    assert math.isclose(s.alpha_olf_rad, math.pi / 4)
    assert math.isclose(s.fov_half_rad, math.pi / 2)
    assert math.isclose(s.edge_window_rad, math.radians(15))
    assert s.receptor_mu_rad[0] == pytest.approx(math.radians(-70))
    assert s.receptor_sigma_rad[2] == pytest.approx(math.radians(8))
    assert isinstance(s.receptor_mu_deg, tuple)


def test_config_is_immutable(training: GenomeConfig) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        training.plants.p_sol = 0.5  # type: ignore[misc]


def test_ints_in_float_fields_are_normalized(training: GenomeConfig) -> None:
    assert isinstance(training.world.width, float)
    assert isinstance(training.water.w_rio, float)


# ---------------------------------------------------------------- hash


def test_hash_is_stable(training: GenomeConfig) -> None:
    again = load_config(TRAINING)
    assert config_hash(training) == config_hash(again)
    assert len(config_hash(training)) == 64


def test_hash_changes_with_any_field(training: GenomeConfig) -> None:
    changed = config_from_mapping(with_value("plants", "p_sol", 0.005))
    assert config_hash(changed) != config_hash(training)


def test_hash_differs_between_presets(training: GenomeConfig, continuous: GenomeConfig) -> None:
    assert config_hash(training) != config_hash(continuous)


def test_hash_independent_of_yaml_key_order(training: GenomeConfig) -> None:
    data = raw()
    reordered = {k: dict(reversed(list(data[k].items()))) for k in reversed(list(data))}
    assert config_hash(config_from_mapping(reordered)) == config_hash(training)


def test_canonical_json_is_sorted_and_compact(training: GenomeConfig) -> None:
    text = canonical_json(training)
    assert " " not in text.split('"mode"')[0]
    assert text.index('"climate"') < text.index('"creatures"') < text.index('"world"')


# ---------------------------------------------------------------- structural errors


def test_missing_key_is_reported_with_path() -> None:
    data = raw()
    del data["plants"]["p_sol"]
    with pytest.raises(ConfigError, match=r"plants\.p_sol: missing key"):
        config_from_mapping(data)


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"plants: unknown keys \['p_lua'\]"):
        config_from_mapping(with_value("plants", "p_lua", 0.1))


def test_unknown_section_is_rejected() -> None:
    data = raw()
    data["predation"] = {}
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_mapping(data)


def test_missing_section_is_rejected() -> None:
    data = raw()
    del data["lab"]
    with pytest.raises(ConfigError, match="lab: missing key"):
        config_from_mapping(data)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("plants", "p_sol", "alto", "expected float"),
        ("plants", "p_sol", True, "expected float"),
        ("plants", "p_sol", float("nan"), "finite"),
        ("plants", "k_max", 3.5, "expected int"),
        ("plants", "k_max", True, "expected int"),
        ("run", "terminate_on_extinction", 1, "expected bool"),
        ("run", "mode", "evolve", "expected one of"),
        ("run", "max_ticks", "x", "expected int"),
        ("sensors", "receptor_mu_deg", 3, "expected a list"),
        ("sensors", "receptor_mu_deg", [1, "a", 3, 4, 5, 6], r"receptor_mu_deg\[1\]"),
        ("metabolism", "reproduction_mode", "clonal", "expected one of"),
    ],
)
def test_wrong_types_are_rejected(section: str, key: str, value: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(with_value(section, key, value))


def test_root_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="<root>: expected a mapping"):
        config_from_mapping([1, 2])


def test_section_must_be_a_mapping() -> None:
    data = raw()
    data["world"] = 5
    with pytest.raises(ConfigError, match="world: expected a mapping"):
        config_from_mapping(data)


@dataclasses.dataclass(frozen=True)
class _Unsupported:
    pair: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class _WideUnion:
    value: int | str | None


@dataclasses.dataclass(frozen=True)
class _Bytes:
    value: bytes


@dataclasses.dataclass(frozen=True)
class _Text:
    name: str


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        (_Unsupported, r"tuple\[X, \.\.\.\]"),
        (_WideUnion, r"X \| None"),
        (_Bytes, "unsupported field type"),
    ],
)
def test_unsupported_field_types_fail_loudly(cls: type, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_dataclass(cls, {dataclasses.fields(cls)[0].name: 1}, "x")


def test_string_fields() -> None:
    assert build_dataclass(_Text, {"name": "a"}, "x") == _Text("a")
    with pytest.raises(ConfigError, match="expected str"):
        build_dataclass(_Text, {"name": 1}, "x")


def test_invalid_yaml_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


# ---------------------------------------------------------------- semantic rules


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("run", "max_population", 0, r"run\.max_population must be > 0"),
        ("run", "max_ticks", 0, r"run\.max_ticks must be > 0"),
        ("run", "max_ticks", None, "required in training mode"),
        ("world", "n_obstacles", 2, r"n_obstacles must be 0"),
        ("world", "n_lakes", -1, r"world\.n_lakes must be >= 0"),
        ("climate", "t_dia", 1200, r"t_dia must be <= t_ciclo"),
        ("climate", "t_crep", 400, r"t_crep must be <= t_dia / 2"),
        ("climate", "p_sc", 1.5, r"climate\.p_sc must be in \[0, 1\]"),
        ("water", "v_max_min", 6000, r"v_max_min must be <= v_max_max"),
        ("water", "s_rio", 5, r"s_rio must be <= w_rio / 2"),
        ("water", "river_vertex_step", 5, r"river_vertex_step must be <= s_rio"),
        ("water", "river_harmonics", 0, r"water.river_harmonics must be > 0"),
        ("plants", "n_plantas_0", 20000, r"n_plantas_0 must be <= max_plants"),
        ("creatures", "mass_min", 3.0, r"mass_min must be <= mass_max"),
        ("creatures", "v_max_min", 2.0, r"v_max_min must be <= v_max_max"),
        ("sensors", "receptor_mu_deg", [-70, -35, 10], "need 6 values"),
        ("sensors", "receptor_mu_deg", [-70, -10, -35, 10, 35, 70], "must be sorted"),
        ("sensors", "receptor_sigma_deg", [20, 14, 0, 8, 14, 20], "sigma_deg must be > 0"),
        ("sensors", "r_vis", 128, r"r_vis must be < min\(width, height\) / 2"),
        ("sensors", "cell_olf", 5, "multiples of sensors.cell_olf"),
        ("sensors", "olf_water_subsamples", 0, r"olf_water_subsamples must be > 0"),
        ("metabolism", "rho_h", 1.0, r"rho_h must be in \[0, 1\)"),
        ("metabolism", "f_sen", 1.0, r"f_sen must be in \[0, 1\)"),
        ("neat", "sparse_min_connections", 25, "sparse_min_connections must be in"),
        ("neat", "delta_increase", 1.0, "delta_increase must be > 1"),
        ("neat", "delta_decrease", 1.0, r"delta_decrease must be in \(0, 1\)"),
        ("training", "w_s", 0.7, "must equal 1"),
        ("training", "t_max", 1500, "must equal training.t_max"),
        ("training", "n", 150, "must equal training.n"),
        ("curriculum", "g_trans", -1, r"g_trans must be >= 0"),
        ("curriculum", "phase1_patch_count", 0, r"phase1_patch_count must be > 0"),
        ("curriculum", "phase1_patch_radius", 0, r"phase1_patch_radius must be > 0"),
        ("curriculum", "phase1_patch_radius", 128, r"phase1_patch_radius must be < min"),
        ("curriculum", "phase1_patch_gap", -1, r"phase1_patch_gap must be >= 0"),
        ("lab", "n_seeds", 5, "n_seeds must be >= n_seeds_min"),
        ("lab", "alpha", 2.0, r"lab\.alpha must be in \[0, 1\]"),
    ],
)
def test_validation_rules(section: str, key: str, value: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(with_value(section, key, value))


def test_initial_population_bounded_by_max_population() -> None:
    data = with_value("world", "initial_population", 2000, base=CONTINUOUS)
    with pytest.raises(ConfigError, match=r"initial_population must be <= run\.max_population"):
        config_from_mapping(data)


def test_continuous_mode_accepts_unbounded_ticks_and_any_population() -> None:
    data = with_value("world", "initial_population", 700, base=CONTINUOUS)
    assert config_from_mapping(data).world.initial_population == 700


def test_all_violations_are_reported_together() -> None:
    data = with_value("plants", "p_sol", 2.0)
    data["climate"]["p_cs"] = -1.0
    with pytest.raises(ConfigError) as info:
        config_from_mapping(data)
    assert "plants.p_sol" in str(info.value)
    assert "climate.p_cs" in str(info.value)
