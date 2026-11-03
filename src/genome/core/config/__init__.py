"""Immutable run configuration: sections, YAML loading, validation and hash (ARQUITETURA §2.3)."""

from genome.core.config.errors import ConfigError
from genome.core.config.loader import (
    canonical_json,
    config_from_mapping,
    config_hash,
    config_to_dict,
    load_config,
)
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
from genome.core.config.validation import validate

__all__ = [
    "ClimateConfig",
    "ConfigError",
    "CreatureConfig",
    "CurriculumConfig",
    "GenomeConfig",
    "LabConfig",
    "MetabolismConfig",
    "NEATConfig",
    "PlantConfig",
    "RunConfig",
    "SensorConfig",
    "TrainingConfig",
    "WaterConfig",
    "WorldConfig",
    "canonical_json",
    "config_from_mapping",
    "config_hash",
    "config_to_dict",
    "load_config",
    "validate",
]
