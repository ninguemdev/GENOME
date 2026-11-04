"""Shared test configuration.

Hypothesis is derandomized so that no test depends on the clock or on the
order of execution (PADROES §7).
"""

from pathlib import Path

import pytest
from hypothesis import settings

from genome.core.config import GenomeConfig, load_config

settings.register_profile("genome", derandomize=True, deadline=None)
settings.load_profile("genome")

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def config() -> GenomeConfig:
    """The training preset, the reference configuration of the unit tests."""
    return load_config(CONFIGS / "training.yaml")
