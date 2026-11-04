"""The environment wired with what ``genome.envs`` cannot import (ARQUITETURA §2.1, ADR 0008)."""

from functools import partial

from genome.core.config import GenomeConfig
from genome.envs.parallel import ControllerFactory, GenomeParallelEnv
from genome.lab.clonal import ClonalOperators
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import ControllerName, fixed_genomes, make_controller


def make_env(config: GenomeConfig) -> GenomeParallelEnv[FixedGenome]:
    """A ``GenomeParallelEnv`` of clonal founders with empty genomes, as F0 to F2 use."""
    return GenomeParallelEnv(config, ClonalOperators(), fixed_genomes(config))


def fixed_controller(name: ControllerName) -> ControllerFactory:
    """The value of ``options["controller"]`` that puts a fixed controller in charge (§2.2)."""
    return partial(make_controller, name)
