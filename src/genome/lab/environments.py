"""The environment wired with what ``genome.envs`` cannot import (ARQUITETURA §2.1, ADR 0008)."""

from functools import partial
from typing import Literal

from genome.core.config import GenomeConfig
from genome.envs.parallel import ControllerFactory, GenomeParallelEnv, Renderer
from genome.lab.clonal import ClonalOperators
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import ControllerName, fixed_genomes, make_controller

RenderMode = Literal["human", "rgb_array"]


def make_env(
    config: GenomeConfig, render_mode: RenderMode | None = None
) -> GenomeParallelEnv[FixedGenome]:
    """A ``GenomeParallelEnv`` of clonal founders with empty genomes, as F0 to F2 use.

    Args:
        config: Run configuration.
        render_mode: With a mode, ``render`` draws through ``genome.viz``, which needs the
            extra ``viz`` (§2.2, ADR 0010).
    """
    renderer: Renderer | None = None
    if render_mode is not None:
        # Imported here so that an environment without rendering does not need pygame.
        from genome.viz.renderer import WorldRenderer

        renderer = WorldRenderer(config, render_mode)
    return GenomeParallelEnv(
        config, ClonalOperators(), fixed_genomes(config), render_mode, renderer
    )


def fixed_controller(name: ControllerName) -> ControllerFactory:
    """The value of ``options["controller"]`` that puts a fixed controller in charge (§2.2)."""
    return partial(make_controller, name)
