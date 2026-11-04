"""PettingZoo adapter: ``GenomeParallelEnv`` over the simulation core.

Thin wrapper over the core's public interface (ARQUITETURA §2.2). Depends only
on ``genome.core``; genome operators, founders, controllers and the renderer are
injected (ADR 0008).
"""
