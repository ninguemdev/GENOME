"""Shared test configuration.

Hypothesis is derandomized so that no test depends on the clock or on the
order of execution (PADROES §7).
"""

from hypothesis import settings

settings.register_profile("genome", derandomize=True, deadline=None)
settings.load_profile("genome")
