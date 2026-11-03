import importlib

import pytest

PACKAGES = (
    "genome",
    "genome.core",
    "genome.neat",
    "genome.envs",
    "genome.lab",
    "genome.io",
    "genome.viz",
)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_is_importable_and_documented(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__
    assert module.__doc__.strip()
