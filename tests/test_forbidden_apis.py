"""Static checks for APIs that ruff cannot ban (AGENTS.md, ARQUITETURA §2.5 and §8.2)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "genome"
SOURCES = sorted(SRC.rglob("*.py"))


def _builtin_hash_calls(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash"
    ]


def test_sources_found() -> None:
    assert SOURCES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_builtin_hash(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _builtin_hash_calls(tree) == [], f"built-in hash() is forbidden in {path}"


def test_detector_finds_builtin_hash() -> None:
    assert _builtin_hash_calls(ast.parse("x = hash('a')\n")) == [1]
    assert _builtin_hash_calls(ast.parse("x = hashlib.sha256(b'a')\n")) == []
