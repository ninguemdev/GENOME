"""Strict construction of the configuration from YAML, and its hash (ARQUITETURA §2.3).

Every field must be present and of the declared type; unknown keys are
rejected. Errors name the full key path (for example ``plants.p_sol``).
"""

import dataclasses
import hashlib
import json
import math
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

from genome.core.config.errors import ConfigError
from genome.core.config.sections import GenomeConfig
from genome.core.config.validation import validate


def load_config(path: str | Path) -> GenomeConfig:
    """Load, build and validate a configuration file.

    Args:
        path: YAML file with one mapping per section of ``GenomeConfig``.

    Returns:
        The validated, immutable configuration.

    Raises:
        ConfigError: If the file is not a mapping, a key is missing, unknown
            or of the wrong type, or a validation rule fails.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return config_from_mapping(data)


def config_from_mapping(data: object) -> GenomeConfig:
    """Build and validate a configuration from plain data (already parsed YAML).

    Raises:
        ConfigError: If the data does not match ``GenomeConfig`` or fails validation.
    """
    config = build_dataclass(GenomeConfig, data, path="")
    validate(config)
    return config


def config_to_dict(config: GenomeConfig) -> dict[str, Any]:
    """Return the configuration as plain data (tuples become lists).

    ``Any`` is the honest type here: the values are heterogeneous JSON scalars.
    """
    return dataclasses.asdict(config)


def canonical_json(config: GenomeConfig) -> str:
    """Return the canonical JSON of the configuration (sorted keys, no spaces)."""
    return json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(config: GenomeConfig) -> str:
    """Return the SHA-256 hex digest of the canonical JSON (ARQUITETURA §2.3)."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def build_dataclass[T](cls: type[T], data: object, path: str) -> T:
    """Build dataclass ``cls`` from a mapping, recursively and strictly.

    Raises:
        ConfigError: On a missing, unknown or mistyped key.
    """
    where = path or "<root>"
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = dataclasses.fields(cls)  # type: ignore[arg-type]
    names = tuple(f.name for f in fields)
    unknown = sorted(str(k) for k in data if k not in names)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    hints = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for name in names:
        key_path = f"{path}.{name}" if path else name
        if name not in data:
            raise ConfigError(f"{key_path}: missing key")
        kwargs[name] = _coerce(data[name], hints[name], key_path)
    return cls(**kwargs)


def _coerce(value: object, hint: object, path: str) -> object:
    """Check ``value`` against the type ``hint`` and return it in canonical form."""
    origin = get_origin(hint)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return build_dataclass(hint, value, path)
    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{path}: expected one of {list(allowed)}, got {value!r}")
        return value
    if origin in (types.UnionType, typing.Union):
        return _coerce_optional(value, get_args(hint), path)
    if origin is tuple:
        return _coerce_tuple(value, get_args(hint), path)
    if hint is bool:
        return _expect(value, bool, path)
    if hint is int:
        return _coerce_int(value, path)
    if hint is float:
        return _coerce_float(value, path)
    if hint is str:
        return _expect(value, str, path)
    raise ConfigError(f"{path}: unsupported field type {hint!r}")


def _coerce_optional(value: object, args: tuple[object, ...], path: str) -> object:
    non_none = tuple(a for a in args if a is not type(None))
    if len(non_none) != 1 or len(args) != len(non_none) + 1:
        raise ConfigError(f"{path}: only 'X | None' unions are supported")
    if value is None:
        return None
    return _coerce(value, non_none[0], path)


def _coerce_tuple(value: object, args: tuple[object, ...], path: str) -> tuple[object, ...]:
    if len(args) != 2 or args[1] is not Ellipsis:
        raise ConfigError(f"{path}: only 'tuple[X, ...]' is supported")
    if not isinstance(value, list | tuple):
        raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
    return tuple(_coerce(item, args[0], f"{path}[{i}]") for i, item in enumerate(value))


def _expect[T](value: object, kind: type[T], path: str) -> T:
    if not isinstance(value, kind):
        raise ConfigError(f"{path}: expected {kind.__name__}, got {type(value).__name__}")
    return value


def _coerce_int(value: object, path: str) -> int:
    # bool is a subclass of int; a YAML 'true' must not become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected int, got {type(value).__name__}")
    return value


def _coerce_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path}: expected float, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path}: expected a finite number, got {value!r}")
    return result
