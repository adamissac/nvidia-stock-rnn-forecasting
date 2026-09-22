"""Config loading: ``configs/base.yaml`` deep-merged with a profile override."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from nvquant.config.schema import Config, ModelConfig, SizingConfig

__all__ = ["Config", "ModelConfig", "SizingConfig", "config_hash", "deep_merge", "load_config"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Dicts merge key by key; any other value in ``override`` replaces the base value.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def _resolve_extends(path: Path, depth: int = 0) -> dict[str, Any]:
    """Read a YAML file and deep-merge it over the chain of files it ``extends``."""
    if depth > 5:
        raise ValueError("config extends chain is too deep")
    raw = _read_yaml(path)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(_resolve_extends(path.parent / parent, depth + 1), raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a profile config.

    A profile file may set ``extends: base.yaml`` (resolved relative to itself,
    and recursively); the base is loaded first and the profile deep-merged on top.

    Parameters
    ----------
    path : str or Path
        Profile YAML file, for example ``configs/full.yaml``.
    overrides : dict, optional
        Extra values merged last (used by tests and the CLI).

    Returns
    -------
    Config
        The validated, frozen config.
    """
    raw = _resolve_extends(Path(path))
    if overrides:
        raw = deep_merge(raw, overrides)
    return Config.model_validate(raw)


def config_hash(cfg: Config | dict[str, Any], length: int = 12) -> str:
    """Stable short hash of a config (or any JSON-serializable mapping)."""
    payload = cfg.model_dump(mode="json") if isinstance(cfg, Config) else cfg
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
