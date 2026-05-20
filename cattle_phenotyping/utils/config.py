"""Config loader.

Reads a YAML file into a nested dict, then exposes typed accessors via the
:class:`Config` dataclass. The loader resolves relative paths against the
project root so behavior is identical regardless of the caller's CWD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILENAME = "configs/default.yaml"


def project_root() -> Path:
    """Return the project root (parent of the ``cattle_phenotyping`` pkg)."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_path(value: Any, root: Path) -> Any:
    """Resolve a single path value against ``root`` if it's a relative path string."""
    if not isinstance(value, str):
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str(root / p)


def _resolve_paths_section(paths: dict[str, Any], root: Path) -> dict[str, Any]:
    return {key: _resolve_path(val, root) for key, val in paths.items()}


@dataclass
class Config:
    """Typed view over the YAML config."""

    raw: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    # ------------------------------------------------------------------ paths
    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})

    def path(self, key: str) -> str:
        try:
            return self.paths[key]
        except KeyError as exc:
            raise KeyError(f"Missing path '{key}' in config (source: {self.source})") from exc

    # --------------------------------------------------------------- pipeline
    @property
    def target_width(self) -> int:
        return int(self.raw.get("pipeline", {}).get("target_width", 1024))

    # --------------------------------------------------------------- detector
    @property
    def detector(self) -> dict[str, Any]:
        return self.raw.get("detector", {})

    # -------------------------------------------------------------- segmenter
    @property
    def segmenter(self) -> dict[str, Any]:
        return self.raw.get("segmenter", {})

    # ------------------------------------------------------------- trait_model
    @property
    def trait_model(self) -> dict[str, Any]:
        return self.raw.get("trait_model", {})

    # ---------------------------------------------------------------- training
    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def seed(self) -> int:
        return int(self.training.get("seed", 42))

    # ---------------------------------------------------------------- logging
    @property
    def logging(self) -> dict[str, Any]:
        return self.raw.get("logging", {})


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a config YAML.

    Args:
        path: Path to a YAML file. If ``None``, loads
            ``<project_root>/configs/default.yaml``.

    Returns:
        A :class:`Config` with all path entries under ``paths:`` resolved
        relative to the project root.
    """
    root = project_root()
    if path is None:
        source = root / DEFAULT_CONFIG_FILENAME
    else:
        candidate = Path(path)
        source = candidate if candidate.is_absolute() else root / candidate

    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")

    with source.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "paths" in raw and isinstance(raw["paths"], dict):
        raw["paths"] = _resolve_paths_section(raw["paths"], root)

    return Config(raw=raw, source=source)
