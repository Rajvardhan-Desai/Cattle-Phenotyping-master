"""Tests for the YAML config loader."""

from pathlib import Path

import pytest

from cattle_phenotyping.utils.config import Config, load_config, project_root


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.target_width > 0
    assert "data_dir" in cfg.paths


def test_default_config_paths_are_absolute():
    cfg = load_config()
    for key, value in cfg.paths.items():
        assert Path(value).is_absolute(), f"{key} path should be absolute, got {value}"


def test_default_config_paths_resolve_under_project_root():
    cfg = load_config()
    root = project_root()
    for key, value in cfg.paths.items():
        assert Path(value).is_relative_to(root) or Path(value).resolve() == Path(value), (
            f"{key} path is unexpected: {value}"
        )


def test_seed_default():
    cfg = load_config()
    assert cfg.seed == 42


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_override_config(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        "pipeline:\n  target_width: 512\ntraining:\n  seed: 7\n",
        encoding="utf-8",
    )
    cfg = load_config(str(override))
    assert cfg.target_width == 512
    assert cfg.seed == 7
