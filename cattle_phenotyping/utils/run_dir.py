"""Run-directory helpers for training artifact versioning.

Each training run writes to ``runs/<timestamp>_<git_sha>/`` containing the
resolved config, metrics JSON, saved models, feature cache, and any plots.
``saved_models/`` is treated as a *pointer* to the latest accepted run.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _git_sha(repo_root: Path, short: bool = True) -> str:
    """Return the current git HEAD SHA, or ``nogit`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short" if short else "HEAD", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def make_run_dir(runs_root: str | Path, repo_root: str | Path | None = None) -> Path:
    """Create and return a fresh ``runs/<timestamp>_<sha>/`` directory."""
    runs_root = Path(runs_root)
    repo_root_path = Path(repo_root) if repo_root else runs_root.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = _git_sha(repo_root_path)
    run_dir = runs_root / f"{timestamp}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, data: Any) -> None:
    """Write a JSON file with stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
