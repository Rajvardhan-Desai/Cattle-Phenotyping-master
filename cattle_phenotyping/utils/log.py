"""Project-wide logging helpers.

Module is named ``log`` (not ``logging``) to avoid shadowing the stdlib.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str | Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure the root logger once.

    Safe to call multiple times; subsequent calls update the level and add
    an extra file handler if ``log_file`` differs from the existing ones.
    """
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(level)

    if not _CONFIGURED:
        # Drop any handlers a library may have attached pre-emptively.
        for handler in list(root.handlers):
            root.removeHandler(handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(stream_handler)
        _CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        existing = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_file.resolve()) not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Modules should call this at import time."""
    return logging.getLogger(name)
