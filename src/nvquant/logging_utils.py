"""Logging setup shared by the CLI and pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root ``nvquant`` logger once (idempotent)."""
    root = logging.getLogger("nvquant")
    root.setLevel(level)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    root.addHandler(handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    for noisy in ("yfinance", "optuna", "urllib3", "matplotlib", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``nvquant`` namespace."""
    return logging.getLogger(name if name.startswith("nvquant") else f"nvquant.{name}")
