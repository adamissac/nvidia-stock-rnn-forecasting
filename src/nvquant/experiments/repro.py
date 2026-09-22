"""Reproducibility helpers: seeding, determinism, device, git SHA, hashing."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np

from nvquant.config import PROJECT_ROOT


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch, and turn on deterministic torch kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (legacy global state some libraries still read)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def detect_device(preference: str = "cpu") -> str:
    """Return the torch device to use.

    ``cpu`` always returns ``"cpu"`` (the default, for determinism). ``auto``
    returns ``"cuda"`` or ``"mps"`` when available.
    """
    if preference == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_accelerators() -> dict[str, bool]:
    """Which accelerators torch can see (logged in the registry, not used by default)."""
    import torch

    return {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}


CODE_PATHS = ("src", "configs", "app", "pyproject.toml", "uv.lock")


def git_sha(short: bool = True) -> str:
    """Current git commit, with ``-dirty`` if code or config has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short" if short else "--verify", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # "dirty" means the code or config that produced an artifact wasn't committed;
        # regenerated outputs (reports/, generated docs) don't count
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
