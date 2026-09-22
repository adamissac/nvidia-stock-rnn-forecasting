"""Lockbox guard: the post-2024 data is evaluated exactly once.

The sentinel file records when the lockbox was opened, on which git commit,
and with which data hash. A second run is refused unless it is forced with a
written reason, and every forced rerun is appended to the sentinel so the
history is visible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class LockboxError(RuntimeError):
    """Raised when the lockbox would be evaluated a second time without --force."""


def sentinel_path(reports_dir: Path) -> Path:
    """Location of the sentinel file."""
    return reports_dir / "lockbox" / "SENTINEL.json"


def check_can_open(reports_dir: Path, force: bool, reason: str | None) -> None:
    """Raise :class:`LockboxError` if the lockbox was already opened and this isn't a forced rerun."""
    path = sentinel_path(reports_dir)
    if not path.exists():
        return
    if not force:
        opened = json.loads(path.read_text(encoding="utf-8"))["runs"][0]["opened_at"]
        raise LockboxError(
            f"lockbox already evaluated at {opened} (see {path}); rerun needs --force --reason"
        )
    if not reason or not reason.strip():
        raise LockboxError("--force needs a non-empty --reason")


def record_open(
    reports_dir: Path, git_sha: str, data_hash: str, config_hash: str, reason: str | None
) -> Path:
    """Append this evaluation to the sentinel (creating it on the first run)."""
    path = sentinel_path(reports_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    payload["runs"].append(
        {
            "opened_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "forced": bool(payload["runs"]),
            "reason": reason,
            "status": "started",
        }
    )
    payload["n_runs"] = len(payload["runs"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def mark_completed(reports_dir: Path) -> Path:
    """Mark the latest run as completed (it was recorded as started before any computation)."""
    path = sentinel_path(reports_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"][-1]["status"] = "completed"
    payload["runs"][-1]["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
