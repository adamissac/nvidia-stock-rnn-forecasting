"""Append-only experiment registry (JSON lines).

Every fitted configuration is one row: model runs, tuning trials, strategy
configurations, and the lockbox run. The Deflated Sharpe Ratio reads its trial
count and the cross-trial Sharpe variance from here. Rows are never deleted.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nvquant.config import config_hash

Kind = Literal["model", "tuning", "strategy", "vol", "lockbox", "peer", "v1_replica", "meta"]
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Trial:
    """One registry row."""

    kind: Kind
    name: str
    config: dict[str, Any]
    git_sha: str
    data_hash: str
    run_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def config_hash(self) -> str:
        """Hash of this trial's own config."""
        return config_hash(self.config)


class Registry:
    """JSONL registry at ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, trial: Trial) -> Trial:
        """Append one trial."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(trial) | {"config_hash": trial.config_hash}
        line = json.dumps(row, sort_keys=True, default=str)
        with _LOCK, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return trial

    def rows(self) -> Iterator[dict[str, Any]]:
        """Every row, oldest first."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def select(self, kind: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Rows filtered by kind and/or run id."""
        return [
            r
            for r in self.rows()
            if (kind is None or r["kind"] == kind) and (run_id is None or r["run_id"] == run_id)
        ]

    def latest_run_id(self, kind: str) -> str | None:
        """Run id of the most recent row of a kind."""
        rows = self.select(kind)
        return rows[-1]["run_id"] if rows else None
