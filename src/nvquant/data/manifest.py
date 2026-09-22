"""Data manifest: where each raw file came from, when, and its SHA256."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nvquant.experiments.repro import sha256_file, sha256_text

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ManifestEntry:
    """One raw file."""

    path: str
    source: str
    url: str
    downloaded_at: str
    sha256: str
    rows: int
    first: str | None
    last: str | None


def entry_for(
    raw_dir: Path, file: Path, source: str, url: str, frame: pd.DataFrame
) -> ManifestEntry:
    """Build a manifest entry for a parquet file that was just written."""
    idx = pd.DatetimeIndex(frame.index) if len(frame) else pd.DatetimeIndex([])
    return ManifestEntry(
        path=file.relative_to(raw_dir).as_posix(),
        source=source,
        url=url,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sha256=sha256_file(file),
        rows=len(frame),
        first=str(idx.min().date()) if len(idx) else None,
        last=str(idx.max().date()) if len(idx) else None,
    )


def write_manifest(raw_dir: Path, entries: list[ManifestEntry]) -> Path:
    """Write ``manifest.json`` (entries sorted by path)."""
    path = raw_dir / MANIFEST_NAME
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [asdict(e) for e in sorted(entries, key=lambda e: e.path)],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(raw_dir: Path) -> dict[str, ManifestEntry]:
    """Read the manifest into ``{path: entry}``."""
    payload = json.loads((raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return {e["path"]: ManifestEntry(**e) for e in payload["files"]}


def data_hash(raw_dir: Path) -> str:
    """Short hash over every file's SHA256, so any data change changes the tag."""
    entries = read_manifest(raw_dir)
    joined = "\n".join(f"{p}:{e.sha256}" for p, e in sorted(entries.items()))
    return sha256_text(joined)[:12]


def verify_manifest(raw_dir: Path) -> list[str]:
    """Return the paths whose bytes no longer match the manifest."""
    bad = []
    for path, entry in read_manifest(raw_dir).items():
        file = raw_dir / path
        if not file.exists() or sha256_file(file) != entry.sha256:
            bad.append(path)
    return bad
