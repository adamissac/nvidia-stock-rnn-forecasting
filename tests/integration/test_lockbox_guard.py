"""The lockbox refuses to open without a committed preregistration, and records every opening."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from nvquant.config import load_config
from nvquant.cv.lockbox import LockboxError
from nvquant.experiments import pipeline

ROOT = Path(__file__).resolve().parents[2]


def test_require_committed_refuses_dirty_files(monkeypatch):
    class Dirty:
        returncode = 1

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Dirty())
    with pytest.raises(LockboxError, match="must be committed"):
        pipeline._require_committed(["docs/PREREGISTRATION.md"])


def test_lockbox_runs_once_then_needs_a_reason(tmp_path):
    cfg = load_config(ROOT / "configs/fast.yaml", overrides={"paths": {
        "data_dir": str(tmp_path / "data"), "reports_dir": str(tmp_path / "reports"),
        "docs_dir": str(tmp_path / "docs"), "readme": str(tmp_path / "README.md")}})
    pipeline.stage_data(cfg)
    out = pipeline.stage_lockbox(cfg)
    res = json.loads(out.read_text())
    assert pd.Timestamp(res["start"]) >= pd.Timestamp(cfg.lockbox.start)
    sentinel = json.loads((tmp_path / "reports/lockbox/SENTINEL.json").read_text())
    assert sentinel["n_runs"] == 1 and sentinel["runs"][0]["status"] == "completed"
    with pytest.raises(LockboxError, match="already evaluated"):
        pipeline.stage_lockbox(cfg)
    with pytest.raises(LockboxError, match="non-empty --reason"):
        pipeline.stage_lockbox(cfg, force=True, reason="")
    pipeline.stage_lockbox(cfg, force=True, reason="synthetic data regenerated")
    sentinel = json.loads((tmp_path / "reports/lockbox/SENTINEL.json").read_text())
    assert sentinel["n_runs"] == 2 and sentinel["runs"][1]["forced"]
    assert sentinel["runs"][1]["reason"] == "synthetic data regenerated"
