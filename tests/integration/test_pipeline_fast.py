"""End-to-end run of every stage on synthetic data, offline, through the CLI."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from nvquant.cli import app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fast_yaml(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("fast")
    cfg = {
        "extends": str(ROOT / "configs" / "fast.yaml"),
        "paths": {
            "data_dir": str(tmp / "data"),
            "reports_dir": str(tmp / "reports"),
            "docs_dir": str(tmp / "docs"),
            "readme": str(tmp / "README.md"),
        },
    }
    path = tmp / "fast_tmp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_full_fast_pipeline(fast_yaml):
    runner = CliRunner()
    info = runner.invoke(app, ["info", "--config", str(fast_yaml)])
    assert info.exit_code == 0 and json.loads(info.stdout)["profile"] == "fast"
    res = runner.invoke(app, ["reproduce", "--config", str(fast_yaml)])
    assert res.exit_code == 0, res.output
    base = fast_yaml.parent
    results = json.loads((base / "reports" / "results.json").read_text())
    assert results["meta"]["engine_max_abs_diff"] < 1e-10
    assert results["meta"]["n_strategy_trials"] > 50
    assert (base / "reports" / "tearsheet.html").stat().st_size > 10_000
    assert "<!-- RESULTS:START -->" in (base / "README.md").read_text()
    assert (base / "docs" / "FEATURES.md").exists()
    rows = [
        json.loads(line)
        for line in (base / "reports" / "registry" / "trials.jsonl").read_text().splitlines()
    ]
    kinds = {r["kind"] for r in rows}
    assert {"model", "tuning", "strategy", "vol", "v1_replica", "peer"} <= kinds
    # the synthetic target has a planted reversal signal, so a linear model should find it
    assert max(results["forecasts"][m]["ic"] for m in ("ridge", "elastic_net")) > 0.05

    # the lockbox opens once, and only once
    lock = runner.invoke(app, ["lockbox", "--config", str(fast_yaml)])
    assert lock.exit_code == 0, lock.output
    again = runner.invoke(app, ["lockbox", "--config", str(fast_yaml)])
    assert again.exit_code != 0 and "already evaluated" in str(again.exception)
    forced = runner.invoke(app, ["lockbox", "--config", str(fast_yaml), "--force"])
    assert forced.exit_code != 0
    report = runner.invoke(app, ["report", "--config", str(fast_yaml)])
    assert report.exit_code == 0
    results = json.loads((base / "reports" / "results.json").read_text())
    assert results["lockbox_runs"] == 1 and results["lockbox"]["n_days"] > 0


def test_app_renders_from_reports(fast_yaml, monkeypatch):
    """The Streamlit app renders the reports the pipeline test just produced."""
    from streamlit.testing.v1 import AppTest

    reports = fast_yaml.parent / "reports"
    if not (reports / "results.json").exists():
        pytest.skip("pipeline test did not run")
    monkeypatch.setenv("NVQUANT_REPORTS", str(reports))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("out-of-sample strategies")
