import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from nvquant.data import sources
from nvquant.data.calendar import nyse_sessions, third_fridays
from nvquant.data.loader import earnings_event_sessions, load_market_data, write_synthetic
from nvquant.data.manifest import data_hash, read_manifest, verify_manifest
from nvquant.data.quality import build_report, check_split, render_markdown, write_report
from nvquant.data.schemas import validate_ohlcv


def test_parse_fred_csv_handles_missing():
    text = "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,\n2020-01-06,.\n2020-01-07,1.81\n"
    s = sources.parse_fred_csv(text, "DGS10")
    assert list(s) == [1.88, 1.81]


def test_parse_french_csv():
    text = "header text\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20200102,  0.86, -0.97, -0.33, -0.12, -0.21, 0.01\n20200103, -0.67, 0.30, 0.00, -0.19, -0.09, 0.01\n\n Copyright\n"
    df = sources.parse_french_csv(text)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert df.loc["2020-01-02", "mkt_rf"] == pytest.approx(0.0086)


def test_parse_earnings_timing():
    idx = pd.DatetimeIndex(
        ["2024-05-22 16:20", "2024-02-21 08:00", "2026-11-17 15:00"]
    ).tz_localize("America/New_York")
    out = sources.parse_earnings(pd.DataFrame({"EPS Estimate": [1, 2, 3]}, index=idx))
    assert list(out["timing"]) == ["bmo", "amc", "amc"]


def test_normalize_yahoo_rejects_multiindex():
    df = pd.DataFrame([[1.0]], columns=pd.MultiIndex.from_tuples([("Close", "NVDA")]))
    with pytest.raises(ValueError):
        sources.normalize_yahoo(df)
    flat = pd.DataFrame(
        {"Open": [2.0, 1.0], "High": [2, 1], "Low": [1, 1], "Close": [2, 1], "Volume": [5, 6]},
        index=pd.DatetimeIndex(["2020-01-03", "2020-01-02"]),
    )
    out = sources.normalize_yahoo(flat)
    assert out.index.is_monotonic_increasing and out.columns[0] == "open"


def test_with_retries(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("boom")
        return 5

    assert sources.with_retries(flaky, "x", retries=3, backoff=0.0) == 5
    with pytest.raises(OSError):
        sources.with_retries(
            lambda: (_ for _ in ()).throw(OSError("no")), "x", retries=2, backoff=0.0
        )


def test_schema_rejects_descending_time():
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1.0, 1.0],
        },
        index=pd.DatetimeIndex(["2020-01-03", "2020-01-02"], name="date"),
    )
    with pytest.raises(pa.errors.SchemaError):
        validate_ohlcv(df, "X")


def test_calendar():
    s = nyse_sessions("2024-06-01", "2024-07-10")
    assert pd.Timestamp("2024-06-19") not in s  # Juneteenth
    tf = third_fridays(s)
    assert pd.Timestamp("2024-06-21") in tf


def test_earnings_event_sessions():
    sessions = nyse_sessions("2024-05-20", "2024-05-24")
    frame = pd.DataFrame(
        {"timing": ["amc", "bmo"]}, index=pd.DatetimeIndex(["2024-05-22", "2024-05-24"])
    )
    out = earnings_event_sessions(frame, sessions)
    assert list(out) == [pd.Timestamp("2024-05-22"), pd.Timestamp("2024-05-23")]


def test_loader_manifest_and_quality(tmp_cfg):
    raw = write_synthetic(tmp_cfg)
    assert verify_manifest(raw) == []
    assert len(data_hash(raw)) == 12
    assert "yahoo/NVDA.parquet" in read_manifest(raw)
    dev = load_market_data(tmp_cfg, "dev")
    full = load_market_data(tmp_cfg, "lockbox")
    cut = pd.Timestamp(tmp_cfg.lockbox.start)
    assert dev.end < cut <= full.end
    assert dev.market.sessions.max() < cut and full.market.sessions.max() >= cut
    assert dev.modeling_start == pd.Timestamp(max(dev.first_valid.values()))
    report = build_report(tmp_cfg, full)
    assert report["manifest_mismatches"] == [] and report["split_checks"] == []
    assert "Data-quality report" in render_markdown(report)
    assert write_report(tmp_cfg, full).exists()
    (raw / "yahoo" / "NVDA.parquet").write_bytes(b"corrupt")
    assert verify_manifest(raw) == ["yahoo/NVDA.parquet"]


def test_missing_raw_data(tmp_cfg):
    with pytest.raises(FileNotFoundError):
        load_market_data(tmp_cfg)


def test_split_check_detects_unadjusted_split():
    idx = pd.bdate_range("2021-06-01", periods=60)
    close = np.r_[np.full(30, 400.0), np.full(30, 100.0)]
    vol = np.r_[np.full(30, 1e6), np.full(30, 4e6)]
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": vol}, index=idx
    )
    bad = check_split(df, "X", str(idx[30].date()), 4.0)
    assert not bad.passed and bad.close_ratio == pytest.approx(0.25)
    adj = df.assign(close=100.0, open=100.0, volume=4e6)
    assert check_split(adj, "X", str(idx[30].date()), 4.0).passed


def test_parse_fred_csv_refuses_a_changed_format():
    rows = "\n".join(f"2020-01-{d:02d},4.1%" for d in range(1, 29))
    with pytest.raises(ValueError, match="aren't numbers"):
        sources.parse_fred_csv("observation_date,DGS10\n" + rows + "\n", "DGS10")
