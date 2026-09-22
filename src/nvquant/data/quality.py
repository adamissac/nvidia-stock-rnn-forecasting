"""Data-quality report: missing sessions, stale prices, outliers, and split checks.

This is the one place allowed to read lockbox-period data before Phase 10 (the
spec's exception for data-quality checks). It only computes data statistics,
never performance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nvquant.config.schema import Config
from nvquant.data.loader import LoadedData, raw_dir, resolve, ticker_filename
from nvquant.data.manifest import verify_manifest

KNOWN_SPLITS: dict[str, list[tuple[str, float]]] = {
    "NVDA": [("2021-07-20", 4.0), ("2024-06-10", 10.0)],
}
SPLIT_FACTORS = (2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0)


@dataclass(frozen=True)
class SplitCheck:
    """Price and volume continuity around one known split date."""

    ticker: str
    date: str
    factor: float
    close_ratio: float
    open_ratio: float
    volume_ratio: float
    passed: bool


@dataclass(frozen=True)
class TickerQuality:
    """Per-ticker data-quality summary."""

    ticker: str
    first: str | None
    last: str | None
    sessions_expected: int
    missing_sessions: int
    off_calendar_rows: int
    max_stale_run: int
    outliers: int
    outlier_dates: list[str]
    suspect_split_dates: list[str]


def _longest_run(values: pd.Series) -> int:
    s = values.dropna()
    if s.empty:
        return 0
    same = s.eq(s.shift())
    groups = (~same).cumsum()
    return int(same.groupby(groups).sum().max()) + 1


def check_split(frame: pd.DataFrame, ticker: str, date: str, factor: float) -> SplitCheck:
    """Compare prices and volume right before and after a split date.

    Unadjusted data would show a close ratio near ``1/factor`` and a volume level
    shift near ``factor``. Adjusted data shows ratios near 1.
    """
    d = pd.Timestamp(date)
    before = frame.loc[:d].iloc[:-1]
    after = frame.loc[d:]
    close_ratio = float(after["close"].iloc[0] / before["close"].iloc[-1])
    open_ratio = float(after["open"].iloc[0] / before["close"].iloc[-1])
    vol_ratio = float(after["volume"].iloc[:20].median() / before["volume"].iloc[-20:].median())
    tol = np.log(factor) / 2
    passed = (
        abs(np.log(close_ratio)) < tol
        and abs(np.log(open_ratio)) < tol
        and abs(np.log(vol_ratio)) < tol
    )
    return SplitCheck(ticker, date, factor, close_ratio, open_ratio, vol_ratio, bool(passed))


def _suspect_splits(frame: pd.DataFrame) -> list[str]:
    """Dates where the close jumps by about a split factor and volume shifts by the inverse."""
    r = np.log(frame["close"]).diff()
    out = []
    for date, value in r[r.abs() > np.log(1.8)].items():
        jump = float(np.exp(-value))
        if not any(abs(np.log(jump / f)) < 0.05 for f in SPLIT_FACTORS):
            continue
        pos = frame.index.get_loc(date)
        vb = frame["volume"].iloc[max(0, pos - 20) : pos].median()
        va = frame["volume"].iloc[pos : pos + 20].median()
        if vb > 0 and abs(np.log(va / vb) - np.log(jump)) < np.log(1.5):
            out.append(str(pd.Timestamp(date).date()))
    return out


def ticker_quality(
    frame: pd.DataFrame, raw_frame: pd.DataFrame, ticker: str, cfg: Config
) -> TickerQuality:
    """Quality summary for one aligned OHLCV frame."""
    close = frame["close"]
    fv, lv = close.first_valid_index(), close.last_valid_index()
    if fv is None or lv is None:
        return TickerQuality(ticker, None, None, 0, 0, 0, 0, 0, [], [])
    live = frame.loc[fv:lv]
    r = np.log(live["close"]).diff().dropna()
    mad = float((r - r.median()).abs().median()) * 1.4826
    flagged = r[(r - r.median()).abs() > cfg.data.outlier_sigma * mad] if mad > 0 else r.iloc[:0]
    off_cal = int((~raw_frame.index.isin(frame.index)).sum())
    return TickerQuality(
        ticker=ticker,
        first=str(pd.Timestamp(fv).date()),
        last=str(pd.Timestamp(lv).date()),
        sessions_expected=len(live),
        missing_sessions=int(live["close"].isna().sum()),
        off_calendar_rows=off_cal,
        max_stale_run=_longest_run(live["close"]),
        outliers=len(flagged),
        # counts cover every session (a data check), but dates are only listed before the
        # lockbox so the report doesn't point at specific post-2024 moves
        outlier_dates=[
            str(pd.Timestamp(d).date())
            for d in flagged.index[flagged.index < pd.Timestamp(cfg.lockbox.start)][:20]
        ],
        suspect_split_dates=_suspect_splits(live) if live["volume"].sum() > 0 else [],
    )


def build_report(cfg: Config, loaded: LoadedData) -> dict[str, object]:
    """Build the data-quality report as a JSON-ready dict."""
    raw = raw_dir(cfg)
    m = loaded.market
    tickers = []
    for t in m.tickers:
        raw_frame = pd.read_parquet(raw / "yahoo" / ticker_filename(t))
        tickers.append(asdict(ticker_quality(m.prices[t], raw_frame, t, cfg)))
    splits: list[dict[str, object]] = []
    if cfg.data.source == "yahoo":
        for t, events in KNOWN_SPLITS.items():
            if t in m.prices:
                splits.extend(asdict(check_split(m.prices[t], t, d, f)) for d, f in events)
    stale_flags = [
        q["ticker"] for q in tickers if int(q["max_stale_run"]) >= cfg.data.stale_run_threshold
    ]
    return {
        "mode": loaded.mode,
        "end": str(loaded.end.date()),
        "data_hash": loaded.data_hash,
        "modeling_start": str(loaded.modeling_start.date()),
        "first_valid": loaded.first_valid,
        "manifest_mismatches": verify_manifest(raw),
        "split_checks": splits,
        "split_checks_passed": all(bool(s["passed"]) for s in splits),
        "stale_tickers": stale_flags,
        "tickers": tickers,
    }


def render_markdown(report: dict[str, object]) -> str:
    """Human-readable version of the report."""
    lines = [
        "# Data-quality report",
        "",
        "Generated by `make data-report`. Every number here comes from the raw cache.",
        "",
        f"- Data hash: `{report['data_hash']}`",
        f"- Last session checked: {report['end']} (includes lockbox dates; data checks only)",
        f"- Modeling start (latest first-valid date across required inputs): **{report['modeling_start']}**",
        f"- Manifest mismatches: {len(report['manifest_mismatches'])}",  # type: ignore[arg-type]
        f"- Split checks passed: {report['split_checks_passed']}",
        "",
        "## First valid date of each required input",
        "",
        "| input | first valid |",
        "|---|---|",
    ]
    for k, v in report["first_valid"].items():  # type: ignore[attr-defined]
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Split checks",
        "",
        "| ticker | date | factor | close ratio | open ratio | volume ratio | passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report["split_checks"]:  # type: ignore[attr-defined]
        lines.append(
            f"| {s['ticker']} | {s['date']} | {s['factor']:g} | {s['close_ratio']:.3f} | "
            f"{s['open_ratio']:.3f} | {s['volume_ratio']:.3f} | {s['passed']} |"
        )
    lines += [
        "",
        "## Per ticker",
        "",
        "| ticker | first | last | sessions | missing | off-calendar rows | longest stale run | outliers | suspect splits |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for q in report["tickers"]:  # type: ignore[attr-defined]
        lines.append(
            f"| {q['ticker']} | {q['first']} | {q['last']} | {q['sessions_expected']} | "
            f"{q['missing_sessions']} | {q['off_calendar_rows']} | {q['max_stale_run']} | "
            f"{q['outliers']} | {', '.join(q['suspect_split_dates']) or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def write_report(cfg: Config, loaded: LoadedData) -> Path:
    """Write ``data_quality.json`` and ``data_quality.md`` into the reports dir."""
    report = build_report(cfg, loaded)
    out = resolve(cfg.paths.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "data_quality.md").write_text(render_markdown(report), encoding="utf-8")
    return out / "data_quality.json"
