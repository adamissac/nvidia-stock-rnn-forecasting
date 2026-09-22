"""Downloaders for Yahoo Finance, FRED, and Ken French data.

Every function here touches the network; nothing in ``tests/`` calls them.
Parsing is split into pure functions (``parse_*``) so tests can cover it on
small fixtures.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

from nvquant.logging_utils import get_logger

log = get_logger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
YAHOO_URL = "https://finance.yahoo.com/quote/{ticker}"
_HEADERS = {"User-Agent": "nvquant research (github.com/adamissac/nvidia-stock-rnn-forecasting)"}


def with_retries[T](fn: Callable[[], T], what: str, retries: int, backoff: float) -> T:
    """Call ``fn`` with exponential backoff; re-raise after ``retries`` failures."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # network code: any failure is retried, the last one raised
            if attempt == retries:
                raise
            wait = backoff * 2 ** (attempt - 1)
            log.warning("%s failed (%s), retry %d/%d in %.0fs", what, exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _check_coerced(raw: pd.Series, parsed: pd.Series, what: str, max_share: float = 0.001) -> None:
    """Raise if more than ``max_share`` of the non-blank raw values failed to parse as numbers."""
    bad = int((raw.notna() & parsed.isna()).sum())
    if bad > max_share * max(len(raw), 1):
        raise ValueError(
            f"{what}: {bad} of {len(raw)} values aren't numbers; did the format change?"
        )


def normalize_yahoo(frame: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV columns on a tz-naive date index, sorted ascending, no duplicates."""
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("expected flat columns; call yf.download with multi_level_index=False")
    out = frame.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype("float64")


def download_yahoo(ticker: str, start: str, retries: int, backoff: float) -> pd.DataFrame:
    """Daily OHLCV from Yahoo, split and dividend adjusted (``auto_adjust=True``)."""
    import yfinance as yf

    def _get() -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
            threads=False,
            actions=False,
        )
        if df is None or df.empty:
            raise ValueError(f"no rows returned for {ticker}")
        return df

    return normalize_yahoo(with_retries(_get, f"yahoo {ticker}", retries, backoff))


def parse_fred_csv(text: str, series: str) -> pd.Series:
    """Parse FRED's ``fredgraph.csv`` (blank or '.' means missing)."""
    df = pd.read_csv(io.StringIO(text), na_values=[".", ""])
    date_col = df.columns[0]
    values = pd.to_numeric(df[series], errors="coerce")
    _check_coerced(df[series], values, f"FRED {series}")
    out = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"),
        name=series,
    )
    return out.dropna().sort_index()


def download_fred(series: str, retries: int, backoff: float) -> pd.Series:
    """One FRED series through the keyless CSV endpoint."""

    def _get() -> str:
        r = requests.get(FRED_URL.format(series=series), headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return r.text

    return parse_fred_csv(with_retries(_get, f"fred {series}", retries, backoff), series)


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV: find the header row, read YYYYMMDD rows, percent to decimal."""
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(","))
    header = [c.strip() for c in lines[header_idx].split(",")]
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or len(parts[0]) != 8 or not parts[0].isdigit():
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date", *header[1:]])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), format="%Y%m%d"), name="date")
    raw = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    for col in df.columns:
        _check_coerced(raw[col].replace("", np.nan), df[col], f"Ken French {col}")
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    rename = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma",
              "RF": "rf", "Mom": "mom", "MOM": "mom", "WML": "mom"}  # fmt: skip
    return df.rename(columns=lambda c: rename.get(c, c.lower()))


def download_french(name: str, retries: int, backoff: float) -> pd.DataFrame:
    """One Ken French daily dataset (``ff5`` or ``mom``)."""
    url = FRENCH_BASE + FRENCH_FILES[name]

    def _get() -> bytes:
        r = requests.get(url, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        return r.content

    content = with_retries(_get, f"french {name}", retries, backoff)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


def parse_earnings(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance ``get_earnings_dates`` output.

    Returns one row per announcement with the local announcement timestamp and a
    ``timing`` flag: ``bmo`` (before noon, so before the open in practice) or ``amc``
    (noon or later; Yahoo stamps unconfirmed future dates at 15:00, and every
    NVDA report in the sample is after the close).
    Rows with no reported EPS and a date in the future are kept (they are
    scheduled dates), but the loader only uses dates it could have known.
    """
    idx = pd.DatetimeIndex(frame.index)
    local = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
    out = pd.DataFrame({"announced": local})
    out["timing"] = np.where(local.hour < 12, "bmo", "amc")
    out["date"] = local.normalize()
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def download_earnings(ticker: str, retries: int, backoff: float, limit: int = 100) -> pd.DataFrame:
    """Historical earnings announcement dates from Yahoo (see caveats in docs/METHODOLOGY.md)."""
    import yfinance as yf

    def _get() -> pd.DataFrame:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        if df is None or df.empty:
            raise ValueError(f"no earnings dates for {ticker}")
        return df

    return parse_earnings(with_retries(_get, f"earnings {ticker}", retries, backoff))
