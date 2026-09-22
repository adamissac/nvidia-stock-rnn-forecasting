"""Write the raw cache and load it back as aligned, validated :class:`MarketData`.

Raw layout under ``<data_dir>/raw``::

    yahoo/<TICKER>.parquet     OHLCV as downloaded (tz-naive dates, ascending)
    fred/<SERIES>.parquet      one column ``value``
    french/ff5.parquet, mom.parquet
    earnings/<TICKER>.parquet  columns announced, timing, date
    manifest.json

The same layout is used for synthetic data, so every later stage runs the same
code in the fast and full profiles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from nvquant.config import PROJECT_ROOT
from nvquant.config.schema import Config
from nvquant.data import sources
from nvquant.data.calendar import nyse_sessions
from nvquant.data.manifest import (
    ManifestEntry,
    data_hash,
    entry_for,
    read_manifest,
    write_manifest,
)
from nvquant.data.market import MarketData
from nvquant.data.schemas import FACTORS_SCHEMA, RATES_SCHEMA, validate_ohlcv
from nvquant.data.synthetic import synthetic_market
from nvquant.logging_utils import get_logger

log = get_logger(__name__)
Mode = Literal["dev", "lockbox"]
FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
RATE_FFILL_LIMIT = 5


def resolve(path: Path) -> Path:
    """Resolve a config path against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def raw_dir(cfg: Config) -> Path:
    """``<data_dir>/raw`` for this profile."""
    return resolve(cfg.paths.data_dir) / "raw"


def ticker_filename(ticker: str) -> str:
    """File-safe name for a ticker (``^VIX`` becomes ``IDX_VIX``)."""
    return ticker.replace("^", "IDX_") + ".parquet"


def _write(raw: Path, rel: str, frame: pd.DataFrame, source: str, url: str) -> ManifestEntry:
    """Write one parquet file and return its manifest entry."""
    file = raw / rel
    file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(file)
    return entry_for(raw, file, source, url, frame)


def download_all(cfg: Config, refresh: bool = False) -> Path:
    """Download every raw input into the parquet cache and write the manifest.

    Files already in the cache (with a manifest entry) are kept unless
    ``refresh`` is set, so a rerun doesn't hit the network and the data hash
    stays the same. Returns the raw dir.
    """
    raw = raw_dir(cfg)
    d = cfg.data
    previous = read_manifest(raw) if (raw / "manifest.json").exists() else {}
    entries: list[ManifestEntry] = []
    start = str(d.download_start)

    def fetch(rel: str, source: str, url: str, get: Callable[[], pd.DataFrame]) -> None:
        if not refresh and rel in previous and (raw / rel).exists():
            entries.append(previous[rel])
            return
        log.info("downloading %s", rel)
        entries.append(_write(raw, rel, get(), source, url))
        write_manifest(raw, entries + [e for p, e in previous.items() if p not in
                                       {x.path for x in entries}])  # fmt: skip

    for ticker in cfg.universe.all_tickers():
        fetch(
            f"yahoo/{ticker_filename(ticker)}", "yahoo", sources.YAHOO_URL.format(ticker=ticker),
            lambda t=ticker: validate_ohlcv(  # type: ignore[misc]
                sources.download_yahoo(t, start, d.max_retries, d.backoff_seconds), t),
        )  # fmt: skip
    for series in d.fred_series:
        fetch(
            f"fred/{series}.parquet", "fred", sources.FRED_URL.format(series=series),
            lambda s=series: sources.download_fred(  # type: ignore[misc]
                s, d.max_retries, d.backoff_seconds).to_frame("value"),
        )  # fmt: skip
    for name in sources.FRENCH_FILES:
        fetch(
            f"french/{name}.parquet", "ken_french", sources.FRENCH_BASE + sources.FRENCH_FILES[name],
            lambda n=name: sources.download_french(n, d.max_retries, d.backoff_seconds),  # type: ignore[misc]
        )  # fmt: skip
    for ticker in [cfg.universe.target, *cfg.universe.peers]:
        fetch(
            f"earnings/{ticker_filename(ticker)}", "yahoo", "yfinance Ticker.get_earnings_dates",
            lambda t=ticker: sources.download_earnings(  # type: ignore[misc]
                t, d.max_retries, d.backoff_seconds).set_index("date"),
        )  # fmt: skip
    write_manifest(raw, entries)
    return raw


def write_synthetic(cfg: Config) -> Path:
    """Generate the synthetic market and write it in the raw layout."""
    raw = raw_dir(cfg)
    s = cfg.data.synthetic
    m = synthetic_market(cfg.universe, s.n_sessions, s.start, s.signal_strength, cfg.seed)
    entries = []
    for ticker, df in m.prices.items():
        entries.append(_write(raw, f"yahoo/{ticker_filename(ticker)}", df, "synthetic", "synthetic"))
    for col in m.rates.columns:
        entries.append(
            _write(raw, f"fred/{col}.parquet", m.rates[[col]].rename(columns={col: "value"}),
                   "synthetic", "synthetic")  # fmt: skip
        )
    ff5 = m.factors[["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]]
    entries.append(_write(raw, "french/ff5.parquet", ff5, "synthetic", "synthetic"))
    entries.append(_write(raw, "french/mom.parquet", m.factors[["mom"]], "synthetic", "synthetic"))
    for ticker, idx in m.earnings.items():
        e = pd.DataFrame(
            {"announced": idx + pd.Timedelta(hours=16, minutes=20), "timing": "amc"},
            index=pd.DatetimeIndex(idx, name="date"),
        )
        entries.append(
            _write(raw, f"earnings/{ticker_filename(ticker)}", e, "synthetic", "synthetic")
        )
    write_manifest(raw, entries)
    return raw


def earnings_event_sessions(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map announcements to the session whose close-to-next-open gap contains them.

    After-close (``amc``) announcements on day d map to d. Before-open (``bmo``)
    announcements on day d map to the session before d.
    """
    out = []
    for date, timing in zip(pd.DatetimeIndex(frame.index), frame["timing"], strict=True):
        if timing == "amc":
            pos = sessions.searchsorted(date, side="right") - 1
        else:
            pos = sessions.searchsorted(date, side="left") - 1
        if 0 <= pos < len(sessions):
            out.append(sessions[int(pos)])
    return pd.DatetimeIndex(sorted(set(out)))


@dataclass(frozen=True)
class LoadedData:
    """Market data plus metadata about how it was loaded."""

    market: MarketData
    mode: Mode
    modeling_start: pd.Timestamp
    first_valid: dict[str, str]
    end: pd.Timestamp
    data_hash: str


def load_market_data(cfg: Config, mode: Mode = "dev") -> LoadedData:
    """Load the raw cache as aligned :class:`MarketData`.

    Parameters
    ----------
    cfg : Config
        Profile config.
    mode : {"dev", "lockbox"}
        ``dev`` (the default, used in every phase before 10) truncates every
        series at the last session before ``lockbox.start``. ``lockbox`` returns
        everything; only the lockbox stage uses it.

    Returns
    -------
    LoadedData
    """
    raw = raw_dir(cfg)
    if not (raw / "manifest.json").exists():
        raise FileNotFoundError(f"no raw data in {raw}; run `make data` first")
    frames = {}
    for ticker in cfg.universe.all_tickers():
        frames[ticker] = validate_ohlcv(
            pd.read_parquet(raw / "yahoo" / ticker_filename(ticker)), ticker
        )
    target = frames[cfg.universe.target]
    sessions = nyse_sessions(target.index.min(), target.index.max())
    prices = {t: df.reindex(sessions) for t, df in frames.items()}

    rate_cols = {}
    for series in cfg.data.fred_series:
        s = pd.read_parquet(raw / "fred" / f"{series}.parquet")["value"]
        rate_cols[series] = s.reindex(sessions.union(s.index)).ffill(limit=RATE_FFILL_LIMIT)
    rates = RATES_SCHEMA.validate(pd.DataFrame(rate_cols).reindex(sessions))

    ff5 = pd.read_parquet(raw / "french" / "ff5.parquet")
    mom = pd.read_parquet(raw / "french" / "mom.parquet")
    factors = FACTORS_SCHEMA.validate(ff5.join(mom, how="outer")[FACTOR_COLS].reindex(sessions))

    earnings = {}
    for ticker in [cfg.universe.target, *cfg.universe.peers]:
        file = raw / "earnings" / ticker_filename(ticker)
        if file.exists():
            earnings[ticker] = earnings_event_sessions(pd.read_parquet(file), sessions)

    market = MarketData(prices, rates, factors, earnings, cfg.universe.target)

    first_valid: dict[str, str] = {}
    for name in cfg.universe.required:
        values = rates[name] if name in rates.columns else prices[name]["close"]
        fv = values.first_valid_index()
        if fv is None:
            raise ValueError(f"required input {name} has no valid data")
        first_valid[name] = str(pd.Timestamp(fv).date())
    modeling_start = pd.Timestamp(max(first_valid.values()))

    lockbox_start = pd.Timestamp(cfg.lockbox.start)
    if mode == "dev":
        end = sessions[sessions < lockbox_start].max()
        market = market.truncate(end)
    else:
        end = sessions.max()
    log.info("loaded %s mode: %s to %s, modeling start %s", mode, sessions.min().date(),
             end.date(), modeling_start.date())  # fmt: skip
    return LoadedData(market, mode, modeling_start, first_valid, end, data_hash(raw))
