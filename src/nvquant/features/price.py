"""Features from the target's own OHLCV (no lag needed: known after the close of t).

Every function is pure and uses trailing windows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.data.market import MarketData
from nvquant.features.registry import FeatureParams, register

ANN = np.sqrt(252.0)


def _logret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


@register(
    "returns",
    "returns",
    lookback=63,
    description="Log close-to-close returns over several horizons, plus overnight and intraday returns.",
)
def returns(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Log returns ``ret_h = log(C_t / C_{t-h})``, overnight gap, and intraday return."""
    px = m.ohlcv()
    lc = np.log(px["close"])
    out = {f"ret_{h}": lc - lc.shift(h) for h in p.return_horizons}
    out["ret_overnight"] = np.log(px["open"] / px["close"].shift(1))
    out["ret_intraday"] = np.log(px["close"] / px["open"])
    return pd.DataFrame(out, index=m.sessions)


def realized_vols(px: pd.DataFrame, window: int) -> dict[str, pd.Series]:
    """Annualized close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang vol.

    All five are trailing-window estimators (Yang and Zhang 2000). The range
    estimators use each day's own high/low, known after that day's close.
    """
    o, h, lo, c = (np.log(px[k]) for k in ("open", "high", "low", "close"))
    r = c.diff()
    hl = h - lo
    co = c - o
    ho, lo_o = h - o, lo - o
    hc, lc_ = h - c, lo - c
    oc_prev = o - c.shift(1)
    park = (hl**2 / (4 * np.log(2))).rolling(window, min_periods=window).mean()
    gk = (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(window, min_periods=window).mean()
    rs_daily = ho * hc + lo_o * lc_
    rs = rs_daily.rolling(window, min_periods=window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_o = oc_prev.rolling(window, min_periods=window).var()
    var_c = co.rolling(window, min_periods=window).var()
    yz = var_o + k * var_c + (1 - k) * rs
    cc = r.rolling(window, min_periods=window).std()
    return {
        f"rv_cc_{window}": cc * ANN,
        f"rv_park_{window}": np.sqrt(park.clip(lower=0)) * ANN,
        f"rv_gk_{window}": np.sqrt(gk.clip(lower=0)) * ANN,
        f"rv_rs_{window}": np.sqrt(rs.clip(lower=0)) * ANN,
        f"rv_yz_{window}": np.sqrt(yz.clip(lower=0)) * ANN,
    }


@register(
    "volatility",
    "volatility",
    lookback=126,
    description="Realized volatility (close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) and vol-of-vol.",
)
def volatility(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Five realized-vol estimators per window, log-transformed, plus vol-of-vol."""
    px = m.ohlcv()
    out: dict[str, pd.Series] = {}
    for w in p.vol_windows:
        for k, v in realized_vols(px, w).items():
            out[f"log_{k}"] = np.log(v.where(v > 0))
    w0 = min(p.vol_windows)
    rv = realized_vols(px, w0)[f"rv_cc_{w0}"]
    out["volvol_63"] = np.log(rv).diff().rolling(63, min_periods=63).std()
    out["rv_ratio"] = np.log(  # type: ignore[assignment]
        rv / realized_vols(px, max(p.vol_windows))[f"rv_cc_{max(p.vol_windows)}"]
    )
    return pd.DataFrame(out, index=m.sessions)


@register(
    "momentum",
    "momentum",
    lookback=252,
    description="12-1 momentum, vol-scaled short-term reversal, distance from moving averages.",
)
def momentum(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """``mom_12_1 = log(C_{t-21}/C_{t-252})``, ``rev_z5`` (5-day return in vol units), MA distances."""
    c = m.ohlcv()["close"]
    lc = np.log(c)
    daily_vol = _logret(c).rolling(p.zscore_window, min_periods=p.zscore_window).std()
    out = {
        "mom_12_1": lc.shift(21) - lc.shift(252),
        "mom_6_1": lc.shift(21) - lc.shift(126),
        "rev_z5": -(lc - lc.shift(5)) / (daily_vol * np.sqrt(5)),
        "rev_z1": -(lc - lc.shift(1)) / daily_vol,
        "dist_ma50": lc - np.log(c.rolling(50, min_periods=50).mean()),
        "dist_ma200": lc - np.log(c.rolling(200, min_periods=200).mean()),
    }
    return pd.DataFrame(out, index=m.sessions)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI with a causal exponential average (``adjust=False``)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    down = (-d).clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = up / down.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


@register(
    "technical",
    "technical",
    lookback=60,
    description="Causal RSI(14), MACD(12,26,9) histogram, Bollinger %B(20), ATR(14).",
)
def technical(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Classic indicators computed with trailing windows and ``adjust=False`` EMAs.

    MACD and ATR are divided by the close so they are scale-free.
    """
    px = m.ohlcv()
    c = px["close"]
    ema12 = c.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    tr = pd.concat(
        [px["high"] - px["low"], (px["high"] - c.shift(1)).abs(), (px["low"] - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out = {
        "rsi_14": rsi(c, 14) / 100.0,
        "macd_hist": (macd - signal) / c,
        "bb_pctb_20": (c - (ma20 - 2 * sd20)) / (4 * sd20),
        "atr_14": atr / c,
    }
    return pd.DataFrame(out, index=m.sessions)


@register(
    "volume",
    "volume",
    lookback=273,
    description="Volume z-score, Amihud illiquidity z-score, dollar volume relative to its one-year average.",
)
def volume(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Rolling z-score of log volume, 21-day Amihud ratio as a 252-day z-score, and
    21-day log dollar volume minus its 252-day average.

    Raw Amihud and dollar volume trend over decades (NVDA's dollar volume grew
    by several orders of magnitude), so both are expressed relative to their
    own trailing year to keep them stationary.
    """
    px = m.ohlcv()
    lv = np.log(px["volume"].where(px["volume"] > 0))
    w = p.zscore_window
    dollar = px["close"] * px["volume"]
    r = _logret(px["close"]).abs()
    log_amihud = np.log((r / dollar.where(dollar > 0)).rolling(21, min_periods=21).mean() * 1e9)
    log_dv = np.log(dollar.rolling(21, min_periods=21).mean())
    year = 252
    out = {
        "volume_z": (lv - lv.rolling(w, min_periods=w).mean()) / lv.rolling(w, min_periods=w).std(),
        "amihud_z": (log_amihud - log_amihud.rolling(year, min_periods=year).mean())
        / log_amihud.rolling(year, min_periods=year).std(),
        "dollar_vol_rel": log_dv - log_dv.rolling(year, min_periods=year).mean(),
    }
    return pd.DataFrame(out, index=m.sessions)
