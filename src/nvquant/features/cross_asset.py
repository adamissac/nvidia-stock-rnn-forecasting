"""Cross-asset and macro features. All are exogenous, so the registry lags them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.data.market import MarketData
from nvquant.features.registry import FeatureParams, register


def _lr(m: MarketData, ticker: str) -> pd.Series:
    return np.log(m.prices[ticker]["close"]).diff()


def peer_tickers(m: MarketData) -> list[str]:
    """Other single stocks in the universe (every ticker with earnings dates, minus the target)."""
    return [t for t in m.earnings if t != m.target and t in m.prices]


@register(
    "beta_corr",
    "cross_asset",
    lookback=63,
    exogenous=True,
    description="Rolling 63-day beta and correlation of the target to SMH and QQQ.",
)
def beta_corr(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Rolling OLS beta and Pearson correlation of daily log returns."""
    r = _lr(m, m.target)
    w = p.beta_window
    out = {}
    for etf in ("SMH", "QQQ"):
        if etf not in m.prices:
            continue
        x = _lr(m, etf)
        cov = r.rolling(w, min_periods=w).cov(x)
        out[f"beta_{etf.lower()}"] = cov / x.rolling(w, min_periods=w).var()
        out[f"corr_{etf.lower()}"] = r.rolling(w, min_periods=w).corr(x)
    return pd.DataFrame(out, index=m.sessions)


@register(
    "peers",
    "cross_asset",
    lookback=21,
    exogenous=True,
    description="Relative strength versus the peer average and cross-sectional peer dispersion.",
)
def peers(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """``rs_peers_h`` = target h-day return minus the mean peer return; ``peer_disp_21``.

    Peers that haven't started trading yet (AVGO before 2009) are skipped at
    those dates, which is what a real-time system would have seen.
    """
    names = peer_tickers(m)
    lc: pd.DataFrame = np.log(m.field("close", [m.target, *names]))  # type: ignore[assignment]
    out = {}
    for h in (5, 21):
        rh = lc - lc.shift(h)
        peer_mean = rh[names].mean(axis=1, skipna=True)
        out[f"rs_peers_{h}"] = rh[m.target] - peer_mean
    r21 = lc[names] - lc[names].shift(21)
    out["peer_disp_21"] = r21.std(axis=1, skipna=True)
    out["peer_ret_1"] = (lc[names] - lc[names].shift(1)).mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=m.sessions)


@register(
    "etf_returns",
    "cross_asset",
    lookback=21,
    exogenous=True,
    description="Recent SMH, QQQ, and SPY log returns.",
)
def etf_returns(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """SMH 1- and 5-day, QQQ 5-day, SPY 21-day log returns."""
    out = {}
    for ticker, h in (("SMH", 1), ("SMH", 5), ("QQQ", 5), ("SPY", 21)):
        if ticker in m.prices:
            lc = np.log(m.prices[ticker]["close"])
            out[f"{ticker.lower()}_ret_{h}"] = lc - lc.shift(h)
    return pd.DataFrame(out, index=m.sessions)


@register(
    "macro",
    "macro",
    lookback=21,
    exogenous=True,
    description="VIX level and change, VIX/VIX3M term structure, 10y-2y slope, rate changes.",
)
def macro(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """Market-observed macro state. Treasury yields are H.15 values, published a day late,
    which the exogenous lag covers.
    """
    vix = m.prices["^VIX"]["close"]
    out = {
        "vix_log": np.log(vix),
        "vix_chg_5": np.log(vix) - np.log(vix.shift(5)),
    }
    if "^VIX3M" in m.prices:
        out["vix_term"] = np.log(vix / m.prices["^VIX3M"]["close"])
    rates = m.rates
    if {"DGS10", "DGS2"} <= set(rates.columns):
        out["slope_10_2"] = rates["DGS10"] - rates["DGS2"]
        out["dgs10_chg_21"] = rates["DGS10"] - rates["DGS10"].shift(21)
        out["dgs2_chg_21"] = rates["DGS2"] - rates["DGS2"].shift(21)
    if "^TNX" in m.prices:
        tnx = m.prices["^TNX"]["close"]
        out["tnx_chg_5"] = tnx - tnx.shift(5)
    return pd.DataFrame(out, index=m.sessions)
