"""Forward-return labels aligned with next-open execution.

For a decision after the close of session t, the position is filled at the
open of t+1. The h-session label is

    fwd_ret_h[t] = log(open[t+1+h] / open[t+1]),   t_end[t] = session t+1+h,

so the label, the forecast, and the strategy's PnL measure the same return.
This is the only package where negative shifts are allowed (lint LK001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.data.market import MarketData


def end_times(sessions: pd.DatetimeIndex, offset: int) -> pd.Series:
    """``sessions[i + offset]`` for each i (NaT past the end)."""
    pos = np.arange(len(sessions)) + offset
    out = pd.Series(pd.NaT, index=sessions, dtype="datetime64[ns]")
    ok = pos < len(sessions)
    out.iloc[np.flatnonzero(ok)] = sessions[pos[ok]]
    return out


def ex_ante_vol(close: pd.Series, span: int) -> pd.Series:
    """EWMA std of daily log close returns, known at the close of t (no lookahead)."""
    return np.log(close).diff().ewm(span=span, adjust=False, min_periods=span).std()


def forward_returns(market: MarketData, horizons: list[int], vol_span: int) -> pd.DataFrame:
    """Forward log returns, vol-normalized versions, and each label's end time.

    Columns: ``fwd_ret_h``, ``fwd_ret_vn_h`` (divided by ``sigma_t * sqrt(h)``),
    ``t_end_h``, and ``sigma`` (the ex-ante daily vol used for normalizing).
    """
    px = market.ohlcv()
    lo = np.log(px["open"])
    sessions = market.sessions
    sigma = ex_ante_vol(px["close"], vol_span)
    out: dict[str, pd.Series] = {"sigma": sigma}
    for h in horizons:
        fwd = lo.shift(-(1 + h)) - lo.shift(-1)
        out[f"fwd_ret_{h}"] = fwd
        out[f"fwd_ret_vn_{h}"] = fwd / (sigma * np.sqrt(h))
        out[f"t_end_{h}"] = end_times(sessions, 1 + h)
    return pd.DataFrame(out, index=sessions)


def holding_log_return(
    market: MarketData, execution: str = "next_open", ticker: str | None = None
) -> pd.Series:
    """Log return earned by a position decided at t.

    ``next_open``: ``log(O[t+2] / O[t+1])`` (the same quantity as ``fwd_ret_1``).
    ``next_close``: ``log(C[t+2] / C[t+1])``.
    """
    px = market.ohlcv(ticker)
    lp = np.log(px["open" if execution == "next_open" else "close"])
    return (lp.shift(-2) - lp.shift(-1)).rename("r_hold")
