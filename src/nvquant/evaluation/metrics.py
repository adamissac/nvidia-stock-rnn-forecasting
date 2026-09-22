"""Performance metrics on daily simple returns (per unit of capital)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def sharpe(r: pd.Series | np.ndarray, annualize: bool = True) -> float:
    """Mean over std of daily returns (ddof=1), times sqrt(252) if annualized."""
    x = np.asarray(r, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    return float(sr * np.sqrt(ANN)) if annualize else float(sr)


def equity_curve(r: pd.Series) -> pd.Series:
    """Compounded growth of one unit of capital."""
    return (1 + r.fillna(0.0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    """Drawdown from the running peak of the equity curve (<= 0)."""
    eq = equity_curve(r)
    return eq / eq.cummax() - 1


def max_drawdown_duration(r: pd.Series) -> int:
    """Longest run of sessions spent below a previous equity peak."""
    under = (drawdown(r) < 0).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return int(longest)


def performance(
    r: pd.Series, position: pd.Series | None = None, turnover: pd.Series | None = None
) -> dict[str, float]:
    """Standard performance summary of a daily net return series."""
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"n": float(n)}
    eq = equity_curve(r)
    years = n / ANN
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(ANN))
    downside = r[r < 0]
    sortino = (
        float(r.mean() / np.sqrt((downside**2).sum() / n) * np.sqrt(ANN))
        if len(downside)
        else float("nan")
    )
    mdd = float(drawdown(r).min())
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    q95, q05 = np.quantile(r, 0.95), np.quantile(r, 0.05)
    out = {
        "n": float(n),
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe(r),
        "sortino": sortino,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "max_drawdown": mdd,
        "max_dd_duration": float(max_drawdown_duration(r)),
        "hit_rate": float((r > 0).sum() / max((r != 0).sum(), 1)),
        "profit_factor": float(gains / losses) if losses > 0 else float("nan"),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r, fisher=False)),
        "tail_ratio": float(abs(q95 / q05)) if q05 != 0 else float("nan"),
    }
    if position is not None:
        p = position.reindex(r.index)
        out["exposure"] = float((p.abs() > 1e-12).mean())
        out["avg_gross"] = float(p.abs().mean())
    if turnover is not None:
        out["turnover_ann"] = float(turnover.reindex(r.index).sum() / years)
    return out


def rolling_sharpe(r: pd.Series, window: int = 252) -> pd.Series:
    """Trailing annualized Sharpe."""
    m = r.rolling(window).mean()
    s = r.rolling(window).std()
    return (m / s * np.sqrt(ANN)).rename("rolling_sharpe")


def calendar_year_returns(r: pd.Series) -> pd.Series:
    """Compounded return per calendar year."""
    return (1 + r).groupby(pd.DatetimeIndex(r.index).year).prod() - 1
