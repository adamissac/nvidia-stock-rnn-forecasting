"""Return attribution with Newey-West t-stats, to separate alpha from beta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nvquant.evaluation.forecast import newey_west_lags


def regress(excess: pd.Series, factors: pd.DataFrame, periods_per_year: int = 252) -> dict[str, object]:
    """OLS of excess returns on factors with HAC (Newey-West) standard errors.

    Returns annualized alpha (times ``periods_per_year``), its t-stat, betas with
    t-stats, R^2, and n.
    """
    df = pd.concat([excess.rename("y"), factors], axis=1).dropna()
    if len(df) < 30:
        return {"n": len(df)}
    fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))}
    )
    betas = {k: {"beta": float(fit.params[k]), "t": float(fit.tvalues[k])} for k in factors.columns}
    return {
        "alpha_ann": float(fit.params["const"] * periods_per_year),
        "alpha_t": float(fit.tvalues["const"]),
        "betas": betas,
        "r2": float(fit.rsquared),
        "n": len(df),
    }


def monthly(simple: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound daily simple returns into calendar-month returns."""
    idx = pd.DatetimeIndex(simple.index).to_period("M")
    return (1 + simple).groupby(idx).prod() - 1


def attribution_suite(
    strategy: pd.Series, rf: pd.Series, etf_returns: pd.DataFrame, ff: pd.DataFrame
) -> dict[str, object]:
    """Three regressions: on QQQ, on SMH, and on the Fama-French five factors plus momentum.

    QQQ and SMH use their own open-to-open returns over the strategy's exact
    holding periods, so the daily regression is aligned. Fama-French factors
    are close-to-close, which only half-overlaps an open-to-open day; a daily
    regression shrinks the market beta and pushes that premium into the
    intercept. So the factor regression runs on calendar-month compounded
    returns, where the offset only matters at the month edges, and alpha is
    annualized by 12.
    """
    ex = strategy - rf.reindex(strategy.index).fillna(0.0)
    out: dict[str, object] = {}
    for col in ("QQQ", "SMH"):
        if col in etf_returns:
            mkt = (etf_returns[col] - rf.reindex(etf_returns.index).fillna(0.0)).rename(
                f"{col}_excess"
            )
            out[col] = regress(ex, mkt.to_frame())
    ff_cols = [c for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom") if c in ff.columns]
    both = pd.concat([strategy.rename("s"), rf.rename("rf"), ff[ff_cols]], axis=1).dropna()
    m = monthly(both)
    out["FF5_MOM"] = regress(m["s"] - m["rf"], m[ff_cols], periods_per_year=12) | {"frequency": "monthly"}
    return out


def align_factors_to_holding(
    ff_daily: pd.DataFrame, dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Map daily factor returns to holding periods.

    A position decided at t is held from open(t+1) to open(t+2), which straddles
    sessions t+1 and t+2. Factors are close-to-close, so the factor return of
    session t+1 is the closest single-day match. The two only half overlap,
    which is why :func:`attribution_suite` compounds both to monthly returns
    before regressing.
    """
    nxt = pd.Series(sessions[1:], index=sessions[:-1])
    mapped = ff_daily.reindex(nxt.reindex(dates).to_numpy())
    mapped.index = dates
    return mapped


def to_simple(log_returns: pd.Series) -> pd.Series:
    """Log to simple returns."""
    return np.expm1(log_returns)
