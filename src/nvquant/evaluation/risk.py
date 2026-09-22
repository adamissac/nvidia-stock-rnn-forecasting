"""Risk: VaR/CVaR with backtests, stress windows, regime splits, and bootstrap Monte Carlo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nvquant.evaluation.metrics import performance


def var_historical(r: pd.Series, level: float, window: int) -> pd.Series:
    """Rolling historical VaR (a positive loss number) from returns up to ``r[t]``."""
    return -r.rolling(window, min_periods=window).quantile(1 - level)


def var_parametric(r: pd.Series, level: float, window: int) -> pd.Series:
    """Rolling Gaussian VaR."""
    z = stats.norm.ppf(1 - level)
    return -(
        r.rolling(window, min_periods=window).mean()
        + z * r.rolling(window, min_periods=window).std()
    )


def var_cornish_fisher(r: pd.Series, level: float, window: int) -> pd.Series:
    """Rolling Cornish-Fisher VaR (skew and kurtosis adjusted quantile)."""
    z = stats.norm.ppf(1 - level)
    roll = r.rolling(window, min_periods=window)
    s, k = roll.skew(), roll.kurt()
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return -(roll.mean() + zcf * roll.std())


def var_garch(
    position: pd.Series, var_forecast: pd.Series, level: float, nu: float = 5.0
) -> pd.Series:
    """VaR from the variance forecast: ``|position| * sigma_t * q_t``, with a unit-variance t quantile."""
    q = stats.t.ppf(1 - level, nu) * np.sqrt((nu - 2) / nu)
    return -(position.abs() * np.sqrt(var_forecast.reindex(position.index)) * q)


def cvar_historical(r: pd.Series, level: float, window: int) -> pd.Series:
    """Rolling expected shortfall beyond the historical VaR."""

    def es(x: np.ndarray) -> float:
        cut = np.quantile(x, 1 - level)
        tail = x[x <= cut]
        return float(-tail.mean()) if len(tail) else float("nan")

    return r.rolling(window, min_periods=window).apply(es, raw=True)


@dataclass(frozen=True)
class VaRBacktest:
    """Kupiec and Christoffersen tests of VaR exceptions."""

    n: int
    exceptions: int
    rate: float
    expected: float
    kupiec_pvalue: float
    christoffersen_ind_pvalue: float
    christoffersen_cc_pvalue: float


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def var_backtest(r: pd.Series, var: pd.Series, level: float, lag: int = 2) -> VaRBacktest:
    """Exceptions are days whose loss exceeds the VaR that was known at the decision.

    ``r`` is indexed by decision date, and ``r[t]`` is realized at the open of
    t+2. A rolling VaR computed on returns up to ``r[t]`` is therefore only
    available for the decision at t+2, so rolling methods use ``lag=2``. The
    GARCH VaR is built from the position and variance forecast at t, so it
    applies to ``r[t]`` directly (``lag=0``).

    Kupiec (1995) tests the exception rate; Christoffersen (1998) tests
    independence of exceptions and conditional coverage.
    """
    df = pd.concat({"r": r, "v": var.shift(lag)}, axis=1).dropna()
    df = df[df["v"] > 0]
    hit = (-df["r"] > df["v"]).astype(int).to_numpy()
    n, x = len(hit), int(hit.sum())
    p = 1 - level
    if n < 20:
        return VaRBacktest(n, x, float("nan"), p, float("nan"), float("nan"), float("nan"))
    pi = x / n
    lr_uc = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    n00 = int(((hit[:-1] == 0) & (hit[1:] == 0)).sum())
    n01 = int(((hit[:-1] == 0) & (hit[1:] == 1)).sum())
    n10 = int(((hit[:-1] == 1) & (hit[1:] == 0)).sum())
    n11 = int(((hit[:-1] == 1) & (hit[1:] == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l_null = _xlogy(n00 + n10, 1 - p1) + _xlogy(n01 + n11, p1)
    l_alt = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr_ind = -2 * (l_null - l_alt)
    return VaRBacktest(
        n,
        x,
        pi,
        p,
        float(stats.chi2.sf(lr_uc, 1)),
        float(stats.chi2.sf(lr_ind, 1)),
        float(stats.chi2.sf(lr_uc + lr_ind, 2)),
    )


def var_suite(
    r: pd.Series, position: pd.Series, var_forecast: pd.Series, levels: list[float], window: int
) -> dict[str, object]:
    """All four VaR methods at each level, with backtests and average CVaR."""
    out: dict[str, object] = {}
    for lvl in levels:
        methods = {
            "historical": var_historical(r, lvl, window),
            "parametric": var_parametric(r, lvl, window),
            "cornish_fisher": var_cornish_fisher(r, lvl, window),
            "garch": var_garch(position, var_forecast, lvl),
        }
        out[f"{lvl:g}"] = {
            "backtests": {
                k: asdict(var_backtest(r, v, lvl, lag=0 if k == "garch" else 2))
                for k, v in methods.items()
            },
            "mean_var": {k: float(v.mean()) for k, v in methods.items()},
            "mean_cvar_historical": float(cvar_historical(r, lvl, window).mean()),
        }
    return out


def stress_windows(
    returns: pd.DataFrame, windows: dict[str, tuple[object, object]]
) -> dict[str, dict[str, object]]:
    """Total return and max drawdown of each column inside each named window.

    Windows with no out-of-sample coverage are reported as uncovered, not skipped.
    """
    out: dict[str, dict[str, object]] = {}
    for name, (a, b) in windows.items():
        sub = returns.loc[pd.Timestamp(a) : pd.Timestamp(b)]
        if sub.dropna(how="all").empty:
            out[name] = {"covered": False, "start": str(a), "end": str(b)}
            continue
        cols = {}
        for c in sub.columns:
            s = sub[c].dropna()
            if s.empty:
                continue
            eq = (1 + s).cumprod()
            cols[c] = {
                "total_return": float(eq.iloc[-1] - 1),
                "max_drawdown": float((eq / eq.cummax() - 1).min()),
            }
        out[name] = {
            "covered": True,
            "start": str(a),
            "end": str(b),
            "n": len(sub),
            "results": cols,
        }
    return out


def regime_conditional(
    r: pd.Series, p_high: pd.Series, threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Performance split by the filtered HMM regime at the decision date."""
    state = (p_high.reindex(r.index) > threshold).map({True: "high_vol", False: "low_vol"})
    return {
        k: performance(r[state == k]) for k in ("low_vol", "high_vol") if (state == k).sum() > 20
    }


def bootstrap_equity_paths(r: pd.Series, n_paths: int, seed: int, block: float) -> dict[str, float]:
    """Stationary-bootstrap Monte Carlo of the equity path: terminal wealth and max drawdown quantiles."""
    from arch.bootstrap import StationaryBootstrap

    x = r.dropna().to_numpy()
    bs = StationaryBootstrap(max(block, 1.0), x, seed=seed)
    terminal, mdd = [], []
    for data in bs.bootstrap(n_paths):
        path = data[0][0]
        eq = np.cumprod(1 + path)
        terminal.append(eq[-1])
        mdd.append((eq / np.maximum.accumulate(eq) - 1).min())
    t, d = np.asarray(terminal), np.asarray(mdd)
    return {
        "terminal_p05": float(np.quantile(t, 0.05)),
        "terminal_p50": float(np.quantile(t, 0.5)),
        "terminal_p95": float(np.quantile(t, 0.95)),
        "prob_loss": float(np.mean(t < 1)),
        "mdd_p05": float(np.quantile(d, 0.05)),
        "mdd_p50": float(np.quantile(d, 0.5)),
    }
