"""Walk-forward volatility forecasts and their evaluation.

The forecast made after the close of t is for the variance of session t+1's
close-to-close log return. The realized proxy is the overnight gap squared
plus the Garman-Klass range variance of session t+1 (a lower-noise proxy than
the squared return, Patton 2011). Sizing and VaR use these forecasts.

Models: EWMA (RiskMetrics, lambda 0.94), GARCH(1,1), GJR-GARCH(1,1,1) with
Student-t errors, EGARCH(1,1,1) (all ``arch``), HAR-RV on the range proxy
(Corsi 2009), and LightGBM on the feature store.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nvquant.config.schema import VolConfig
from nvquant.data.market import MarketData
from nvquant.evaluation.forecast import newey_west_lags
from nvquant.logging_utils import get_logger
from nvquant.models.trees import LGBMForecaster

log = get_logger(__name__)

GARCH_SPECS: dict[str, dict[str, object]] = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    "gjr_t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
}


def realized_variance_proxy(market: MarketData) -> pd.Series:
    """Daily variance proxy: overnight gap squared plus Garman-Klass variance of the session."""
    px = market.ohlcv()
    o, h, lo, c = (np.log(px[k]) for k in ("open", "high", "low", "close"))
    gk = 0.5 * (h - lo) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return ((o - c.shift(1)) ** 2 + gk.clip(lower=0)).rename("rv")


def _returns_pct(market: MarketData) -> pd.Series:
    return (100 * np.log(market.ohlcv()["close"]).diff()).dropna()


def ewma_forecast(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics variance forecast for t+1 made at t (decimal units)."""
    return (r**2).ewm(alpha=1 - lam, adjust=False).mean()


def garch_forecasts(
    market: MarketData, spec: dict[str, object], refit_dates: pd.DatetimeIndex, start: pd.Timestamp
) -> pd.Series:
    """Variance forecasts for t+1 at every t from the first refit on.

    Parameters are estimated on data from ``start`` to each refit date; between
    refits they are held fixed and the variance recursion is filtered through
    new data, so each forecast only uses returns up to t.
    """
    from arch import arch_model

    r = _returns_pct(market).loc[start:]
    out = []
    failed: list[str] = []
    for i, d in enumerate(refit_dates):
        nxt = refit_dates[i + 1] if i + 1 < len(refit_dates) else r.index[-1]
        am = arch_model(r.loc[:nxt], mean="Constant", **spec)  # type: ignore[arg-type]
        last = r.index[r.index.get_loc(d) + 1] if d < r.index[-1] else None
        # arch resets its ConvergenceWarning filter inside fit(), so record the warnings and
        # check the optimizer's own flag instead of letting them print or disappear
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = am.fit(last_obs=last, disp="off", options={"maxiter": 1000})
        if res.convergence_flag != 0:
            failed.append(str(d.date()))
        fc = res.forecast(horizon=1, start=d, reindex=False).variance["h.1"]
        lo = d if i == 0 else d + pd.Timedelta(days=1)
        out.append(fc.loc[lo:nxt])
    if failed:
        log.warning("%s: optimizer did not converge at %d of %d refits (%s)", spec["vol"],
                    len(failed), len(refit_dates), ", ".join(failed[:5]))  # fmt: skip
    series = (pd.concat(out) / 1e4).rename("var")
    series.attrs["convergence_failures"] = failed
    return series


def har_forecasts(rv: pd.Series, refit_dates: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Series:
    """HAR-RV: ``RV_{t+1} = b0 + b1 RV_t + b5 mean(RV_{t-4..t}) + b22 mean(RV_{t-21..t})``, OLS on logs.

    Fit in logs for stability, then mapped back with the lognormal bias correction.
    """
    lrv = np.log(rv.clip(lower=1e-8))
    X = pd.DataFrame(
        {"d": lrv, "w": lrv.rolling(5).mean(), "m": lrv.rolling(22).mean()}, index=rv.index
    )
    y = lrv.shift(-1)  # leakage-ok: HAR regression target, used only on rows before the refit date
    out = []
    for i, d in enumerate(refit_dates):
        nxt = refit_dates[i + 1] if i + 1 < len(refit_dates) else rv.index[-1]
        train = X.loc[start:d].iloc[:-1].dropna().index
        train = train[y.loc[train].notna()]
        fit = sm.OLS(y.loc[train], sm.add_constant(X.loc[train])).fit()
        s2 = float(fit.mse_resid)
        lo = d if i == 0 else d + pd.Timedelta(days=1)
        block = X.loc[lo:nxt].dropna()
        pred = fit.predict(sm.add_constant(block, has_constant="add"))
        out.append(np.exp(pred + s2 / 2))
    return pd.concat(out).rename("var")


def lgbm_vol_forecasts(
    features: pd.DataFrame,
    rv: pd.Series,
    refit_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    seed: int,
) -> pd.Series:
    """LightGBM on the feature store predicting log RV of the next session."""
    lrv_next = np.log(rv.clip(lower=1e-8)).shift(
        -1
    )  # leakage-ok: target; training rows end before each refit
    X = features.loc[start:]
    out = []
    for i, d in enumerate(refit_dates):
        nxt = refit_dates[i + 1] if i + 1 < len(refit_dates) else X.index[-1]
        train = X.loc[:d].index[:-1]
        train = train[lrv_next.reindex(train).notna().to_numpy()]
        m = LGBMForecaster(
            seed=seed, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50
        )
        yt = lrv_next.loc[train]
        m.fit(X, yt)
        resid_var = float(np.var(yt - m.predict(X, train)))
        lo = d if i == 0 else d + pd.Timedelta(days=1)
        block = X.loc[lo:nxt].index
        out.append(np.exp(m.predict(X.loc[:nxt], block) + resid_var / 2))
    return pd.concat(out).rename("var")


def vol_refit_dates(
    sessions: pd.DatetimeIndex, start: pd.Timestamp, cfg: VolConfig
) -> pd.DatetimeIndex:
    """Refit dates for the vol models: first after ``min_train`` sessions, then every ``refit_every``."""
    usable = sessions[sessions >= start]
    return pd.DatetimeIndex(usable[cfg.min_train - 1 :: cfg.refit_every])


def all_vol_forecasts(
    market: MarketData, features: pd.DataFrame, start: pd.Timestamp, cfg: VolConfig, seed: int
) -> pd.DataFrame:
    """Every configured vol model's one-step variance forecast, on one index."""
    rv = realized_variance_proxy(market)
    dates = vol_refit_dates(market.sessions, start, cfg)
    cols: dict[str, pd.Series] = {}
    r = np.log(market.ohlcv()["close"]).diff()
    for name in cfg.models:
        if name == "ewma":
            cols[name] = ewma_forecast(r).loc[dates[0] :]
        elif name in GARCH_SPECS:
            cols[name] = garch_forecasts(market, GARCH_SPECS[name], dates, start)
        elif name == "har":
            cols[name] = har_forecasts(rv, dates, start)
        elif name == "lgbm_vol":
            cols[name] = lgbm_vol_forecasts(features, rv, dates, start, seed)
        else:
            raise KeyError(f"unknown vol model {name}")
    out = pd.DataFrame(cols)
    out.attrs["convergence_failures"] = {
        k: v.attrs.get("convergence_failures", []) for k, v in cols.items()
    }
    out["rv_next"] = rv.shift(-1).reindex(out.index)  # leakage-ok: realized target for scoring only
    return out


@dataclass(frozen=True)
class VolScore:
    """Accuracy of one variance forecast against the realized proxy."""

    qlike: float
    mse: float
    mz_alpha: float
    mz_beta: float
    mz_r2: float
    mz_wald_pvalue: float
    n: int


def qlike(rv: np.ndarray, h: np.ndarray) -> float:
    """QLIKE loss ``mean(rv/h - log(rv/h) - 1)``, which ranks forecasts consistently under proxy noise (Patton 2011)."""
    ratio = np.asarray(rv) / np.asarray(h)
    return float(np.mean(ratio - np.log(ratio) - 1))


def score_vol(rv: pd.Series, h: pd.Series) -> VolScore:
    """QLIKE, MSE, and a Mincer-Zarnowitz regression ``rv = a + b h`` with a HAC Wald test of (0, 1)."""
    df = pd.concat({"rv": rv, "h": h}, axis=1).dropna()
    df = df[(df["rv"] > 0) & (df["h"] > 0)]
    fit = sm.OLS(df["rv"], sm.add_constant(df["h"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))}
    )
    wald = fit.wald_test("const = 0, h = 1", scalar=True)
    return VolScore(
        qlike=qlike(df["rv"].to_numpy(), df["h"].to_numpy()),
        mse=float(np.mean((df["rv"] - df["h"]) ** 2)),
        mz_alpha=float(fit.params["const"]),
        mz_beta=float(fit.params["h"]),
        mz_r2=float(fit.rsquared),
        mz_wald_pvalue=float(wald.pvalue),
        n=len(df),
    )
