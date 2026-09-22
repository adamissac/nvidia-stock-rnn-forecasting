"""Forecast evaluation: accuracy, information coefficient, direction, calibration.

References: Campbell and Thompson (2008) for R2_oos; Diebold and Mariano (1995)
with the Harvey, Leybourne, and Newbold (1997) small-sample correction;
Pesaran and Timmermann (1992); Gibbs and Candes (2021) for adaptive conformal
inference; Hansen, Lunde, and Nason (2011) for the model confidence set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) automatic lag: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the mean of ``x`` with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    lags = newey_west_lags(n) if lags is None else lags
    e = x - x.mean()
    lrv = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(lrv, 0.0) / n))


def safe_spearman(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    """Spearman correlation, NaN if either input is constant or too short."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def r2_oos(y: np.ndarray, pred: np.ndarray, bench: np.ndarray | float = 0.0) -> float:
    """Out-of-sample R^2 against a benchmark forecast (zero by default):
    ``1 - sum (y - pred)^2 / sum (y - bench)^2``.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    b = np.broadcast_to(np.asarray(bench, float), y.shape)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - b) ** 2))


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test of equal predictive accuracy."""

    stat: float
    pvalue: float
    mean_loss_diff: float


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> DMResult:
    """DM test with the HLN correction; loss ``|e|^power``. Positive stat means model 1 is worse.

    The HLN version scales the DM statistic by
    ``sqrt((n + 1 - 2h + h(h-1)/n) / n)`` and uses a t distribution with n-1 df.
    """
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    d = d[~np.isnan(d)]
    n = len(d)
    se = newey_west_se(d, lags=max(h - 1, 0)) if h > 1 else float(np.std(d, ddof=0) / np.sqrt(n))
    if not np.isfinite(se) or se == 0:
        return DMResult(float("nan"), float("nan"), float(d.mean()) if n else float("nan"))
    dm = d.mean() / se
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return DMResult(float(hln), float(p), float(d.mean()))


@dataclass(frozen=True)
class ICResult:
    """Spearman information coefficient with a Newey-West t-stat."""

    ic: float
    t_stat: float
    n: int


def information_coefficient(pred: np.ndarray, y: np.ndarray, lags: int | None = None) -> ICResult:
    """Time-series Spearman IC and a HAC t-stat.

    The t-stat is computed on the products of standardized ranks, whose mean is
    the Spearman correlation, so autocorrelation (for example from overlapping
    multi-day returns) is handled by the Newey-West variance.
    """
    p, t = np.asarray(pred, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(t))
    p, t = p[ok], t[ok]
    if len(p) < 10 or np.ptp(p) == 0:
        return ICResult(float("nan"), float("nan"), len(p))
    rp = stats.rankdata(p)
    rt = stats.rankdata(t)
    zp = (rp - rp.mean()) / rp.std()
    zt = (rt - rt.mean()) / rt.std()
    prod = zp * zt
    se = newey_west_se(prod, lags)
    return ICResult(float(prod.mean()), float(prod.mean() / se) if se > 0 else float("nan"), len(p))


@dataclass(frozen=True)
class PTResult:
    """Pesaran-Timmermann test of directional accuracy."""

    hit_rate: float
    stat: float
    pvalue: float


def pesaran_timmermann(pred: np.ndarray, y: np.ndarray) -> PTResult:
    """Tests whether sign(pred) predicts sign(y) better than independence would."""
    p, t = np.asarray(pred, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(t))
    up_p, up_y = p[ok] > 0, t[ok] > 0
    n = len(up_p)
    if n < 10:
        return PTResult(float("nan"), float("nan"), float("nan"))
    hit = float(np.mean(up_p == up_y))
    py, px = up_y.mean(), up_p.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hit = p_star * (1 - p_star) / n
    v_star = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n**2
    )
    denom = v_hit - v_star
    if denom <= 0:
        return PTResult(hit, float("nan"), float("nan"))
    stat = (hit - p_star) / np.sqrt(denom)
    return PTResult(hit, float(stat), float(stats.norm.sf(stat)))


def pit_gaussian(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Probability integral transform under a Gaussian forecast (uniform if calibrated)."""
    return stats.norm.cdf((np.asarray(y) - np.asarray(mean)) / np.asarray(std))


def pit_quantiles(y: np.ndarray, qs: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """Approximate PIT from a few quantiles by linear interpolation of the CDF (clipped at the ends)."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    lv = np.asarray(levels)
    for i in range(len(y)):
        out[i] = np.interp(y[i], qs[i], lv, left=lv[0] / 2, right=(1 + lv[-1]) / 2)
    return out


def pit_uniformity(pit: np.ndarray, bins: int = 10) -> dict[str, float | list[float]]:
    """Histogram of PIT values and a KS test against the uniform distribution."""
    pit = pit[~np.isnan(pit)]
    hist, _ = np.histogram(pit, bins=bins, range=(0, 1))
    ks = stats.kstest(pit, "uniform")
    return {
        "hist": (hist / max(len(pit), 1)).tolist(),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of outcomes inside ``[lo, hi]``."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


@dataclass(frozen=True)
class ConformalResult:
    """Adaptive conformal intervals and their realized coverage."""

    lower: np.ndarray
    upper: np.ndarray
    coverage: float
    mean_width: float


def adaptive_conformal(
    y: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.005,
    window: int = 250,
) -> ConformalResult:
    """Adaptive conformal inference (Gibbs and Candes 2021) around point forecasts.

    At each t the interval is ``pred_t +/- Q_{1 - alpha_t}`` of the last
    ``window`` absolute residuals that are already known. Because the label at t
    resolves two sessions later, only residuals up to t-2 are used, and the
    update ``alpha <- alpha + gamma (alpha - err_s)`` for the interval at s is
    applied at s+2, when its outcome is known.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(y)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    a_t = alpha
    errs = []
    resid = np.abs(y - pred)
    pending: dict[int, float] = {}
    for t in range(n):
        # the outcome of interval t-2 resolves at the open of t, before the decision at t
        if t - 2 in pending:
            a_t = a_t + gamma * (alpha - pending.pop(t - 2))
        past = resid[max(0, t - 1 - window) : max(0, t - 1)]
        past = past[~np.isnan(past)]
        if len(past) >= 30:
            q = np.quantile(past, min(max(1 - a_t, 0.0), 1.0))
            lo[t], hi[t] = pred[t] - q, pred[t] + q
            if not np.isnan(y[t]):
                err = float(not (lo[t] <= y[t] <= hi[t]))
                errs.append(err)
                pending[t] = err
    width = hi - lo
    return ConformalResult(
        lo, hi, float(1 - np.mean(errs)) if errs else float("nan"), float(np.nanmean(width))
    )


def model_confidence_set(
    losses: pd.DataFrame, size: float = 0.1, reps: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Hansen-Lunde-Nason MCS p-values (``arch.bootstrap.MCS``). Models with p > size are in the set."""
    from arch.bootstrap import MCS

    mcs = MCS(losses.dropna(), size=size, reps=reps, seed=seed)
    mcs.compute()
    return {str(k): float(v) for k, v in mcs.pvalues["Pvalue"].items()}


def summarize_forecast(
    pred: pd.Series, y: pd.Series, horizon_returns: dict[int, pd.Series] | None = None
) -> dict[str, object]:
    """Standard forecast summary used in reports."""
    df = pd.concat({"pred": pred, "y": y}, axis=1).dropna()
    ic = information_coefficient(df["pred"].to_numpy(), df["y"].to_numpy())
    pt = pesaran_timmermann(df["pred"].to_numpy(), df["y"].to_numpy())
    dm = diebold_mariano(df["y"] - df["pred"], df["y"].to_numpy())
    out: dict[str, object] = {
        "n": len(df),
        "r2_oos": r2_oos(df["y"], df["pred"]),
        "ic": ic.ic,
        "ic_t": ic.t_stat,
        "hit_rate": pt.hit_rate,
        "pt_stat": pt.stat,
        "pt_pvalue": pt.pvalue,
        "dm_vs_zero": asdict(dm),
    }
    if horizon_returns:
        out["ic_decay"] = {
            str(h): information_coefficient(*pd.concat([pred, r], axis=1).dropna().to_numpy().T).ic
            for h, r in horizon_returns.items()
        }
    return out
