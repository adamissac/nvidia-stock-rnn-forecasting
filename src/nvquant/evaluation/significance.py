"""Trial-adjusted significance of backtest Sharpe ratios.

- PSR and DSR: Bailey and Lopez de Prado (2012, 2014). Sharpe ratios in the
  formulas are per period (daily), not annualized.
- MinTRL: Bailey and Lopez de Prado (2012).
- PBO via CSCV: Bailey, Borwein, Lopez de Prado, and Zhu (2017).
- SPA and White's Reality Check: Hansen (2005), White (2000), via ``arch``.
- Stationary bootstrap CIs with Politis-White optimal block length.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    sr = r.mean() / r.std(ddof=1)
    return float(sr), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), len(r)


def psr(sr: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio ``P(true SR > sr_star)``.

    ``Phi((sr - sr_star) sqrt(n - 1) / sqrt(1 - skew sr + (kurt - 1)/4 sr^2))``
    with per-period Sharpe ``sr`` and non-excess kurtosis ``kurt``.
    """
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0 or n < 2:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)))


def psr_from_returns(r: np.ndarray, sr_star: float = 0.0) -> float:
    """PSR of a return series against a per-period benchmark Sharpe."""
    sr, g3, g4, n = _moments(r)
    return psr(sr, n, g3, g4, sr_star)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` unskilled trials:
    ``sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))``.
    """
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    return float(
        np.sqrt(sr_variance)
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
        )
    )


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio and its inputs."""

    dsr: float
    sr: float
    sr_star: float
    n_trials: int
    sr_variance: float
    n_obs: int


def deflated_sharpe(r: np.ndarray, n_trials: int, trial_srs: np.ndarray) -> DSRResult:
    """DSR: PSR against the expected max Sharpe of ``n_trials`` trials.

    ``trial_srs`` are the per-period Sharpe ratios of every trial (for their
    cross-sectional variance).
    """
    sr, g3, g4, n = _moments(r)
    var = float(np.var(np.asarray(trial_srs, float), ddof=1)) if len(trial_srs) > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, var)
    return DSRResult(psr(sr, n, g3, g4, sr_star), sr, sr_star, n_trials, var, n)


def min_track_record_length(r: np.ndarray, sr_star: float = 0.0, alpha: float = 0.05) -> float:
    """Observations needed for ``PSR(sr_star) >= 1 - alpha``; inf if the Sharpe isn't above sr_star."""
    sr, g3, g4, _ = _moments(r)
    if sr <= sr_star:
        return float("inf")
    z = stats.norm.ppf(1 - alpha)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (z / (sr - sr_star)) ** 2)


@dataclass(frozen=True)
class PBOResult:
    """Probability of backtest overfitting and supporting statistics."""

    pbo: float
    logits: np.ndarray
    n_combinations: int
    perf_degradation_slope: float
    prob_oos_loss: float


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> PBOResult:
    """Combinatorially symmetric cross-validation.

    Split the T x N return matrix into ``n_blocks`` row blocks. For each half of
    the blocks used in-sample, pick the trial with the best in-sample Sharpe and
    find its relative rank ``w`` out of sample. ``lambda = logit(w)``; PBO is the
    share of combinations with ``lambda <= 0`` (the IS winner is at or below the
    OOS median).
    """
    M = returns.dropna().to_numpy()
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    logits, is_best, oos_best = [], [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        with np.errstate(divide="ignore", invalid="ignore"):
            is_sr = M[is_idx].mean(0) / M[is_idx].std(0, ddof=1)
            oos_sr = M[oos_idx].mean(0) / M[oos_idx].std(0, ddof=1)
        oos_sr = np.nan_to_num(oos_sr, nan=0.0)
        best = int(np.nanargmax(is_sr))
        rank = stats.rankdata(oos_sr)[best]
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(is_sr[best])
        oos_best.append(oos_sr[best])
    lg = np.asarray(logits)
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if len(is_best) > 2 else float("nan")
    return PBOResult(
        float(np.mean(lg <= 0)), lg, len(lg), slope, float(np.mean(np.asarray(oos_best) < 0))
    )


@dataclass(frozen=True)
class SPAResult:
    """Hansen SPA and White Reality Check p-values against one benchmark."""

    benchmark: str
    spa_consistent: float
    spa_lower: float
    spa_upper: float
    reality_check: float


def spa_test(bench: pd.Series, models: pd.DataFrame, reps: int, seed: int) -> SPAResult:
    """Is any model better than the benchmark after data snooping? Losses are negative returns."""
    from arch.bootstrap import SPA

    df = pd.concat([bench.rename("__bench__"), models], axis=1).dropna()
    spa = SPA(-df["__bench__"], -df.drop(columns="__bench__"), reps=reps, seed=seed)
    spa.compute()
    pv = spa.pvalues
    rc = SPA(
        -df["__bench__"], -df.drop(columns="__bench__"), reps=reps, seed=seed, studentize=False
    )
    rc.compute()
    return SPAResult(
        str(bench.name),
        float(pv["consistent"]),
        float(pv["lower"]),
        float(pv["upper"]),
        float(rc.pvalues["upper"]),
    )


def bootstrap_sharpe_ci(
    r: pd.Series, reps: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Stationary-bootstrap CI for the annualized Sharpe with the optimal block length.

    Returns (lower, upper, block_length).
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    x = r.dropna().to_numpy()
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), 1.0
    block = float(optimal_block_length(x)["stationary"].iloc[0])
    block = block if np.isfinite(block) and block >= 1.0 else 1.0
    bs = StationaryBootstrap(block, x, seed=seed)

    def f(v: np.ndarray) -> float:
        sd = v.std(ddof=1)
        return float(v.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")

    draws = np.array([f(d[0][0]) for d in bs.bootstrap(reps)])
    a = (1 - level) / 2
    if np.isnan(draws).all():
        return float("nan"), float("nan"), block
    return float(np.nanquantile(draws, a)), float(np.nanquantile(draws, 1 - a)), block
