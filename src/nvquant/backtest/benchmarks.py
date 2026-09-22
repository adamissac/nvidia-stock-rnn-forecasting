"""Benchmark positions and the random-signal null distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.config.schema import SizingConfig, StrategyConfig
from nvquant.portfolio.meta_labeling import trend_side
from nvquant.portfolio.sizing import vol_target_position


def buy_and_hold(dates: pd.DatetimeIndex) -> pd.Series:
    """Always fully long."""
    return pd.Series(1.0, index=dates, name="bh")


def vol_targeted_buy_and_hold(
    dates: pd.DatetimeIndex, var_forecast: pd.Series, st: StrategyConfig
) -> pd.Series:
    """Always long, scaled to the target vol with the same variance forecast the strategies use.

    This is the key ablation: a model only adds value if it beats this.
    """
    sc = SizingConfig(name="bh_voltarget", rule="voltarget")
    return vol_target_position(buy_and_hold(dates), var_forecast, sc, st).rename("bh_voltarget")


def trend_rule(close: pd.Series, dates: pd.DatetimeIndex, window: int) -> pd.Series:
    """Long when the close is above its ``window``-day average, else flat."""
    return trend_side(close, window).reindex(dates).fillna(0.0).rename(f"trend_{window}")


def random_signals(
    dates: pd.DatetimeIndex, switch_prob: float, long_share: float, n_paths: int, seed: int
) -> np.ndarray:
    """Random long/flat paths from a two-state Markov chain with matched turnover and exposure.

    With ``q`` the switch probability out of flat and ``p`` out of long, the
    stationary long share is ``q / (p + q)`` and the expected switch rate is
    ``2pq / (p + q)``. Solving for the strategy's long share and switch rate
    gives ``p = s / (2 L)`` and ``q = s / (2 (1 - L))``.
    """
    rng = np.random.default_rng(seed)
    L = min(max(long_share, 0.02), 0.98)
    s = min(max(switch_prob, 1e-4), 2 * min(L, 1 - L))
    p_exit = min(s / (2 * L), 1.0)
    p_enter = min(s / (2 * (1 - L)), 1.0)
    n = len(dates)
    out = np.zeros((n_paths, n))
    state = (rng.random(n_paths) < L).astype(float)
    u = rng.random((n_paths, n))
    for t in range(n):
        out[:, t] = state
        flip = np.where(state == 1.0, u[:, t] < p_exit, u[:, t] < p_enter)
        state = np.where(flip, 1.0 - state, state)
    return out
