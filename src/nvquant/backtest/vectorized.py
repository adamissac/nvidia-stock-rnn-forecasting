"""Vectorized backtest engine.

A position chosen at decision date t earns the holding-period simple return
``R_t = exp(r_t) - 1``, where ``r_t`` is ``log(open[t+2] / open[t+1])`` for
next-open execution. Accounting per unit of capital, rebalanced to the target
weight at each fill:

    net_t = p_t * R_t - rate_t * |p_t - p_{t-1}| - borrow * max(-p_t, 0)

with ``p_{-1} = 0``. Turnover ignores intraday weight drift (the same
convention in both engines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nvquant.backtest.costs import CostInputs, borrow_rate, cost_rate
from nvquant.config.schema import CostsConfig


@dataclass(frozen=True)
class BacktestResult:
    """Daily results of one strategy."""

    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    position: pd.Series
    turnover: pd.Series


def run_vectorized(
    position: pd.Series,
    holding_log_return: pd.Series,
    inputs: CostInputs,
    cfg: CostsConfig,
    aum: float | None = None,
    flat_bps: float | None = None,
) -> BacktestResult:
    """Backtest a position series (indexed by decision date).

    Dates where the holding return is unknown (the last sessions) are dropped.
    """
    r = holding_log_return.reindex(position.index)
    keep = r.notna()
    p = position[keep].fillna(0.0)
    R = np.expm1(r[keep])
    trade = p.diff().abs()
    trade.iloc[0] = abs(p.iloc[0])
    rate = cost_rate(trade, inputs, cfg, aum, flat_bps)
    costs = rate * trade + borrow_rate(cfg) * (-p).clip(lower=0)
    gross = p * R
    return BacktestResult(gross - costs, gross, costs, p, trade)
