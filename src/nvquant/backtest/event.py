"""Event-driven backtest engine: an independent loop over fills, used to check the vectorized one.

It walks day by day, generating an order at each open from the difference
between the target and current weight, charging costs on the order's dollar
notional, and marking the book to the next fill. Written separately from
:mod:`nvquant.backtest.vectorized` on purpose; the two must agree to 1e-10.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nvquant.backtest.costs import BPS, CostInputs, borrow_rate
from nvquant.backtest.vectorized import BacktestResult
from nvquant.config.schema import CostsConfig


def run_event(
    position: pd.Series,
    holding_log_return: pd.Series,
    inputs: CostInputs,
    cfg: CostsConfig,
    aum: float | None = None,
    flat_bps: float | None = None,
) -> BacktestResult:
    """Simulate fills one day at a time. Returns per-day results in capital units."""
    capital = cfg.aum if aum is None else aum
    fixed = (cfg.half_spread_bps + cfg.commission_bps) * BPS
    sig_fallback = float(inputs.sigma.median())
    dates, nets, grosses, costs_l, pos_l, trades = [], [], [], [], [], []
    held = 0.0
    last_sigma = float("nan")
    last_adv = float("nan")
    for date, target in position.items():
        lr = holding_log_return.get(date, np.nan)
        if lr is None or not math.isfinite(lr):
            continue
        target = 0.0 if not math.isfinite(target) else float(target)
        order = target - held  # weight to buy (+) or sell (-) at the open fill
        s = inputs.sigma.get(date, np.nan)
        a = inputs.adv.get(date, np.nan)
        last_sigma = s if math.isfinite(s) else last_sigma
        last_adv = a if math.isfinite(a) else last_adv
        if flat_bps is not None:
            rate = flat_bps * BPS
        elif not math.isfinite(last_sigma):
            rate = fixed + cfg.slippage_vol_mult * sig_fallback
        else:
            notional = abs(order) * capital
            part = notional / last_adv if math.isfinite(last_adv) and last_adv > 0 else 0.0
            rate = (
                fixed
                + cfg.slippage_vol_mult * last_sigma
                + cfg.impact_coef * last_sigma * math.sqrt(part)
            )
        fill_cost = rate * abs(order)
        held = target
        pnl = held * math.expm1(lr)
        carry = borrow_rate(cfg) * max(-held, 0.0)
        dates.append(date)
        grosses.append(pnl)
        costs_l.append(fill_cost + carry)
        nets.append(pnl - fill_cost - carry)
        pos_l.append(held)
        trades.append(abs(order))
    idx = pd.DatetimeIndex(dates)
    return BacktestResult(
        pd.Series(nets, index=idx),
        pd.Series(grosses, index=idx),
        pd.Series(costs_l, index=idx),
        pd.Series(pos_l, index=idx),
        pd.Series(trades, index=idx),
    )
