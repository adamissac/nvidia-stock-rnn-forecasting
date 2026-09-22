"""Transaction cost model, per unit of traded notional.

``rate_t = (half_spread + commission) / 1e4 + k_slip * sigma_t
+ k_impact * sigma_t * sqrt(trade_notional_t / ADV_t)``

where ``sigma_t`` is the ex-ante daily vol, ``ADV_t`` the trailing dollar
volume (lagged one session), and ``trade_notional_t = |delta position| * AUM``.
The square-root term is the usual market-impact law (Almgren et al. 2005;
Toth et al. 2011). Borrow on shorts is charged per day held.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nvquant.config.schema import CostsConfig
from nvquant.data.market import MarketData

BPS = 1e-4


@dataclass(frozen=True)
class CostInputs:
    """Market inputs the cost model needs, aligned to decision dates."""

    sigma: pd.Series
    adv: pd.Series


def cost_inputs(
    market: MarketData, sigma: pd.Series, cfg: CostsConfig, ticker: str | None = None
) -> CostInputs:
    """Ex-ante daily vol and trailing dollar ADV (lagged one session) for the traded ticker."""
    px = market.ohlcv(ticker)
    dollar = px["close"] * px["volume"]
    adv = dollar.rolling(cfg.adv_window, min_periods=cfg.adv_window).mean().shift(1)
    return CostInputs(sigma.reindex(px.index), adv)


def cost_rate(
    trade: pd.Series,
    inputs: CostInputs,
    cfg: CostsConfig,
    aum: float | None = None,
    flat_bps: float | None = None,
) -> pd.Series:
    """Cost per unit of traded notional for each decision date.

    Parameters
    ----------
    trade : Series
        ``|position_t - position_{t-1}|`` in units of capital.
    inputs : CostInputs
    cfg : CostsConfig
    aum : float, optional
        Capital for the impact term (``cfg.aum`` by default).
    flat_bps : float, optional
        If given, replace the whole model with a flat per-side cost (the cost sweep).
    """
    if flat_bps is not None:
        return pd.Series(flat_bps * BPS, index=trade.index)
    aum = cfg.aum if aum is None else aum
    sigma = inputs.sigma.reindex(trade.index).ffill()
    adv = inputs.adv.reindex(trade.index).ffill()
    participation = (trade.abs() * aum / adv).clip(lower=0).fillna(0.0)
    fixed = (cfg.half_spread_bps + cfg.commission_bps) * BPS
    rate = fixed + cfg.slippage_vol_mult * sigma + cfg.impact_coef * sigma * np.sqrt(participation)
    return rate.fillna(fixed + cfg.slippage_vol_mult * float(inputs.sigma.median()))


def borrow_rate(cfg: CostsConfig) -> float:
    """Daily borrow cost per unit of short notional."""
    return cfg.borrow_bps_annual * BPS / 252.0
