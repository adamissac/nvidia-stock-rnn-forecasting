"""Strategy grid: every (forecast, sizing rule) pair, plus meta-labeling and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nvquant.backtest.benchmarks import buy_and_hold, trend_rule, vol_targeted_buy_and_hold
from nvquant.backtest.costs import CostInputs, cost_inputs
from nvquant.backtest.event import run_event
from nvquant.backtest.vectorized import BacktestResult, run_vectorized
from nvquant.config.schema import Config, SizingConfig
from nvquant.data.market import MarketData
from nvquant.portfolio.sizing import size, vol_target_position

STRATEGY_SEP = "__"


@dataclass(frozen=True)
class StrategySpec:
    """One backtested configuration."""

    name: str
    kind: str  # "model", "meta", or "benchmark"
    forecast: str | None
    sizing: str | None
    ticker: str


def model_positions(
    forecasts: dict[str, pd.DataFrame],
    var_forecast: pd.Series,
    p_high: pd.Series,
    cfg: Config,
    skip: tuple[str, ...] = ("zero",),
) -> dict[str, tuple[StrategySpec, pd.Series]]:
    """Positions for every forecast x sizing rule. The zero forecast is skipped (always flat)."""
    out = {}
    for fname, frame in forecasts.items():
        if fname in skip:
            continue
        for sc in cfg.strategy.sizings:
            name = f"{fname}{STRATEGY_SEP}{sc.name}"
            pos = size(frame["pred"], var_forecast, p_high, sc, cfg.strategy)
            out[name] = (StrategySpec(name, "model", fname, sc.name, cfg.universe.target), pos)
    return out


def benchmark_positions(
    market: MarketData, dates: pd.DatetimeIndex, var_forecast: pd.Series, cfg: Config
) -> dict[str, tuple[StrategySpec, pd.Series]]:
    """Buy-and-hold target, vol-targeted buy-and-hold, SMH, QQQ, and the trend rule."""
    t = cfg.universe.target
    out = {
        "bh_target": (StrategySpec("bh_target", "benchmark", None, None, t), buy_and_hold(dates)),
        "bh_voltarget": (
            StrategySpec("bh_voltarget", "benchmark", None, "voltarget", t),
            vol_targeted_buy_and_hold(dates, var_forecast, cfg.strategy),
        ),
        f"trend_{cfg.strategy.trend_window}": (
            StrategySpec(f"trend_{cfg.strategy.trend_window}", "benchmark", None, None, t),
            trend_rule(market.ohlcv()["close"], dates, cfg.strategy.trend_window),
        ),
    }
    for etf in ("SMH", "QQQ"):
        if etf in market.prices:
            out[f"bh_{etf.lower()}"] = (
                StrategySpec(f"bh_{etf.lower()}", "benchmark", None, None, etf),
                buy_and_hold(dates),
            )
    return out


def meta_voltarget(meta_pos: pd.Series, var_forecast: pd.Series, cfg: Config) -> pd.Series:
    """Meta-label bet size times the vol-target leverage."""
    sc = SizingConfig(name="meta_voltarget", rule="voltarget")
    return vol_target_position(meta_pos, var_forecast, sc, cfg.strategy).rename("meta_voltarget")


def run_both(
    position: pd.Series,
    r_hold: pd.Series,
    inputs: CostInputs,
    cfg: Config,
) -> tuple[BacktestResult, float]:
    """Run both engines; return the vectorized result and the max absolute difference in net returns."""
    vec = run_vectorized(position, r_hold, inputs, cfg.costs)
    ev = run_event(position, r_hold, inputs, cfg.costs)
    diff = float((vec.net - ev.net.reindex(vec.net.index)).abs().max()) if len(vec.net) else 0.0
    return vec, diff


def inputs_for(market: MarketData, sigma: pd.Series, cfg: Config, ticker: str) -> CostInputs:
    """Cost inputs for the traded ticker (ETF benchmarks use their own vol and volume)."""
    if ticker == market.target:
        return cost_inputs(market, sigma, cfg.costs)
    r = np.log(market.ohlcv(ticker)["close"]).diff()
    etf_sigma = r.ewm(span=cfg.labels.vol_span, adjust=False, min_periods=cfg.labels.vol_span).std()
    return cost_inputs(market, etf_sigma, cfg.costs, ticker)
