"""The two engines agree, and the backtest invariants hold."""

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nvquant.backtest.benchmarks import (
    buy_and_hold,
    random_signals,
    trend_rule,
    vol_targeted_buy_and_hold,
)
from nvquant.backtest.costs import BPS, cost_inputs, cost_rate
from nvquant.backtest.event import run_event
from nvquant.backtest.strategies import (
    benchmark_positions,
    inputs_for,
    meta_voltarget,
    model_positions,
    run_both,
)
from nvquant.backtest.vectorized import run_vectorized
from nvquant.config.schema import CostsConfig, SizingConfig, StrategyConfig
from nvquant.labels.forward import ex_ante_vol, holding_log_return
from nvquant.portfolio.meta_labeling import bet_size, trend_side
from nvquant.portfolio.sizing import apply_regime_filter, size

COSTS = CostsConfig()
WARMUP = 30  # sessions before the ex-ante vol and trailing ADV exist


@pytest.fixture(scope="module")
def setup(market):
    sigma = ex_ante_vol(market.ohlcv()["close"], 20)
    return holding_log_return(market), cost_inputs(market, sigma, COSTS)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), short=st.booleans())
def test_engines_agree(market, seed, short):
    r = holding_log_return(market)
    inputs = cost_inputs(market, ex_ante_vol(market.ohlcv()["close"], 20), COSTS)
    rng = np.random.default_rng(seed)
    lo = -1.5 if short else 0.0
    live = market.sessions[WARMUP:]
    pos = pd.Series(rng.uniform(lo, 1.5, len(live)), index=live)
    pos[rng.random(len(pos)) < 0.3] = 0.0
    v = run_vectorized(pos, r, inputs, COSTS)
    e = run_event(pos, r, inputs, COSTS)
    assert v.net.index.equals(e.net.index)
    assert float((v.net - e.net).abs().max()) < 1e-10
    for bps in (0.0, 7.5):
        assert (
            float(
                (
                    run_vectorized(pos, r, inputs, COSTS, flat_bps=bps).net
                    - run_event(pos, r, inputs, COSTS, flat_bps=bps).net
                )
                .abs()
                .max()
            )
            < 1e-12
        )


def test_zero_signal_earns_exactly_zero(market, setup):
    r, inputs = setup
    res = run_vectorized(pd.Series(0.0, index=market.sessions), r, inputs, COSTS)
    assert (res.net == 0).all() and (res.costs == 0).all()


def test_always_long_is_buy_and_hold_minus_entry_cost(market, setup):
    r, inputs = setup
    res = run_vectorized(pd.Series(1.0, index=market.sessions), r, inputs, COSTS, flat_bps=5.0)
    bh = np.expm1(r.dropna())
    diff = bh - res.net
    assert diff.iloc[0] == pytest.approx(5.0 * BPS)
    assert np.allclose(diff.iloc[1:], 0.0)


def test_higher_costs_never_raise_pnl(market, setup):
    r, inputs = setup
    live = market.sessions[WARMUP:]
    pos = pd.Series(np.sign(np.sin(np.arange(len(live)) / 3)), index=live)
    totals = [
        run_vectorized(pos, r, inputs, COSTS, flat_bps=b).net.sum() for b in (0, 1, 5, 10, 20)
    ]
    assert all(a >= b for a, b in itertools.pairwise(totals))
    cheap = CostsConfig(aum=1e6)
    rich = CostsConfig(aum=1e11)
    assert (
        run_vectorized(pos, r, inputs, cheap).net.sum()
        > run_vectorized(pos, r, inputs, rich).net.sum()
    )


def test_positions_are_lagged(market, setup):
    """A position decided at t earns the open(t+1)->open(t+2) return, never the return that ends at t."""
    r, inputs = setup
    o = market.ohlcv()["open"]
    pos = pd.Series(0.0, index=market.sessions)
    t = market.sessions[100]
    pos.loc[t] = 1.0
    res = run_vectorized(pos, r, inputs, COSTS, flat_bps=0.0)
    i = 100
    assert res.gross.loc[t] == pytest.approx(o.iloc[i + 2] / o.iloc[i + 1] - 1)
    assert res.gross.drop(t).abs().sum() == 0


def test_borrow_charged_on_shorts(market, setup):
    r, inputs = setup
    res = run_vectorized(pd.Series(-1.0, index=market.sessions), r, inputs, COSTS, flat_bps=0.0)
    assert res.costs.iloc[5] == pytest.approx(COSTS.borrow_bps_annual * BPS / 252)


def test_cost_rate_matches_formula_and_refuses_missing_inputs(market, setup):
    _, inputs = setup
    live = market.sessions[WARMUP:]
    trade = pd.Series(0.5, index=live)
    only_impact = CostsConfig(half_spread_bps=0.0, commission_bps=0.0, slippage_vol_mult=0.0, impact_coef=0.1, aum=1e9)
    got = cost_rate(trade, inputs, only_impact)
    sig, adv = inputs.sigma.reindex(live), inputs.adv.reindex(live)
    np.testing.assert_allclose(got, 0.1 * sig * np.sqrt(0.5 * 1e9 / adv), rtol=1e-12)
    with pytest.raises(ValueError, match="missing"):
        cost_rate(pd.Series(0.5, index=market.sessions[:5]), inputs, COSTS)
    with pytest.raises(ValueError, match="missing"):
        run_event(pd.Series(1.0, index=market.sessions[:5]), holding_log_return(market), inputs, COSTS)
    # no trade, no cost, even during the warmup
    assert (cost_rate(pd.Series(0.0, index=market.sessions[:5]), inputs, COSTS) >= 0).all()


def test_sizing_rules():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    fc = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    var = pd.Series(0.02**2, index=idx)
    ph = pd.Series(np.r_[np.zeros(200), np.ones(200)], index=idx)
    st_cfg = StrategyConfig()
    for rule in ("sign", "scaled", "voltarget", "kelly"):
        lf = size(fc, var, ph, SizingConfig(name=rule, rule=rule), st_cfg)
        ls = size(fc, var, ph, SizingConfig(name=rule, rule=rule, long_short=True), st_cfg)
        assert lf.min() >= 0 and lf.max() <= st_cfg.max_gross
        assert ls.min() < 0 and ls.abs().max() <= st_cfg.max_gross + 1e-12
    reg = size(fc, var, ph, SizingConfig(name="r", rule="sign", regime_filter=True), st_cfg)
    assert (reg.iloc[200:] == 0).all()
    assert (apply_regime_filter(pd.Series(1.0, index=idx), ph).iloc[:200] == 1).all()
    vt = size(pd.Series(1.0, index=idx), var, ph, SizingConfig(name="v", rule="voltarget"), st_cfg)
    assert vt.iloc[0] == pytest.approx(min(0.35 / (0.02 * np.sqrt(252)), 1.5))


def test_meta_and_benchmarks(market):
    close = market.ohlcv()["close"]
    side = trend_side(close, 50)
    assert set(side.unique()) <= {0.0, 1.0}
    assert bet_size(np.array([0.5]))[0] == pytest.approx(0.0) and bet_size(np.array([0.9]))[0] > 0.5
    idx = market.sessions[100:]
    var = pd.Series(0.0004, index=idx)
    assert (buy_and_hold(idx) == 1).all()
    assert vol_targeted_buy_and_hold(idx, var, StrategyConfig()).iloc[0] > 0
    assert set(trend_rule(close, idx, 50).unique()) <= {0.0, 1.0}
    paths = random_signals(idx, 0.1, 0.6, 200, seed=0)
    assert abs(paths.mean() - 0.6) < 0.05
    switches = np.abs(np.diff(paths, axis=1)).mean()
    assert abs(switches - 0.1) < 0.02


def test_strategy_grid_helpers(market, fast_cfg):
    idx = market.sessions[100:]
    var = pd.Series(0.0004, index=idx)
    fc = {
        "zero": pd.DataFrame({"pred": 0.0}, index=idx),
        "m": pd.DataFrame({"pred": np.sin(np.arange(len(idx)))}, index=idx),
    }
    grid = model_positions(fc, var, pd.Series(0.0, index=idx), fast_cfg)
    assert len(grid) == len(fast_cfg.strategy.sizings) and all(k.startswith("m__") for k in grid)
    bench = benchmark_positions(market, idx, var, fast_cfg)
    assert {"bh_target", "bh_voltarget", "bh_smh", "bh_qqq"} <= set(bench)
    sigma = ex_ante_vol(market.ohlcv()["close"], 20)
    inputs = inputs_for(market, sigma, fast_cfg, "SMH")
    res, diff = run_both(
        bench["bh_smh"][1], holding_log_return(market, ticker="SMH"), inputs, fast_cfg
    )
    assert diff < 1e-10 and len(res.net) > 0
    assert meta_voltarget(pd.Series(0.5, index=idx), var, fast_cfg).iloc[0] > 0
