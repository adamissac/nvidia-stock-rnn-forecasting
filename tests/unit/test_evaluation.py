import numpy as np
import pandas as pd
import pytest

from nvquant.evaluation.attribution import (
    align_factors_to_holding,
    attribution_suite,
    regress,
    to_simple,
)
from nvquant.evaluation.forecast import (
    adaptive_conformal,
    diebold_mariano,
    information_coefficient,
    interval_coverage,
    model_confidence_set,
    newey_west_se,
    pesaran_timmermann,
    pit_gaussian,
    pit_quantiles,
    pit_uniformity,
    r2_oos,
    safe_spearman,
    summarize_forecast,
)
from nvquant.evaluation.importance import correlation_clusters, mda_purged, stability
from nvquant.evaluation.metrics import (
    calendar_year_returns,
    drawdown,
    max_drawdown_duration,
    performance,
    rolling_sharpe,
    sharpe,
)
from nvquant.evaluation.risk import (
    bootstrap_equity_paths,
    regime_conditional,
    stress_windows,
    var_backtest,
    var_historical,
    var_suite,
)

RNG = np.random.default_rng(0)
IDX = pd.bdate_range("2012-01-02", periods=2000)


def test_forecast_metrics():
    y = RNG.normal(size=2000)
    good = y * 0.3 + RNG.normal(size=2000)
    assert r2_oos(y, np.zeros(2000)) == pytest.approx(0.0)
    assert r2_oos(y, y) == pytest.approx(1.0)
    dm = diebold_mariano(y - good * 0.1, y)  # a small signal beats zero
    assert dm.stat < 0
    assert np.isnan(diebold_mariano(y, y).stat)  # identical errors: no information
    assert diebold_mariano(y - good * 0.1, y, h=3).stat < 0
    ic = information_coefficient(good, y)
    assert ic.ic > 0.2 and ic.t_stat > 5
    assert np.isnan(information_coefficient(np.zeros(100), y[:100]).ic)
    pt = pesaran_timmermann(good, y)
    assert pt.hit_rate > 0.55 and pt.pvalue < 0.01
    assert np.isnan(safe_spearman(np.zeros(10), y[:10]))
    assert np.isnan(newey_west_se(np.ones(2)))
    iid = newey_west_se(y, lags=0)
    assert iid == pytest.approx(y.std() / np.sqrt(len(y)), rel=1e-3)


def test_calibration_and_conformal():
    y = RNG.normal(size=3000)
    pit = pit_gaussian(y, np.zeros(3000), np.ones(3000))
    assert pit_uniformity(pit)["ks_pvalue"] > 0.01
    assert interval_coverage(y, -1.645 * np.ones(3000), 1.645 * np.ones(3000)) == pytest.approx(
        0.9, abs=0.03
    )
    qs = np.tile([-1.2816, 0.0, 1.2816], (3000, 1))
    assert pit_uniformity(pit_quantiles(y, qs, (0.1, 0.5, 0.9)))["hist"]
    conf = adaptive_conformal(y, np.zeros(3000), alpha=0.1)
    assert conf.coverage == pytest.approx(0.9, abs=0.02)


def test_mcs_and_summary():
    losses = pd.DataFrame({"a": RNG.normal(1, 0.1, 500), "b": RNG.normal(2, 0.1, 500)})
    p = model_confidence_set(losses, 0.1, reps=100)
    assert p["a"] > 0.1 > p["b"]
    y = pd.Series(RNG.normal(size=500), index=IDX[:500])
    s = summarize_forecast(y * 0.2 + RNG.normal(size=500), y, {5: y.rolling(5).sum()})
    assert set(s) >= {"ic", "r2_oos", "dm_vs_zero", "ic_decay"}


def test_performance_metrics():
    r = pd.Series(RNG.normal(0.0005, 0.01, 2000), index=IDX)
    m = performance(r, pd.Series(1.0, index=IDX), pd.Series(0.01, index=IDX))
    assert (
        m["sharpe"] == pytest.approx(sharpe(r)) and m["max_drawdown"] < 0 and m["exposure"] == 1.0
    )
    assert sharpe(r, annualize=False) == pytest.approx(r.mean() / r.std())
    assert np.isnan(sharpe(pd.Series([0.0, 0.0])))
    assert (drawdown(r) <= 0).all() and max_drawdown_duration(r) > 0
    assert rolling_sharpe(r).dropna().shape[0] == 2000 - 251
    assert len(calendar_year_returns(r)) == len(set(IDX.year))
    assert performance(r.iloc[:1]) == {"n": 1.0}


def test_var_backtests():
    r = pd.Series(RNG.normal(0, 0.01, 2000), index=IDX)
    v = var_historical(r, 0.99, 250)
    bt = var_backtest(r, v, 0.99)
    assert bt.rate == pytest.approx(0.01, abs=0.006) and bt.kupiec_pvalue > 0.01
    # a VaR that is far too small fails Kupiec
    assert var_backtest(r, v * 0.3, 0.99).kupiec_pvalue < 1e-6
    suite = var_suite(r, pd.Series(1.0, index=IDX), pd.Series(1e-4, index=IDX), [0.95], 250)
    assert set(suite["0.95"]["backtests"]) == {
        "historical",
        "parametric",
        "cornish_fisher",
        "garch",
    }
    assert np.isnan(var_backtest(r.iloc[:10], v.iloc[:10], 0.99).rate)
    # lag=0 compares r[t] with the VaR stamped at t; lag=2 with the VaR stamped at t-2
    rr = pd.Series([-0.05, 0.0, 0.0, 0.0] * 10, index=IDX[:40])
    vv = pd.Series([0.01, 0.01, 0.10, 0.01] * 10, index=IDX[:40])
    assert var_backtest(rr, vv, 0.99, lag=0).exceptions == 10  # the loss day's own VaR is small
    assert var_backtest(rr, vv, 0.99, lag=2).exceptions == 0  # the VaR from two days earlier is big


def test_stress_regimes_monte_carlo():
    r = pd.DataFrame({"a": RNG.normal(0, 0.01, 2000)}, index=IDX)
    s = stress_windows(r, {"in": ("2013-01-01", "2013-06-01"), "out": ("2030-01-01", "2030-02-01")})
    assert s["in"]["covered"] and not s["out"]["covered"]
    reg = regime_conditional(r["a"], pd.Series(np.r_[np.zeros(1000), np.ones(1000)], index=IDX))
    assert set(reg) == {"low_vol", "high_vol"}
    mc = bootstrap_equity_paths(r["a"], 50, seed=0, block=5.0)
    assert mc["terminal_p05"] <= mc["terminal_p50"] <= mc["terminal_p95"]


def test_attribution_recovers_beta():
    mkt = pd.Series(RNG.normal(0, 0.01, 2000), index=IDX)
    strat = 1.5 * mkt + RNG.normal(0, 0.002, 2000)
    out = regress(strat, mkt.to_frame("mkt"))
    assert out["betas"]["mkt"]["beta"] == pytest.approx(1.5, abs=0.05) and abs(out["alpha_t"]) < 3
    assert regress(strat.iloc[:5], mkt.iloc[:5].to_frame("mkt")) == {"n": 5}
    ff = pd.DataFrame(
        {c: RNG.normal(0, 0.01, 2000) for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]},
        index=IDX,
    )
    suite = attribution_suite(
        strat, pd.Series(0.0, index=IDX), pd.DataFrame({"QQQ": mkt, "SMH": mkt}), ff
    )
    assert set(suite) == {"QQQ", "SMH", "FF5_MOM"}
    mapped = align_factors_to_holding(ff, IDX[:-1], IDX)
    assert mapped.iloc[0, 0] == ff.iloc[1, 0]
    assert to_simple(pd.Series([0.0])).iloc[0] == 0.0


def test_importance():
    X = pd.DataFrame(RNG.normal(size=(600, 4)), index=IDX[:600], columns=list("abcd"))
    X["e"] = X["a"] + RNG.normal(0, 0.05, 600)
    y = 0.5 * X["a"] + RNG.normal(size=600)
    t_end = pd.Series(IDX[np.arange(600) + 2], index=IDX[:600])
    clusters = correlation_clusters(X, max_clusters=4)
    assert any({"a", "e"} <= set(v) for v in clusters.values())
    mda, mdi = mda_purged(
        X,
        y,
        t_end,
        IDX,
        {c: [c] for c in X.columns},
        {"n_estimators": 50, "num_leaves": 4, "min_child_samples": 20},
        3,
        2,
        2,
        0,
    )
    assert mda.mean().idxmax() in {"a", "e"} and mdi.shape[0] == 3
    assert -1 <= stability(mda) <= 1


def test_to_holding_dates_moves_each_return_to_the_next_session():
    from nvquant.evaluation.risk import to_holding_dates

    idx = pd.DatetimeIndex(["2025-01-23", "2025-01-24", "2025-01-27"])
    out = to_holding_dates(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx))
    assert list(out.index) == [
        pd.Timestamp("2025-01-24"),
        pd.Timestamp("2025-01-27"),
        pd.Timestamp("2025-01-28"),
    ]
