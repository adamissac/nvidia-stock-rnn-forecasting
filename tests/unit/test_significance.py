"""PSR, DSR, MinTRL, and PBO against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from nvquant.evaluation.significance import (
    bootstrap_sharpe_ci,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    psr,
    psr_from_returns,
    spa_test,
)


def test_psr_hand_example():
    # SR = 0.1 per day, T = 253, normal returns: Phi(0.1 * sqrt(252) / sqrt(1 + 0.1^2 / 2))
    assert psr(0.1, 253, 0.0, 3.0) == pytest.approx(0.943345882804017, abs=1e-12)
    assert psr(0.1, 253, 0.0, 3.0, sr_star=0.1) == pytest.approx(0.5)
    assert np.isnan(psr(0.1, 1, 0.0, 3.0))


def test_expected_max_sharpe_hand_example():
    # N = 100 trials with Sharpe std 0.01: 0.01 * ((1 - g) Phi^-1(0.99) + g Phi^-1(1 - 1/(100 e)))
    assert expected_max_sharpe(100, 1e-4) == pytest.approx(0.025306028932016847, rel=1e-12)
    assert expected_max_sharpe(1, 1e-4) == 0.0


def test_min_trl_hand_example():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200_000)
    sr = r.mean() / r.std(ddof=1)
    assert min_track_record_length(r) == pytest.approx(
        1 + (1 + sr**2 / 2) * (1.6448536269514722 / sr) ** 2, rel=0.01
    )
    assert min_track_record_length(-np.abs(r)) == float("inf")


def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0008, 0.01, 2500)
    trials = rng.normal(0, 0.02, 50)
    few = deflated_sharpe(r, 2, trials[:2])
    many = deflated_sharpe(r, 50, trials)
    assert many.sr_star > few.sr_star and many.dsr < few.dsr
    assert psr_from_returns(r) > many.dsr


def test_pbo_low_for_dominant_strategy_and_high_for_noise():
    rng = np.random.default_rng(2)
    T = 1600
    noise = pd.DataFrame(rng.normal(0, 0.01, (T, 20)))
    assert 0.2 < pbo_cscv(noise, 8).pbo < 0.8
    dominant = noise.copy()
    dominant[0] += 0.004
    res = pbo_cscv(dominant, 8)
    assert res.pbo == 0.0 and res.n_combinations == 70


def test_spa_and_bootstrap():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    bench = pd.Series(rng.normal(0, 0.01, 1500), index=idx, name="b")
    models = pd.DataFrame(
        {"good": rng.normal(0.002, 0.01, 1500), "bad": rng.normal(0, 0.01, 1500)}, index=idx
    )
    res = spa_test(bench, models, reps=200, seed=0)
    assert res.spa_consistent < 0.05 and 0 <= res.reality_check <= 1
    lo, hi, block = bootstrap_sharpe_ci(models["good"], 200, seed=0)
    sr = models["good"].mean() / models["good"].std() * np.sqrt(252)
    assert lo < sr < hi and block >= 1
    lo, hi, _ = bootstrap_sharpe_ci(pd.Series(0.0, index=idx), 20, seed=0)
    assert np.isnan(lo) and np.isnan(hi)
