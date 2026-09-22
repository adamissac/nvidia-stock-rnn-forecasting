import numpy as np
import pandas as pd
import pytest

from nvquant.config.schema import VolConfig
from nvquant.data.synthetic import garch
from nvquant.models.volatility import (
    all_vol_forecasts,
    ewma_forecast,
    qlike,
    realized_variance_proxy,
    score_vol,
    vol_refit_dates,
)


def test_qlike_is_zero_at_truth_and_positive_elsewhere():
    rv = np.array([1.0, 2.0, 3.0])
    assert qlike(rv, rv) == pytest.approx(0.0)
    assert qlike(rv, rv * 2) > 0


def test_score_vol_mincer_zarnowitz_on_truth():
    g = garch(3000, seed=1)
    h = g.variance
    rv = h * np.random.default_rng(2).chisquare(5, len(h)) / 5
    s = score_vol(rv, h)
    assert s.mz_beta == pytest.approx(1.0, abs=0.15) and s.n == 3000


def test_ewma_and_proxy(market):
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    assert ewma_forecast(r).iloc[-1] == pytest.approx(1e-4, rel=0.5)
    rv = realized_variance_proxy(market)
    assert (rv.dropna() >= 0).all()


def test_all_vol_forecasts_are_causal_and_complete(market):
    from nvquant.config.schema import FeaturesConfig
    from nvquant.features.registry import build_features

    cfg = VolConfig(min_train=300, refit_every=150)
    F = build_features(market, FeaturesConfig())
    start = market.sessions[0]
    vf = all_vol_forecasts(market, F, start, cfg, seed=0)
    first = vol_refit_dates(market.sessions, start, cfg)[0]
    assert set(cfg.models) <= set(vf.columns)
    assert (vf.loc[first:, cfg.models].dropna() > 0).all().all()
    cut = market.sessions[450]
    vf2 = all_vol_forecasts(market.perturb_after(cut, seed=3), F, start, cfg, seed=0)
    cols = ["ewma", "garch", "gjr_t", "egarch", "har"]
    pd.testing.assert_frame_equal(vf.loc[:cut, cols], vf2.loc[:cut, cols], rtol=1e-6)
    with pytest.raises(KeyError):
        all_vol_forecasts(market, F, start, VolConfig(models=["nope"], min_train=300), seed=0)
