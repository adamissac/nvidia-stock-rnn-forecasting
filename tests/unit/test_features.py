import numpy as np
import pandas as pd
import pytest

from nvquant.config.schema import FeaturesConfig, FracdiffConfig, RegimeConfig
from nvquant.features.fitted import FracDiffFeature, RegimeFeature, refit_dates, walk_forward_fitted
from nvquant.features.fracdiff import ffd_weights, frac_diff, min_d_passing_adf
from nvquant.features.price import realized_vols, rsi
from nvquant.features.registry import (
    FeatureSpec,
    all_specs,
    build_features,
    column_metadata,
    register,
)


def test_registry_builds_unique_columns(market):
    F = build_features(market, FeaturesConfig())
    assert F.index.equals(market.sessions)
    assert not F.columns.duplicated().any()
    meta = column_metadata(market, FeaturesConfig())
    assert set(meta["column"]) == set(F.columns)
    assert (meta.loc[meta["group"].isin(["cross_asset", "macro"]), "lag"] == 1).all()


def test_exclude_and_bad_index(market):
    F = build_features(market, FeaturesConfig(exclude=["macro"]))
    assert "vix_log" not in F.columns
    bad = FeatureSpec("bad", "x", lambda m, p: pd.DataFrame({"a": [1.0]}), lookback=0)
    with pytest.raises(ValueError):
        bad.compute(market, None)  # type: ignore[arg-type]


def test_duplicate_registration_rejected():
    all_specs()
    with pytest.raises(ValueError):
        register("returns", "returns", 1)(lambda m, p: pd.DataFrame())


def test_returns_feature_matches_definition(market):
    F = build_features(market, FeaturesConfig())
    c = market.ohlcv()["close"]
    assert F["ret_5"].iloc[10] == pytest.approx(np.log(c.iloc[10] / c.iloc[5]))


def test_rsi_bounds_and_realized_vol_constant_path():
    s = pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 300)))
    r = rsi(s).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    px = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=range(50))
    v = realized_vols(px, 21)
    assert v["rv_cc_21"].dropna().eq(0).all() and v["rv_park_21"].dropna().eq(0).all()


def test_ffd_weights():
    assert np.allclose(ffd_weights(1.0), [-1.0, 1.0])
    assert np.allclose(ffd_weights(0.0), [1.0])
    w = ffd_weights(0.4, threshold=1e-3)
    assert w[-1] == 1.0 and len(w) > 5


def test_frac_diff_d1_is_first_difference():
    x = pd.Series(np.arange(10.0) ** 2)
    assert np.allclose(frac_diff(x, 1.0).dropna(), np.diff(x.to_numpy()))


def test_min_d_passes_adf_for_random_walk():
    rw = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=1500)))
    d, pv = min_d_passing_adf(rw, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert d > 0.0 and pv[d] < 0.05
    assert min_d_passing_adf(rw.iloc[:50], [0.5])[0] == 0.5  # too short: falls back to the max


def test_walk_forward_fitted(market):
    start = market.sessions[0]
    frame, recs = walk_forward_fitted(
        market, FracDiffFeature(FracdiffConfig(max_width=60)), start, 200, 150
    )
    assert frame.index.equals(market.sessions) and len(recs) == len(
        refit_dates(market.sessions, start, 200, 150)
    )
    regime, _ = walk_forward_fitted(market, RegimeFeature(RegimeConfig(n_iter=20)), start, 200, 300)
    p = regime["regime_p_high"].dropna()
    assert ((p >= 0) & (p <= 1)).all()
    with pytest.raises(ValueError):
        refit_dates(market.sessions, start, 10_000, 10)
    with pytest.raises(RuntimeError):
        FracDiffFeature(FracdiffConfig()).transform(market)
    with pytest.raises(RuntimeError):
        RegimeFeature(RegimeConfig()).transform(market)
