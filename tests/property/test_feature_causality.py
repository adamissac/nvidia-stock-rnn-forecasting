"""The signature test: perturbing the future never changes a feature's past.

For every registered feature spec (and the fitted features), for random cutoffs
t, every value strictly after t is perturbed; every feature value at or before
t must be unchanged. Scheduled inputs declare ``known_ahead``, and only data
more than that many sessions past t is perturbed for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nvquant.config.schema import FeaturesConfig, FracdiffConfig, RegimeConfig, UniverseConfig
from nvquant.data.synthetic import synthetic_market
from nvquant.features.fitted import FracDiffFeature, RegimeFeature, walk_forward_fitted
from nvquant.features.registry import FeatureParams, all_specs
from nvquant.models.regime import fit_hmm, forward_filter, regime_observations

MARKET = synthetic_market(
    UniverseConfig(), n_sessions=420, start="2018-01-02", signal_strength=0.2, seed=11
)
N = len(MARKET.sessions)
PARAMS = FeatureParams.from_config(FeaturesConfig())
SPECS = all_specs()
SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
@SETTINGS
@given(cut=st.integers(min_value=5, max_value=N - 3), seed=st.integers(0, 10_000))
def test_feature_is_causal(spec, cut, seed):
    t = MARKET.sessions[cut]
    t_perturb = MARKET.sessions[min(cut + spec.known_ahead, N - 1)]
    perturbed = MARKET.perturb_after(t_perturb, seed=seed)
    a = spec.compute(MARKET, PARAMS).loc[:t]
    b = spec.compute(perturbed, PARAMS).loc[:t]
    _same(a, b)


@SETTINGS
@given(cut=st.integers(min_value=260, max_value=N - 3), seed=st.integers(0, 10_000))
def test_fitted_features_are_causal(cut, seed):
    t = MARKET.sessions[cut]
    perturbed = MARKET.perturb_after(t, seed=seed)
    start = MARKET.sessions[0]
    for feat in (
        FracDiffFeature(FracdiffConfig(max_width=60)),
        RegimeFeature(RegimeConfig(n_iter=15)),
    ):
        a, _ = walk_forward_fitted(MARKET, feat, start, 200, 60)
        b, _ = walk_forward_fitted(perturbed, type(feat)(feat.cfg), start, 200, 60)
        _same(a.loc[:t], b.loc[:t])


OBS = regime_observations(
    MARKET.ohlcv()["close"], MARKET.ohlcv()["high"], MARKET.ohlcv()["low"]
).to_numpy()
HMM = fit_hmm(OBS[1:300], n_states=2, n_iter=30, seed=0)


@settings(max_examples=50, deadline=None)
@given(cut=st.integers(min_value=5, max_value=N - 3), seed=st.integers(0, 10_000))
def test_forward_filter_is_causal(cut, seed):
    rng = np.random.default_rng(seed)
    x2 = OBS.copy()
    x2[cut + 1 :] += rng.normal(0, 1, size=x2[cut + 1 :].shape)
    np.testing.assert_allclose(
        forward_filter(OBS, HMM)[: cut + 1], forward_filter(x2, HMM)[: cut + 1]
    )
