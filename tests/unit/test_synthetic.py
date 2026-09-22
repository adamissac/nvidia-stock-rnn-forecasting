import numpy as np
import pytest

from nvquant.config.schema import UniverseConfig
from nvquant.data.synthetic import garch, gbm, planted_signal, regime_switching, synthetic_market


def test_gbm_log_returns_are_iid_normalish():
    p = gbm(5000, mu=0.0, sigma=0.2, seed=1)
    r = np.log(p).diff().dropna()
    assert abs(r.std() * np.sqrt(252) - 0.2) < 0.01
    assert abs(r.autocorr()) < 0.05


def test_garch_variance_clusters_and_rejects_nonstationary():
    g = garch(4000, seed=2)
    assert (g.variance > 0).all()
    assert (g.returns**2).autocorr() > 0.05
    with pytest.raises(ValueError):
        garch(10, alpha=0.5, beta=0.6)


def test_regime_switching_states_have_different_vol():
    rp = regime_switching(4000, seed=3)
    assert set(rp.states.unique()) == {0, 1}
    assert rp.returns[rp.states == 1].std() > 2 * rp.returns[rp.states == 0].std()


def test_planted_signal_correlation():
    x, y = planted_signal(20000, strength=0.1, seed=4)
    assert 0.07 < np.corrcoef(x["x0"], y)[0, 1] < 0.13
    assert abs(np.corrcoef(x["x1"], y)[0, 1]) < 0.03


def test_synthetic_market_schema():
    m = synthetic_market(UniverseConfig(), n_sessions=300, seed=5)
    assert len(m.sessions) == 300 and m.sessions.is_monotonic_increasing
    for t, df in m.prices.items():
        assert list(df.columns) == ["open", "high", "low", "close", "volume"], t
        assert (df["high"] >= df["low"]).all()
    assert set(m.rates.columns) == {"DGS10", "DGS2"}
    assert {"mkt_rf", "rf", "mom"} <= set(m.factors.columns)
    assert "NVDA" in m.earnings
