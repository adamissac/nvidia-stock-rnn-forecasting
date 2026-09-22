import numpy as np
import pandas as pd
import pytest
import torch

from nvquant.config.schema import ModelConfig
from nvquant.data.synthetic import gbm, planted_signal, regime_switching
from nvquant.evaluation.forecast import safe_spearman
from nvquant.models.baselines import ARForecaster, HistMeanForecaster, ZeroForecaster
from nvquant.models.deep.train import DeepForecaster, windows
from nvquant.models.ensemble import equal_weight, stacked
from nvquant.models.factory import build_model
from nvquant.models.linear import ElasticNetForecaster, RidgeForecaster
from nvquant.models.regime import fit_hmm, forward_filter
from nvquant.models.trees import LGBMClassifierModel, LGBMForecaster
from nvquant.models.v1_replica import V1Net, n_parameters, run_v1_replica, v1_metrics


def _split(n=3000, strength=0.15, seed=0):
    X, y = planted_signal(n, n_features=8, strength=strength, seed=seed)
    tr, te = y.index[: n * 2 // 3], y.index[n * 2 // 3 :]
    return X, y, tr, te


@pytest.mark.parametrize(
    "model",
    [
        RidgeForecaster([1.0, 100.0]),
        ElasticNetForecaster([0.001, 0.01]),
        LGBMForecaster(n_estimators=150, learning_rate=0.03, num_leaves=5, min_child_samples=50),
    ],
)
def test_models_recover_planted_signal(model):
    X, y, tr, te = _split()
    pred = model.fit(X, y.loc[tr]).predict(X, te)
    assert pred.index.equals(te)
    assert safe_spearman(pred, y.loc[te]) > 0.08


def test_models_find_nothing_in_noise():
    X, y, tr, te = _split(strength=0.0, seed=1)
    ic = safe_spearman(RidgeForecaster([1.0, 1000.0]).fit(X, y.loc[tr]).predict(X, te), y.loc[te])
    assert abs(ic) < 0.08  # about 2.5 standard errors at n=1000


def test_baselines():
    X, y, tr, te = _split(500)
    assert (ZeroForecaster().fit(X, y).predict(X, te) == 0).all()
    assert HistMeanForecaster().fit(X, y.loc[tr]).predict(X, te).iloc[0] == pytest.approx(
        y.loc[tr].mean()
    )
    p = gbm(800, seed=3)
    r = np.log(p).diff()
    Xr = pd.DataFrame({"ret_1": r})
    yr = r.shift(-2).dropna()
    ar = ARForecaster(max_lag=3).fit(Xr, yr.iloc[:500])
    assert 1 <= ar.p_ <= 3 and len(ar.predict(Xr, yr.index[500:])) == len(yr) - 500


def test_lgbm_shap_and_gain_and_classifier():
    X, y, tr, te = _split(800)
    m = LGBMForecaster(n_estimators=50).fit(X, y.loc[tr])
    shap = m.shap_values(X, te)
    assert shap.shape == (len(te), X.shape[1])
    # TreeSHAP contributions plus the bias add up to the prediction
    raw = m.model_.predict(X.loc[te].to_numpy(), pred_contrib=True)
    np.testing.assert_allclose(raw.sum(axis=1), m.predict(X, te), rtol=1e-6)
    assert m.gain_importance().sum() == pytest.approx(1.0)
    clf = LGBMClassifierModel(n_estimators=20).fit(X.loc[tr].to_numpy(), (y.loc[tr] > 0).to_numpy())
    p = clf.predict_proba(X.loc[te].to_numpy())
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize(
    "arch,head,extra",
    [
        ("rnn", "gaussian", {"cell": "lstm", "hidden": 4}),
        ("rnn", "quantile", {"cell": "gru", "hidden": 4}),
        ("tcn", "gaussian", {"channels": 4, "levels": 2}),
        ("patchtst", "quantile", {"d_model": 8, "patch_len": 4, "stride": 2}),
    ],
)
def test_deep_models_fit_predict_deterministic(arch, head, extra):
    X, y, tr, te = _split(700)
    kw = dict(arch=arch, head=head, seq_len=12, seeds=2, max_epochs=3, batch_size=64, **extra)
    a = DeepForecaster(**kw).fit(X, y.loc[tr])
    b = DeepForecaster(**kw).fit(X, y.loc[tr])
    d = a.predict_dist(X, te)
    assert d.index.equals(te) and d.notna().all().all()
    np.testing.assert_allclose(a.predict(X, te), b.predict(X, te), rtol=1e-5, atol=1e-6)
    if head == "quantile":
        assert (d["q0.1"] <= d["q0.5"]).all() and (d["q0.5"] <= d["q0.9"]).all()
    else:
        assert (d["std"] > 0).all()


def test_deep_model_errors_and_windows():
    X, y, tr, _ = _split(100)
    with pytest.raises(ValueError):
        DeepForecaster("rnn", batch_size=512).fit(X, y.loc[tr])
    with pytest.raises(ValueError):
        DeepForecaster("nope")._net(3)
    Z = np.arange(12, dtype=np.float32).reshape(6, 2)
    w = windows(Z, np.array([0, 5]), 3)
    assert w.shape == (2, 3, 2) and (w[0, :2] == 0).all() and (w[1, -1] == Z[5]).all()


def test_factory_builds_every_kind(fast_cfg):
    for mcfg in fast_cfg.models.values():
        assert build_model(mcfg, fast_cfg).name
    with pytest.raises(KeyError):
        build_model(ModelConfig(kind="nope"), fast_cfg)


def test_hmm_forward_filter_matches_final_smoothed_posterior():
    rp = regime_switching(1500, seed=4)
    x = np.column_stack([rp.returns.to_numpy(), np.log(rp.returns.abs() + 0.1)])
    params = fit_hmm(x, 2, 50, seed=0)
    f = forward_filter(x, params)
    np.testing.assert_allclose(f.sum(axis=1), 1.0)
    # at the last observation, filtered and smoothed probabilities coincide
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(2, covariance_type="full")
    m.startprob_, m.transmat_, m.means_, m.covars_ = (
        params.startprob,
        params.transmat,
        params.means,
        params.covars,
    )
    z = (x - params.feature_mean) / params.feature_std
    np.testing.assert_allclose(f[-1], m.predict_proba(z)[-1], atol=1e-6)
    # the high-vol state (last) is more likely in the high-vol regime
    assert f[rp.states.to_numpy() == 1, -1].mean() > f[rp.states.to_numpy() == 0, -1].mean()
    x_nan = x.copy()
    x_nan[10] = np.nan
    assert np.isfinite(forward_filter(x_nan, params)).all()


def test_ensembles():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=400), index=idx)
    P = pd.DataFrame({"good": y + rng.normal(0, 1, 400), "bad": rng.normal(size=400)}, index=idx)
    assert equal_weight(P).iloc[0] == pytest.approx(P.iloc[0].mean())
    t_end = pd.Series(idx[np.minimum(np.arange(400) + 2, 399)], index=idx)
    s, w = stacked(P, y, t_end, refit_every=50, min_history=100)
    assert s.notna().all() and np.allclose(w.sum(axis=1), 1.0)
    assert w.iloc[-1]["good"] > w.iloc[-1]["bad"]


def test_v1_replica_matches_v1_architecture():
    assert n_parameters(V1Net()) == 282_101
    opens = gbm(500, seed=5)
    res = run_v1_replica(
        opens, opens.index[400], opens.index[-1], train_len=100, refit_every=60, epochs=1
    )
    assert len(res.frame) == len(opens.index[400:])
    m = v1_metrics(res.frame)
    assert m["rmse_persistence"] > 0 and 0 <= m["direction_accuracy"] <= 1
    torch.manual_seed(0)
    out = V1Net(units=4, layers=2)(torch.randn(3, 10, 1))
    assert out.shape == (3,)
