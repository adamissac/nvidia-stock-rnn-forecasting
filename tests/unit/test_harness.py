"""The walk-forward harness never lets a prediction see data from after its date."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nvquant.data.loader import load_market_data, write_synthetic
from nvquant.experiments.harness import first_test_date, sample_index, walk_forward
from nvquant.features.store import build_store


@pytest.fixture(scope="module")
def store_and_cfg(tmp_path_factory):
    from pathlib import Path

    from nvquant.config import load_config

    root = Path(__file__).resolve().parents[2]
    tmp = tmp_path_factory.mktemp("harness")
    cfg = load_config(root / "configs/fast.yaml", overrides={"paths": {"data_dir": str(tmp / "data"), "reports_dir": str(tmp / "r")}})
    write_synthetic(cfg)
    return build_store(cfg, load_market_data(cfg, "dev")), cfg


@pytest.mark.parametrize("model", ["ridge", "lgbm"])
def test_predictions_ignore_everything_after_their_date(store_and_cfg, model):
    store, cfg = store_and_cfg
    mcfg = cfg.models[model].model_copy(update={"tune": False})
    base = walk_forward(model, mcfg, store, cfg).frame
    D = base.index[len(base) // 2]
    rng = np.random.default_rng(0)
    feats = store.features.copy()
    after = feats.index > D
    feats.loc[after] = rng.normal(size=(int(after.sum()), feats.shape[1]))
    labels = store.labels.copy()
    late = (labels["t_end_1"] >= D).to_numpy()
    for col in ("fwd_ret_1", "fwd_ret_vn_1"):
        labels.loc[late, col] = rng.normal(size=int(late.sum()))
    other = walk_forward(model, mcfg, replace(store, features=feats, labels=labels), cfg).frame
    before = base.index[base.index <= D]
    pd.testing.assert_series_equal(base.loc[before, "pred"], other.loc[before, "pred"])


def test_fold_metrics_and_first_test_date(store_and_cfg):
    store, cfg = store_and_cfg
    res = walk_forward("ridge", cfg.models["ridge"], store, cfg)
    for fm in res.fold_metrics:
        assert fm["train_end"] < fm["refit_date"]
    first = first_test_date(store, cfg)
    assert res.frame.index[0] == first
    refit = max(pd.Timestamp(v) for v in store.info["fitted_first_refit"].values())
    assert first > refit
    late = replace(store, info={**store.info, "fitted_first_refit": {"x": str(store.features.index[-200].date())}})
    idx = sample_index(late, cfg.labels.target)
    assert first_test_date(late, cfg) == idx[idx > store.features.index[-200]][0]
