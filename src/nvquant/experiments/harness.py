"""Walk-forward evaluation harness shared by every return model.

For each refit the harness: takes the training samples whose labels resolved
before the test block (purging), tunes if the model asks for it (inside the
training window only), fits, and predicts the test block. Predictions are made
in the target's units (vol-normalized returns by default) and converted back
to returns with the ex-ante vol known at the decision date.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from nvquant.config.schema import Config, ModelConfig
from nvquant.cv.splits import WalkForward
from nvquant.evaluation.forecast import safe_spearman
from nvquant.experiments.tuning import tune_lgbm
from nvquant.features.store import FeatureStore
from nvquant.models.factory import build_model


@dataclass
class ForecastResult:
    """Out-of-sample forecasts of one model plus per-fold diagnostics."""

    name: str
    frame: pd.DataFrame
    fold_metrics: list[dict[str, Any]]
    tuning_trials: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0


def sample_index(store: FeatureStore, target: str) -> pd.DatetimeIndex:
    """Dates with a known label and a complete feature row."""
    X = store.features
    y = store.labels[target]
    ok = X.notna().all(axis=1) & y.reindex(X.index).notna()
    return pd.DatetimeIndex(X.index[ok])


def first_test_date(store: FeatureStore, cfg: Config) -> pd.Timestamp:
    """First out-of-sample decision date, shared by every model.

    It is ``initial_train`` samples in, and strictly after the first refit of
    every fitted feature, so no test row uses burn-in (in-sample) fitted values.
    """
    idx = sample_index(store, cfg.labels.target)
    first = idx[min(cfg.cv.initial_train, len(idx) - 1)]
    refits = cast(dict[str, str], store.info.get("fitted_first_refit", {}))
    if refits:
        latest = max(pd.Timestamp(v) for v in refits.values())
        after = idx[idx > latest]
        first = max(first, after[0])
    return pd.Timestamp(first)


def walk_forward(
    name: str,
    mcfg: ModelConfig,
    store: FeatureStore,
    cfg: Config,
    first_test: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> ForecastResult:
    """Run one model through walk-forward.

    Parameters
    ----------
    name, mcfg : model name and config entry
    store : FeatureStore
    cfg : Config
    first_test : Timestamp, optional
        Override the first OOS date (the lockbox run passes ``lockbox.start``).
    end : Timestamp, optional
        Last decision date to predict.
    """
    t0 = time.perf_counter()
    target = cfg.labels.target
    X = store.features
    labels = store.labels
    idx = sample_index(store, target)
    y = labels[target].loc[idx]
    sigma = labels["sigma"]
    t_end = labels["t_end_1"].loc[idx]
    weights = labels["uniq_1"].loc[idx]
    ft = first_test or first_test_date(store, cfg)
    # Test dates also include the last sessions whose label isn't known yet, so a
    # forecast exists for every date a position could be taken on.
    pred_idx = pd.DatetimeIndex(X.index[X.notna().all(axis=1)])
    pred_idx = pred_idx[pred_idx >= ft]
    if end is not None:
        pred_idx = pred_idx[pred_idx <= end]
    every = mcfg.retrain_every or cfg.cv.retrain_every
    splitter = WalkForward(every, cfg.cv.scheme, cfg.cv.rolling_window)
    retune_every_blocks = max(1, cfg.tuning.retune_every // every)

    frames = []
    fold_metrics: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    tuned_params: dict[str, Any] | None = None
    for k, blk in enumerate(splitter.blocks(pred_idx, t_end)):
        r, block, train = blk.refit_date, blk.test, blk.train
        fs = time.perf_counter()
        hist = X.loc[: train.max()]
        if mcfg.tune and k % retune_every_blocks == 0:
            tuned_params, trials = tune_lgbm(
                hist, y.loc[train], t_end, dict(mcfg.params), cfg.tuning, cfg.cv.embargo, cfg.seed
            )
            for tr in trials:
                tuning.append({**tr, "model": name, "refit_date": str(r.date())})
        model = build_model(mcfg, cfg, tuned_params)
        model.fit(hist, y.loc[train], weights.loc[train])
        ctx = X.loc[: block.max()]
        out = pd.DataFrame({"pred_vn": model.predict(ctx, block)})
        if model.probabilistic:
            dist = model.predict_dist(ctx, block)
            for c in dist.columns:
                if c != "mean":
                    out[c + "_vn"] = dist[c]
        frames.append(out)
        yb = y.reindex(block).dropna()
        ic = safe_spearman(out["pred_vn"].loc[yb.index], yb)
        fold_metrics.append(
            {
                "refit_date": str(r.date()),
                "train_start": str(train.min().date()),
                "train_end": str(train.max().date()),
                "n_train": len(train),
                "n_test": len(block),
                "ic": ic,
                "mse": float(np.mean((out["pred_vn"].loc[yb.index] - yb) ** 2))
                if len(yb)
                else float("nan"),
                "fit_seconds": time.perf_counter() - fs,
                **({"params": tuned_params} if tuned_params else {}),
                **({"epochs": model.epochs_} if hasattr(model, "epochs_") else {}),  # type: ignore[attr-defined]
            }
        )
    frame = pd.concat(frames).sort_index()
    frame["sigma"] = sigma.reindex(frame.index)
    scale = frame["sigma"] if target.startswith("fwd_ret_vn") else 1.0
    frame["pred"] = frame["pred_vn"] * scale
    for c in [c for c in frame.columns if c.endswith("_vn") and c != "pred_vn"]:
        frame[c.removesuffix("_vn")] = frame[c] * scale
    frame["y_vn"] = labels[target].reindex(frame.index)
    frame["y"] = labels["fwd_ret_1"].reindex(frame.index)
    return ForecastResult(name, frame, fold_metrics, tuning, time.perf_counter() - t0)
