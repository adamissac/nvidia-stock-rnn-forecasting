"""Nested hyperparameter tuning with Optuna, inside one walk-forward training window.

The objective is the mean squared error over purged k-fold splits of the
training rows only. The walk-forward test block is never seen. Every trial is
returned so the caller can write it to the registry.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd

from nvquant.config.schema import TuningConfig
from nvquant.cv.splits import PurgedKFold
from nvquant.models.trees import LGBMForecaster


def lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for LightGBM. Kept shallow and heavily regularized (low signal-to-noise)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 3, 31, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
    }


def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    t_end: pd.Series,
    base_params: dict[str, Any],
    tcfg: TuningConfig,
    embargo: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune LightGBM on purged folds of the training window.

    Returns the best parameters (merged over ``base_params``) and one dict per trial.
    """
    sessions = pd.DatetimeIndex(X.index)
    folds = PurgedKFold(tcfg.inner_splits, embargo).split(t_end.loc[y.index], sessions)
    dates = y.index
    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **lgbm_space(trial)}
        start = time.perf_counter()
        errs = []
        for f in folds:
            tr, te = dates[f.train], dates[f.test]
            m = LGBMForecaster(seed=seed, **params).fit(X, y.loc[tr])
            errs.append(float(np.mean((m.predict(X, te) - y.loc[te]) ** 2)))
        score = float(np.mean(errs))
        records.append(
            {
                "params": params,
                "inner_mse": score,
                "fold_mse": errs,
                "wall_time_s": time.perf_counter() - start,
            }
        )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=tcfg.n_trials, timeout=tcfg.timeout_seconds)
    return {**base_params, **study.best_params}, records
