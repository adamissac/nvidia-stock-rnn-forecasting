"""LightGBM regressor and classifier on the shared interface."""

from __future__ import annotations

from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from nvquant.models.base import Forecaster

_FIXED = {"verbose": -1, "deterministic": True, "force_row_wise": True}


class LGBMForecaster(Forecaster):
    """Gradient-boosted trees (LightGBM). NaNs are handled natively; no scaling needed."""

    name = "lgbm"

    def __init__(self, seed: int = 0, n_jobs: int = 1, **params: Any) -> None:
        self.params = params
        self.seed = seed
        self.n_jobs = n_jobs

    def _model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            random_state=self.seed, n_jobs=self.n_jobs, **_FIXED, **self.params
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Fit on the training rows."""
        self.columns_ = list(X.columns)
        self.model_ = self._model()
        w = None if sample_weight is None else sample_weight.to_numpy(dtype=float)
        self.model_.fit(
            X.loc[y.index].to_numpy(dtype=float), y.to_numpy(dtype=float), sample_weight=w
        )
        return self

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Tree ensemble forecast."""
        return pd.Series(self.model_.predict(X.loc[index].to_numpy(dtype=float)), index=index)

    def shap_values(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Exact TreeSHAP contributions from LightGBM (``pred_contrib=True``), without the bias column."""
        contrib = self.model_.predict(X.loc[index].to_numpy(dtype=float), pred_contrib=True)
        return pd.DataFrame(np.asarray(contrib)[:, :-1], index=index, columns=self.columns_)

    def gain_importance(self) -> pd.Series:
        """Total split gain per feature (the MDI analogue for boosted trees)."""
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / max(gain.sum(), 1e-12), index=self.columns_)


class LGBMClassifierModel:
    """LightGBM binary classifier used for meta-labeling (predicts P(label = 1))."""

    def __init__(self, seed: int = 0, n_jobs: int = 1, **params: Any) -> None:
        self.params = params
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Self:
        """Fit on training rows."""
        self.model_ = lgb.LGBMClassifier(
            random_state=self.seed, n_jobs=self.n_jobs, **_FIXED, **self.params
        )
        self.model_.fit(X, y.astype(int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 1."""
        classes = list(self.model_.classes_)
        proba = np.asarray(self.model_.predict_proba(X))
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
