"""Regularized linear models. Scaling and penalty choice happen on training rows only."""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from nvquant.cv.splits import PurgedKFold
from nvquant.models.base import Forecaster, tabular_rows


def _inner_cv_score(
    make: type[Ridge] | type[ElasticNet],
    kwargs: dict[str, float],
    Xs: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean squared error across purged inner folds."""
    errs = []
    for tr, te in folds:
        m = make(**kwargs)
        with warnings.catch_warnings():
            # a large alpha can leave elastic net at max_iter; that candidate just scores worse
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xs[tr], y[tr], sample_weight=w[tr])
        errs.append(float(np.mean((m.predict(Xs[te]) - y[te]) ** 2)))
    return float(np.mean(errs))


class _PenalizedLinear(Forecaster):
    """Shared fit/predict for ridge and elastic net with purged inner-CV penalty choice."""

    estimator: type[Ridge] | type[ElasticNet] = Ridge

    def __init__(self, alphas: list[float], inner_splits: int = 3, embargo: int = 5) -> None:
        self.alphas = alphas
        self.inner_splits = inner_splits
        self.embargo = embargo

    def _kwargs(self, alpha: float) -> dict[str, float]:
        return {"alpha": alpha}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Standardize on training rows, pick alpha by purged inner CV, refit on all rows."""
        rows = tabular_rows(X, y.index)
        self.scaler_ = StandardScaler().fit(rows)
        Xs = np.nan_to_num(self.scaler_.transform(rows))
        yv = y.to_numpy(dtype=float)
        w = np.ones(len(yv)) if sample_weight is None else sample_weight.to_numpy(dtype=float)
        t_end = pd.Series(
            X.index[np.minimum(X.index.get_indexer(y.index) + 2, len(X) - 1)], index=y.index
        )
        folds = PurgedKFold(self.inner_splits, self.embargo).sklearn_splits(
            t_end, pd.DatetimeIndex(X.index)
        )
        scores = {
            a: _inner_cv_score(self.estimator, self._kwargs(a), Xs, yv, w, folds)
            for a in self.alphas
        }
        self.alpha_ = min(scores, key=lambda a: scores[a])
        self.model_ = self.estimator(**self._kwargs(self.alpha_))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(Xs, yv, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Linear forecast."""
        Xs = np.nan_to_num(self.scaler_.transform(tabular_rows(X, index)))
        return pd.Series(self.model_.predict(Xs), index=index)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients on standardized features."""
        return np.asarray(self.model_.coef_)


class RidgeForecaster(_PenalizedLinear):
    """Ridge regression."""

    name = "ridge"
    estimator = Ridge


class ElasticNetForecaster(_PenalizedLinear):
    """Elastic net with a fixed L1 ratio."""

    name = "elastic_net"
    estimator = ElasticNet

    def __init__(
        self, alphas: list[float], l1_ratio: float = 0.5, inner_splits: int = 3, embargo: int = 5
    ) -> None:
        super().__init__(alphas, inner_splits, embargo)
        self.l1_ratio = l1_ratio

    def _kwargs(self, alpha: float) -> dict[str, float]:
        return {"alpha": alpha, "l1_ratio": self.l1_ratio, "max_iter": 5000}
