"""The shared forecaster interface."""

from __future__ import annotations

import abc
from typing import Self

import numpy as np
import pandas as pd


class Forecaster(abc.ABC):
    """A model that maps feature rows to a return forecast.

    ``fit`` receives every feature row up to the training cutoff (sequence models
    need the history before the first label) plus labels for the training
    dates. ``predict`` receives every row up to the last prediction date. The
    harness guarantees neither call sees rows past its cutoff.
    """

    name: str = "forecaster"
    probabilistic: bool = False
    quantiles: tuple[float, ...] = ()

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Fit on rows ``y.index`` (a subset of ``X.index``)."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Point forecast for each date in ``index``."""

    def predict_dist(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Distribution summary: ``mean`` and ``std``, and/or quantile columns ``q0.1`` etc."""
        raise NotImplementedError(f"{self.name} is not probabilistic")


def tabular_rows(X: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Rows of ``X`` for ``index`` as a float array with NaNs replaced by 0.

    NaN only appears in warmup rows of slow features; replacing with 0 after
    standardization means "at the mean". The harness drops rows where the
    target model's required inputs are all missing.
    """
    return np.nan_to_num(X.loc[index].to_numpy(dtype=float), nan=0.0)
