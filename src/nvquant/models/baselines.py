"""Naive return baselines: zero (random walk), historical mean, AR(p)."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from nvquant.models.base import Forecaster


class ZeroForecaster(Forecaster):
    """Predicts a zero return: the random-walk benchmark for R2_oos."""

    name = "zero"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Nothing to fit."""
        return self

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """All zeros."""
        return pd.Series(0.0, index=index)


class HistMeanForecaster(Forecaster):
    """Predicts the training-sample mean of the target (the Goyal-Welch benchmark)."""

    name = "hist_mean"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Store the training mean."""
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """The training mean for every date."""
        return pd.Series(self._mean, index=index)


class ARForecaster(Forecaster):
    """Linear regression of the target on lags of the target's daily log return.

    The label at t is the open(t+1) to open(t+2) return, which isn't known at
    t, so the regressors are the close-to-close returns ``ret_1`` at t, t-1,
    ..., t-p+1 (all known at t). The order p <= ``max_lag`` is chosen by BIC on
    the training rows.
    """

    name = "ar"

    def __init__(self, max_lag: int = 5, return_col: str = "ret_1") -> None:
        self.max_lag = max_lag
        self.return_col = return_col

    def _lags(self, X: pd.DataFrame, p: int) -> pd.DataFrame:
        r = X[self.return_col]
        return pd.DataFrame({f"lag{k}": r.shift(k) for k in range(p)}, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Choose p by BIC and fit OLS."""
        best: tuple[float, int, np.ndarray] | None = None
        for p in range(1, self.max_lag + 1):
            lags = self._lags(X, p).loc[y.index]
            ok = lags.notna().all(axis=1) & y.notna()
            A = np.column_stack([np.ones(int(ok.sum())), lags[ok].to_numpy()])
            b = y[ok].to_numpy()
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            resid = b - A @ coef
            n = len(b)
            bic = n * np.log(resid @ resid / n) + (p + 1) * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, p, coef)
        assert best is not None
        _, self.p_, self.coef_ = best
        return self

    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Linear forecast from the lagged returns."""
        lags = self._lags(X, self.p_).loc[index].fillna(0.0).to_numpy()
        return pd.Series(self.coef_[0] + lags @ self.coef_[1:], index=index)
