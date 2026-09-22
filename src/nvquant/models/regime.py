"""Gaussian HMM regimes with a hand-written forward filter.

hmmlearn estimates the parameters (Baum-Welch) on a training window. The state
probabilities used downstream come from :func:`forward_filter`, which computes
``P(s_t | x_1, ..., x_t)``: each value only uses observations up to t.
hmmlearn's ``predict_proba`` returns smoothed posteriors ``P(s_t | x_1..x_T)``
and ``predict`` returns a Viterbi path, both of which use the future and are
never called in this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass(frozen=True)
class HMMParams:
    """Gaussian HMM parameters, states sorted by the variance of the first feature."""

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @property
    def n_states(self) -> int:
        """Number of hidden states."""
        return len(self.startprob)


def fit_hmm(x: np.ndarray, n_states: int = 2, n_iter: int = 200, seed: int = 0) -> HMMParams:
    """Fit a full-covariance Gaussian HMM on standardized training observations.

    Standardization uses the training rows only, and the same mean and std are
    stored so the filter applies the identical transform to later rows.
    States are relabeled so state 0 has the lowest variance in the first column
    (the return), which fixes label switching across refits.
    """
    from hmmlearn.hmm import GaussianHMM

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd
    z = z[~np.isnan(z).any(axis=1)]
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
        min_covar=1e-4,
    )
    model.fit(z)
    order = np.argsort(model.covars_[:, 0, 0])
    return HMMParams(
        startprob=np.asarray(model.startprob_)[order],
        transmat=np.asarray(model.transmat_)[np.ix_(order, order)],
        means=np.asarray(model.means_)[order],
        covars=np.asarray(model.covars_)[order],
        feature_mean=mu,
        feature_std=sd,
    )


def forward_filter(x: np.ndarray, params: HMMParams) -> np.ndarray:
    """Filtered state probabilities ``P(s_t | x_1..x_t)`` for each row.

    Rows with missing observations carry the previous filtered distribution
    forward through the transition matrix (a pure prediction step).

    Parameters
    ----------
    x : ndarray, shape (T, d)
        Raw (unstandardized) observations in time order.
    params : HMMParams
        Parameters from :func:`fit_hmm`.

    Returns
    -------
    ndarray, shape (T, n_states)
    """
    z = (x - params.feature_mean) / params.feature_std
    k = params.n_states
    log_a = np.log(params.transmat + 1e-300)
    log_emis = np.full((len(z), k), np.nan)
    ok = ~np.isnan(z).any(axis=1)
    for s in range(k):
        dist = multivariate_normal(params.means[s], params.covars[s], allow_singular=True)
        log_emis[ok, s] = dist.logpdf(z[ok])
    out = np.empty((len(z), k))
    log_prev = np.log(params.startprob + 1e-300)
    first = True
    for t in range(len(z)):
        log_pred = log_prev if first else logsumexp(log_prev[:, None] + log_a, axis=0)
        first = False
        log_post = log_pred + log_emis[t] if ok[t] else log_pred
        log_post = log_post - logsumexp(log_post)
        out[t] = np.exp(log_post)
        log_prev = log_post
    return out


def regime_observations(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """HMM inputs: daily log return and log Parkinson range (both known at the close)."""
    r = np.log(close).diff()
    rng = np.log((np.log(high / low) ** 2 / (4 * np.log(2))).clip(lower=1e-10)) / 2
    return pd.DataFrame({"ret": r, "log_range": rng})
