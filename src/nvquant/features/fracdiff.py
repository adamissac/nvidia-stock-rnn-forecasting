"""Fixed-width fractional differentiation (Lopez de Prado 2018, ch. 5).

``d`` is chosen as the smallest value on a grid whose fractionally
differentiated log price passes an ADF test, using training data only (see
:class:`nvquant.features.fitted.FracDiffFeature`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller

from nvquant.logging_utils import get_logger

log = get_logger(__name__)


def ffd_weights(d: float, threshold: float = 1e-4, max_width: int = 252) -> np.ndarray:
    """Weights ``w_k`` of ``(1 - B)^d``, truncated when ``|w_k| < threshold`` or at ``max_width``.

    ``w_0 = 1`` and ``w_k = -w_{k-1} (d - k + 1) / k``. Returned oldest-first so
    ``np.dot(w, x[t - width + 1 : t + 1])`` is the value at t.
    """
    w = [1.0]
    k = 1
    while k < max_width:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < threshold:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w[::-1])


def frac_diff(
    series: pd.Series, d: float, threshold: float = 1e-4, max_width: int = 252
) -> pd.Series:
    """Causal fixed-width fractional difference; the first ``width - 1`` values are NaN."""
    w = ffd_weights(d, threshold, max_width)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(x, width)
        out[width - 1 :] = windows @ w
    return pd.Series(out, index=series.index, name=series.name)


def min_d_passing_adf(
    series: pd.Series,
    grid: list[float],
    pvalue: float = 0.05,
    threshold: float = 1e-4,
    max_width: int = 252,
) -> tuple[float, dict[float, float]]:
    """Smallest ``d`` in ``grid`` whose fracdiff series has ADF p-value below ``pvalue``.

    Returns the chosen ``d`` and the p-value for every ``d`` tried. Falls back to
    the largest grid value if none pass.
    """
    pvals: dict[float, float] = {}
    for d in sorted(grid):
        fd = frac_diff(series, d, threshold, max_width).dropna()
        if len(fd) < 100:
            continue
        with warnings.catch_warnings():
            # statsmodels warns when the p-value is outside its lookup table (it clips to 0.001/0.1)
            warnings.simplefilter("ignore", InterpolationWarning)
            pvals[d] = float(adfuller(fd.to_numpy(), maxlag=1, regression="c", autolag=None)[1])
        if pvals[d] < pvalue:
            return d, pvals
    log.warning("no d in %s passed ADF at %.2f; using d=%s", grid, pvalue, max(grid))
    return max(grid), pvals
