"""Forecast ensembles built from the members' out-of-sample forecasts.

- Equal weight: the average of the members' vol-normalized forecasts.
- Stacking: non-negative least squares weights on the members' past OOS
  forecasts, refit every ``refit_every`` sessions using only rows whose label
  resolved before the refit date (purged), with the weights normalized to sum
  to one. Its own forecasts are therefore also out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def equal_weight(preds: pd.DataFrame) -> pd.Series:
    """Row mean of member forecasts (rows with any missing member are dropped)."""
    return preds.dropna().mean(axis=1)


def stacked(
    preds: pd.DataFrame,
    y: pd.Series,
    t_end: pd.Series,
    refit_every: int = 21,
    min_history: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward NNLS stacking.

    Returns the stacked forecast and the weight history (one row per refit).
    Before ``min_history`` resolved rows exist, it falls back to equal weights.
    """
    P = preds.dropna()
    dates = P.index
    out = pd.Series(np.nan, index=dates)
    weights = []
    for k in range(0, len(dates), refit_every):
        r = dates[k]
        block = dates[k : k + refit_every]
        ends = t_end.reindex(P.index)
        hist = P.index[
            (P.index < r) & (ends < r).to_numpy() & y.reindex(P.index).notna().to_numpy()
        ]
        if len(hist) >= min_history:
            w, _ = nnls(P.loc[hist].to_numpy(), y.loc[hist].to_numpy())
            w = w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])
        else:
            w = np.full(P.shape[1], 1 / P.shape[1])
        weights.append(pd.Series(w, index=P.columns, name=r))
        out.loc[block] = P.loc[block].to_numpy() @ w
    return out, pd.DataFrame(weights)
