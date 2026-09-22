"""Sample weights from average label uniqueness (Lopez de Prado 2018, ch. 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_uniqueness(
    t_start: pd.Series, t_end: pd.Series, sessions: pd.DatetimeIndex
) -> pd.Series:
    """Average uniqueness of each label over its lifespan.

    A label observed at t covers the open-to-open periods from t+1 to ``t_end``.
    The concurrency ``c_u`` of period u is how many labels cover it, and a
    label's uniqueness is the mean of ``1 / c_u`` over the periods it covers.

    Parameters
    ----------
    t_start : Series
        Decision date of each label (usually the index).
    t_end : Series
        Label end session (NaT for labels that aren't resolved yet; they get NaN).
    sessions : DatetimeIndex
        Session calendar.
    """
    pos_start = sessions.get_indexer(pd.DatetimeIndex(t_start)) + 1
    ok = t_end.notna().to_numpy()
    pos_end = np.full(len(t_end), -1)
    pos_end[ok] = sessions.get_indexer(pd.DatetimeIndex(t_end[ok]))
    # a label from t covers periods (open t+1 -> open t+2), ..., ending at t_end; index a
    # period by the session it starts on, so the label covers [t+1, t_end - 1]
    last = pos_end - 1
    conc = np.zeros(len(sessions) + 1)
    for s, e, good in zip(pos_start, last, ok, strict=True):
        if good and e >= s:
            conc[s] += 1
            conc[e + 1] -= 1
    conc = np.cumsum(conc)[: len(sessions)]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    out = np.full(len(t_end), np.nan)
    for i, (s, e, good) in enumerate(zip(pos_start, last, ok, strict=True)):
        if good and e >= s:
            out[i] = (csum[e + 1] - csum[s]) / (e - s + 1)
    return pd.Series(out, index=t_end.index, name="uniqueness")
