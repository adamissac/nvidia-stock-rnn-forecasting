"""Meta-labeling (Lopez de Prado 2018, ch. 3.6 and 10).

A primary rule picks the side (here: long when the close is above its 200-day
average, else flat). A secondary classifier, trained walk-forward on
triple-barrier outcomes of the primary's past bets, predicts the probability
that the current bet succeeds. The bet size is ``2 * Phi(z) - 1`` with
``z = (p - 0.5) / sqrt(p (1 - p))``, floored at zero, so the secondary model
can only shrink or skip the primary's bets, never flip them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nvquant.cv.splits import WalkForward
from nvquant.labels.triple_barrier import meta_labels
from nvquant.models.trees import LGBMClassifierModel


def trend_side(close: pd.Series, window: int) -> pd.Series:
    """Primary side: 1 when close > SMA(window), else 0 (causal)."""
    sma = close.rolling(window, min_periods=window).mean()
    return (close > sma).astype(float).where(sma.notna(), 0.0)


def bet_size(p: np.ndarray) -> np.ndarray:
    """``max(0, 2 Phi((p - 0.5) / sqrt(p (1 - p))) - 1)``."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    return np.maximum(0.0, 2 * norm.cdf(z) - 1)


def meta_label_positions(
    features: pd.DataFrame,
    side: pd.Series,
    tb: pd.DataFrame,
    uniqueness: pd.Series,
    pred_dates: pd.DatetimeIndex,
    retrain_every: int,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    """Walk-forward meta-labeled positions and the classifier's probabilities.

    Training rows are past dates where the primary was active, with the
    triple-barrier outcome resolved (``tb_t_end`` before the refit date).
    Samples are weighted by average uniqueness because barrier windows overlap.
    """
    y = meta_labels(side, tb).dropna()
    t_end = tb["tb_t_end"].reindex(y.index)
    X = features.assign(primary_side=side.reindex(features.index))
    prob = pd.Series(np.nan, index=pred_dates)
    for blk in WalkForward(retrain_every).blocks(pred_dates, t_end):
        train = blk.train
        if len(train) < 200 or y.loc[train].nunique() < 2:
            continue
        clf = LGBMClassifierModel(
            seed=seed, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50
        )
        clf.fit(
            np.nan_to_num(X.loc[train].to_numpy(float)),
            y.loc[train].to_numpy(),
            uniqueness.reindex(train).fillna(1.0).to_numpy(),
        )
        prob.loc[blk.test] = clf.predict_proba(np.nan_to_num(X.loc[blk.test].to_numpy(float)))
    active = side.reindex(pred_dates).fillna(0.0)
    pos = pd.Series(bet_size(prob.fillna(0.5).to_numpy()), index=pred_dates) * active
    return pos.rename("meta"), prob
