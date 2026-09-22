"""Feature importance: TreeSHAP, MDA under purged CV, MDI, clustered MDA, and stability.

References: Lopez de Prado (2018, ch. 8) and Lopez de Prado (2020, ch. 6) for
clustered feature importance. MDA here is the increase in out-of-fold MSE when
a feature (or a whole cluster) is permuted, on purged k-fold splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from nvquant.cv.splits import PurgedKFold
from nvquant.models.trees import LGBMForecaster


def correlation_clusters(X: pd.DataFrame, max_clusters: int = 12) -> dict[str, list[str]]:
    """Hierarchical clusters on the distance ``sqrt((1 - rho) / 2)`` (Ward linkage)."""
    corr = X.corr(method="spearman").fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    Z = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(Z, t=max_clusters, criterion="maxclust")
    out: dict[str, list[str]] = {}
    for col, lab in zip(X.columns, labels, strict=True):
        out.setdefault(f"c{lab:02d}", []).append(col)
    return out


def mda_purged(
    X: pd.DataFrame,
    y: pd.Series,
    t_end: pd.Series,
    sessions: pd.DatetimeIndex,
    groups: dict[str, list[str]],
    params: dict[str, object],
    n_splits: int,
    embargo: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation importance of each group on purged folds.

    Returns (per-fold importance, per-fold MDI) with one row per fold.
    """
    rng = np.random.default_rng(seed)
    folds = PurgedKFold(n_splits, embargo).split(t_end, sessions)
    dates = y.index
    mda_rows, mdi_rows = [], []
    for f in folds:
        tr, te = dates[f.train], dates[f.test]
        m = LGBMForecaster(seed=seed, **params).fit(X, y.loc[tr])  # type: ignore[arg-type]
        base = float(np.mean((m.predict(X, te) - y.loc[te]) ** 2))
        row = {}
        for g, cols in groups.items():
            incs = []
            for _ in range(repeats):
                Xp = X.loc[te].copy()
                perm = rng.permutation(len(te))
                Xp[cols] = Xp[cols].to_numpy()[perm]
                incs.append(float(np.mean((m.predict(Xp, te) - y.loc[te]) ** 2)) - base)
            row[g] = float(np.mean(incs))
        mda_rows.append(row)
        mdi_rows.append(m.gain_importance())
    return pd.DataFrame(mda_rows), pd.DataFrame(mdi_rows)


def stability(per_fold: pd.DataFrame) -> float:
    """Mean pairwise Spearman correlation of importance ranks across folds."""
    rows = per_fold.to_numpy()
    cors = [
        spearmanr(rows[i], rows[j]).statistic
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    ]
    return float(np.nanmean(cors)) if cors else float("nan")
