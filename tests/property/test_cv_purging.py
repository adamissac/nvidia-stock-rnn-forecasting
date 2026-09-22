"""No training label interval overlaps a test fold, and the embargo holds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from nvquant.cv.splits import CombinatorialPurgedCV, PurgedKFold, WalkForward

S = pd.bdate_range("2015-01-01", periods=700)


def _t_end(n: int, max_h: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    h = rng.integers(1, max_h + 1, n)
    return pd.Series(S[np.arange(n) + h], index=S[:n])


def _overlap(ts, te, a, b) -> bool:
    return (ts <= b) and (te >= a)


@settings(max_examples=40, deadline=None)
@given(
    k=st.integers(2, 8),
    embargo=st.integers(0, 10),
    max_h=st.integers(1, 20),
    seed=st.integers(0, 1000),
)
def test_purged_kfold_no_overlap_and_embargo(k, embargo, max_h, seed):
    t_end = _t_end(600, max_h, seed)
    pos_s = S.get_indexer(t_end.index)
    pos_e = S.get_indexer(t_end)
    for f in PurgedKFold(k, embargo).split(t_end, S):
        a, b = pos_s[f.test].min(), max(pos_e[f.test].max(), pos_s[f.test].max())
        for i in f.train:
            assert not _overlap(pos_s[i], pos_e[i], a, b)
            assert not (b < pos_s[i] <= b + embargo)


@settings(max_examples=30, deadline=None)
@given(
    groups=st.integers(3, 7),
    embargo=st.integers(0, 5),
    max_h=st.integers(1, 10),
    seed=st.integers(0, 1000),
)
def test_cpcv_no_overlap(groups, embargo, max_h, seed):
    t_end = _t_end(420, max_h, seed)
    pos_s = S.get_indexer(t_end.index)
    pos_e = S.get_indexer(t_end)
    cv = CombinatorialPurgedCV(groups, 2, embargo)
    g_idx = cv.groups(len(t_end))
    for sp in cv.split(t_end, S):
        for g in sp.test_groups:
            idx = g_idx[g]
            a, b = pos_s[idx[0]], max(pos_e[idx].max(), pos_s[idx[-1]])
            for i in sp.fold.train:
                assert not _overlap(pos_s[i], pos_e[i], a, b)
                assert not (b < pos_s[i] <= b + embargo)


@settings(max_examples=40, deadline=None)
@given(every=st.integers(1, 60), max_h=st.integers(1, 25), seed=st.integers(0, 1000))
def test_walk_forward_training_labels_resolve_before_test(every, max_h, seed):
    t_end = _t_end(600, max_h, seed)
    pred = t_end.index[200:]
    for blk in WalkForward(every).blocks(pred, t_end):
        assert (t_end.loc[blk.train] < blk.test.min()).all()
