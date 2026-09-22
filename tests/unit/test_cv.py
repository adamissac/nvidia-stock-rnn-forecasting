import numpy as np
import pandas as pd
import pytest

from nvquant.cv.lockbox import LockboxError, check_can_open, record_open, sentinel_path
from nvquant.cv.splits import CombinatorialPurgedCV, PurgedKFold, WalkForward, purge_mask


def _labels(n: int, h: int):
    s = pd.bdate_range("2020-01-01", periods=n + h + 2)
    idx = s[:n]
    t_end = pd.Series(s[np.arange(n) + 1 + h], index=idx)
    return s, t_end


def test_walk_forward_blocks_purge_and_cover():
    _, t_end = _labels(300, 5)
    pred = t_end.index[100:]
    blocks = WalkForward(20).blocks(pred, t_end)
    covered = pd.DatetimeIndex(np.concatenate([b.test for b in blocks]))
    assert covered.equals(pred)
    for b in blocks:
        assert (t_end.loc[b.train] < b.refit_date).all()
        assert b.train.max() < b.test.min()
    rolling = WalkForward(20, "rolling", 50).blocks(pred, t_end)
    assert all(len(b.train) <= 50 for b in rolling)
    with pytest.raises(ValueError):
        WalkForward(0)
    with pytest.raises(ValueError):
        WalkForward(5, "weird")


def test_purge_mask_embargo():
    start = np.arange(20)
    end = start + 3
    keep = purge_mask(start, end, 8, 11, embargo=2)
    assert not keep[5:12].any() and not keep[12:14].any() and keep[14:].all() and keep[:5].all()


def test_purged_kfold():
    s, t_end = _labels(200, 5)
    folds = PurgedKFold(4, embargo=3).split(t_end, s)
    assert len(folds) == 4
    assert len(PurgedKFold(4).sklearn_splits(t_end, s)) == 4
    with pytest.raises(ValueError):
        PurgedKFold(1)


def test_cpcv_counts_and_paths():
    s, t_end = _labels(240, 2)
    cv = CombinatorialPurgedCV(6, 2, embargo=2)
    splits = cv.split(t_end, s)
    assert len(splits) == 15 and cv.n_paths == 5
    paths = cv.paths()
    assert len(paths) == 5 and all(len(p) == 6 for p in paths)
    for p in paths:  # every path uses each group once, from a split that tests that group
        for g, sid in p:
            assert g in splits[sid].test_groups
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(3, 3)


def test_splits_reject_non_session_dates():
    s = pd.bdate_range("2020-01-01", periods=10)
    t_end = pd.Series(s[2:], index=pd.DatetimeIndex(["2019-01-01", *list(s[1:8])]))
    with pytest.raises(ValueError):
        PurgedKFold(2).split(t_end, s)


def test_lockbox_guard(tmp_path):
    check_can_open(tmp_path, force=False, reason=None)
    record_open(tmp_path, "abc", "dh", "ch", None)
    with pytest.raises(LockboxError):
        check_can_open(tmp_path, force=False, reason=None)
    with pytest.raises(LockboxError):
        check_can_open(tmp_path, force=True, reason=" ")
    check_can_open(tmp_path, force=True, reason="data vendor fixed a bad print")
    record_open(tmp_path, "abd", "dh", "ch", "data vendor fixed a bad print")
    import json

    payload = json.loads(sentinel_path(tmp_path).read_text())
    assert payload["n_runs"] == 2 and payload["runs"][1]["forced"]
