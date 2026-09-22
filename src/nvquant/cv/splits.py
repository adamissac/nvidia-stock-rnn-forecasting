"""Time-series splitters with purging and embargo (Lopez de Prado 2018, ch. 7 and 12).

Every sample i is an interval ``[t_i, t_end_i]``: the decision date and the
session at which its label is fully known. A training sample leaks into a test
fold if its interval overlaps the fold's span, so it is purged. After a test
fold, an embargo drops training samples that start within ``embargo`` sessions
of the fold's end, because their features overlap the test period's labels
through serial correlation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One split: positional indices into the sample index."""

    train: np.ndarray
    test: np.ndarray
    fold_id: int = 0


def _as_positions(t_end: pd.Series, sessions: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Start and end positions on the session calendar for each sample."""
    start = sessions.get_indexer(pd.DatetimeIndex(t_end.index))
    if (start < 0).any():
        raise ValueError("every sample date must be a session")
    end = np.full(len(t_end), np.iinfo(np.int64).max // 2)
    ok = t_end.notna().to_numpy()
    end[ok] = sessions.get_indexer(pd.DatetimeIndex(t_end[ok]))
    return start, end


def purge_mask(
    start: np.ndarray,
    end: np.ndarray,
    test_start: int,
    test_end: int,
    embargo: int,
) -> np.ndarray:
    """Boolean mask of samples allowed in training given one test span.

    A sample is dropped if its label interval overlaps ``[test_start, test_end]``
    (positions on the session calendar), or if it starts after the test span but
    within ``embargo`` sessions of its end.
    """
    overlaps = (start <= test_end) & (end >= test_start)
    embargoed = (start > test_end) & (start <= test_end + embargo)
    return ~(overlaps | embargoed)


@dataclass(frozen=True)
class Block:
    """One walk-forward refit: fit on ``train`` (decision dates), predict ``test``."""

    refit_date: pd.Timestamp
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


class WalkForward:
    """Expanding or rolling walk-forward with a fixed retrain frequency.

    Refits happen every ``retrain_every`` prediction dates starting at the first
    one. At refit date r the training set is every labeled sample with
    ``t_end < r``: its label was fully known before the first decision in the
    block. With ``scheme="rolling"`` only the last ``rolling_window`` of those
    are kept. The test block is every prediction date from r up to the next
    refit, including recent dates whose labels aren't known yet (a live system
    still has to hold a position on them).
    """

    def __init__(
        self, retrain_every: int, scheme: str = "expanding", rolling_window: int | None = None
    ) -> None:
        if scheme not in ("expanding", "rolling"):
            raise ValueError("scheme must be expanding or rolling")
        if retrain_every < 1:
            raise ValueError("retrain_every must be >= 1")
        self.retrain_every = retrain_every
        self.scheme = scheme
        self.rolling_window = rolling_window

    def blocks(self, pred_dates: pd.DatetimeIndex, t_end: pd.Series) -> list[Block]:
        """Walk-forward blocks.

        Parameters
        ----------
        pred_dates : DatetimeIndex
            Every out-of-sample decision date, ascending.
        t_end : Series
            Label end time for every labeled sample, indexed by decision date.
        """
        refits = pred_dates[:: self.retrain_every]
        sample_dates = pd.DatetimeIndex(t_end.index)
        ends = pd.DatetimeIndex(t_end.to_numpy())
        out = []
        for k, r in enumerate(refits):
            nxt = refits[k + 1] if k + 1 < len(refits) else None
            test = pred_dates[(pred_dates >= r) & ((pred_dates < nxt) if nxt is not None else True)]
            train = sample_dates[(sample_dates < r) & (ends < r)]
            if self.scheme == "rolling" and self.rolling_window:
                train = train[-self.rolling_window :]
            if len(train):
                out.append(Block(pd.Timestamp(r), train, test))
        return out


class PurgedKFold:
    """K contiguous test folds; training excludes overlapping and embargoed samples."""

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, t_end: pd.Series, sessions: pd.DatetimeIndex) -> list[Fold]:
        """Purged k-fold splits."""
        start, end = _as_positions(t_end, sessions)
        n = len(t_end)
        bounds = np.array_split(np.arange(n), self.n_splits)
        folds = []
        for fid, test in enumerate(bounds):
            ts = start[test[0]]
            te = max(int(end[test].max()), int(start[test[-1]]))
            keep = purge_mask(start, end, ts, te, self.embargo)
            keep[test] = False
            folds.append(Fold(np.flatnonzero(keep), test, fid))
        return folds

    def sklearn_splits(
        self, t_end: pd.Series, sessions: pd.DatetimeIndex
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The same folds as ``(train, test)`` tuples for sklearn's ``cv=`` argument."""
        return [(f.train, f.test) for f in self.split(t_end, sessions)]


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial split: which groups are in test."""

    fold: Fold
    test_groups: tuple[int, ...]


class CombinatorialPurgedCV:
    """Combinatorial purged CV (Lopez de Prado 2018, ch. 12).

    The sample is cut into ``n_groups`` contiguous groups. Every choice of
    ``n_test`` groups is a test set (``C(N, k)`` splits); training uses the other
    groups after purging and embargo around each test group. Each group is in
    ``C(N-1, k-1)`` test sets, which can be assembled into
    ``phi = k/N * C(N, k)`` complete backtest paths.
    """

    def __init__(self, n_groups: int = 6, n_test: int = 2, embargo: int = 0) -> None:
        if not 0 < n_test < n_groups:
            raise ValueError("need 0 < n_test < n_groups")
        self.n_groups = n_groups
        self.n_test = n_test
        self.embargo = embargo

    @property
    def n_paths(self) -> int:
        """Number of backtest paths ``phi``."""
        from math import comb

        return self.n_test * comb(self.n_groups, self.n_test) // self.n_groups

    def groups(self, n: int) -> list[np.ndarray]:
        """Positional indices of each group."""
        return np.array_split(np.arange(n), self.n_groups)

    def split(self, t_end: pd.Series, sessions: pd.DatetimeIndex) -> list[CPCVSplit]:
        """All ``C(N, k)`` purged splits."""
        start, end = _as_positions(t_end, sessions)
        groups = self.groups(len(t_end))
        out = []
        for fid, combo in enumerate(itertools.combinations(range(self.n_groups), self.n_test)):
            keep = np.ones(len(t_end), dtype=bool)
            test = np.concatenate([groups[g] for g in combo])
            for g in combo:
                idx = groups[g]
                ts = start[idx[0]]
                te = max(int(end[idx].max()), start[idx[-1]])
                keep &= purge_mask(start, end, ts, te, self.embargo)
            keep[test] = False
            out.append(CPCVSplit(Fold(np.flatnonzero(keep), test, fid), combo))
        return out

    def paths(self) -> list[list[tuple[int, int]]]:
        """Assign (group, split id) pairs to paths.

        Path j uses, for each group g, the j-th split (in combination order)
        that has g in its test set. Returns ``n_paths`` lists of ``(group, split_id)``.
        """
        combos = list(itertools.combinations(range(self.n_groups), self.n_test))
        per_group: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for sid, combo in enumerate(combos):
            for g in combo:
                per_group[g].append(sid)
        return [[(g, per_group[g][j]) for g in range(self.n_groups)] for j in range(self.n_paths)]
