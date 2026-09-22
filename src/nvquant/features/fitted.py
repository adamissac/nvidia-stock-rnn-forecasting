"""Features whose parameters are estimated on data (fracdiff ``d``, HMM regimes).

They are computed walk-forward: at each refit session r the parameters are fit
on data up to and including r, and those parameters produce the values for the
sessions after r, up to and including the next refit. So every value at t uses
parameters fit strictly before t and observations up to t.

The rows before the first refit get values from the first fit (a burn-in).
Those rows are only ever training rows: the harness checks that the first
out-of-sample date is after every fitted feature's first refit.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nvquant.config.schema import FracdiffConfig, RegimeConfig
from nvquant.data.market import MarketData
from nvquant.features.fracdiff import frac_diff, min_d_passing_adf
from nvquant.models.regime import HMMParams, fit_hmm, forward_filter, regime_observations


class FittedFeature(abc.ABC):
    """A feature with parameters fit on a training window."""

    name: str

    @abc.abstractmethod
    def fit(
        self, market: MarketData, train_start: pd.Timestamp, train_end: pd.Timestamp
    ) -> dict[str, float]:
        """Fit on rows in ``[train_start, train_end]``; return a loggable summary."""

    @abc.abstractmethod
    def transform(self, market: MarketData) -> pd.DataFrame:
        """Values for every session, computed causally with the fitted parameters."""


@dataclass
class FracDiffFeature(FittedFeature):
    """Fractionally differentiated log close with the minimum ``d`` passing ADF."""

    cfg: FracdiffConfig
    name: str = "fracdiff"
    d: float | None = None
    pvalues: dict[float, float] = field(default_factory=dict)

    def fit(
        self, market: MarketData, train_start: pd.Timestamp, train_end: pd.Timestamp
    ) -> dict[str, float]:
        """Choose ``d`` on training rows only."""
        logp = np.log(market.ohlcv()["close"]).loc[train_start:train_end].dropna()
        self.d, self.pvalues = min_d_passing_adf(
            logp,
            self.cfg.d_grid,
            self.cfg.adf_pvalue,
            self.cfg.weight_threshold,
            self.cfg.max_width,
        )
        return {"d": self.d}

    def transform(self, market: MarketData) -> pd.DataFrame:
        """Fixed-width fracdiff of the log close (causal)."""
        if self.d is None:
            raise RuntimeError("fit before transform")
        logp = np.log(market.ohlcv()["close"])
        fd = frac_diff(logp, self.d, self.cfg.weight_threshold, self.cfg.max_width)
        return pd.DataFrame({"fracdiff_logp": fd}, index=market.sessions)


@dataclass
class RegimeFeature(FittedFeature):
    """Forward-filtered probability of the high-volatility HMM state."""

    cfg: RegimeConfig
    seed: int = 0
    name: str = "regime"
    params: HMMParams | None = None

    def _obs(self, market: MarketData) -> pd.DataFrame:
        px = market.ohlcv()
        return regime_observations(px["close"], px["high"], px["low"])

    def fit(
        self, market: MarketData, train_start: pd.Timestamp, train_end: pd.Timestamp
    ) -> dict[str, float]:
        """Baum-Welch on training rows only."""
        obs = self._obs(market).loc[train_start:train_end].dropna()
        self.params = fit_hmm(obs.to_numpy(), self.cfg.n_states, self.cfg.n_iter, self.seed)
        return {"p_stay_high": float(self.params.transmat[-1, -1])}

    def transform(self, market: MarketData) -> pd.DataFrame:
        """Filtered ``P(high-vol state | x_1..x_t)``."""
        if self.params is None:
            raise RuntimeError("fit before transform")
        obs = self._obs(market)
        probs = forward_filter(obs.to_numpy(), self.params)
        return pd.DataFrame({"regime_p_high": probs[:, -1]}, index=market.sessions)


@dataclass(frozen=True)
class FitRecord:
    """One refit of a fitted feature."""

    feature: str
    refit_date: str
    train_start: str
    summary: dict[str, float]


def refit_dates(
    sessions: pd.DatetimeIndex, start: pd.Timestamp, min_train: int, every: int
) -> pd.DatetimeIndex:
    """Refit sessions: the first after ``min_train`` sessions from ``start``, then every ``every``."""
    usable = sessions[sessions >= start]
    if len(usable) <= min_train:
        raise ValueError("not enough sessions for the first fit")
    return pd.DatetimeIndex(usable[min_train - 1 :: every])


def walk_forward_fitted(
    market: MarketData,
    feature: FittedFeature,
    start: pd.Timestamp,
    min_train: int,
    every: int,
) -> tuple[pd.DataFrame, list[FitRecord]]:
    """Walk-forward values of a fitted feature (expanding training window from ``start``).

    Returns the feature frame and one :class:`FitRecord` per refit.
    """
    sessions = market.sessions
    dates = refit_dates(sessions, start, min_train, every)
    blocks = []
    records = []
    for i, r in enumerate(dates):
        summary = feature.fit(market.truncate(r), start, r)
        records.append(FitRecord(feature.name, str(r.date()), str(start.date()), summary))
        nxt = dates[i + 1] if i + 1 < len(dates) else sessions[-1]
        lo = sessions[0] if i == 0 else r + pd.Timedelta(days=1)
        values = feature.transform(market.truncate(nxt))
        blocks.append(values.loc[lo:nxt])
    out = pd.concat(blocks)
    return out.reindex(sessions), records
