"""The in-memory market data container shared by every pipeline stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketData:
    """Aligned daily data on one NYSE session index.

    Attributes
    ----------
    prices : Mapping[str, DataFrame]
        Ticker to OHLCV frame (lowercase columns). Every frame shares ``sessions``;
        rows before a ticker's first trade are NaN.
    rates : DataFrame
        Treasury yields in percent (``DGS10``, ``DGS2``), aligned to sessions.
    factors : DataFrame
        Fama-French daily factors as decimals (``mkt_rf, smb, hml, rmw, cma, mom, rf``).
    earnings : Mapping[str, DatetimeIndex]
        Earnings announcement sessions per ticker.
    target : str
        The traded ticker.
    """

    prices: Mapping[str, pd.DataFrame]
    rates: pd.DataFrame
    factors: pd.DataFrame
    earnings: Mapping[str, pd.DatetimeIndex] = field(default_factory=dict)
    target: str = "NVDA"

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """The shared session index."""
        return pd.DatetimeIndex(self.prices[self.target].index)

    @property
    def tickers(self) -> list[str]:
        """All tickers in ``prices``."""
        return list(self.prices)

    def ohlcv(self, ticker: str | None = None) -> pd.DataFrame:
        """OHLCV frame for ``ticker`` (the target by default)."""
        return self.prices[ticker or self.target]

    def field(self, name: str, tickers: list[str] | None = None) -> pd.DataFrame:
        """One OHLCV field across tickers, as a sessions x tickers frame."""
        cols = tickers or self.tickers
        return pd.DataFrame({t: self.prices[t][name] for t in cols if t in self.prices})

    def truncate(self, end: pd.Timestamp | str) -> MarketData:
        """Return a copy with every series cut at ``end`` (inclusive)."""
        end = pd.Timestamp(end)
        return replace(
            self,
            prices={t: df.loc[:end].copy() for t, df in self.prices.items()},
            rates=self.rates.loc[:end].copy(),
            factors=self.factors.loc[:end].copy(),
            earnings={t: idx[idx <= end] for t, idx in self.earnings.items()},
        )

    def slice_from(self, start: pd.Timestamp | str) -> MarketData:
        """Return a copy starting at ``start`` (inclusive)."""
        start = pd.Timestamp(start)
        return replace(
            self,
            prices={t: df.loc[start:].copy() for t, df in self.prices.items()},
            rates=self.rates.loc[start:].copy(),
            factors=self.factors.loc[start:].copy(),
        )

    def with_target(self, target: str) -> MarketData:
        """Same data with a different traded ticker (used by the peer study)."""
        if target not in self.prices:
            raise KeyError(f"{target} not in market data")
        return replace(self, target=target)

    def perturb_after(self, cutoff: pd.Timestamp, seed: int, scale: float = 0.5) -> MarketData:
        """Randomly perturb every value strictly after ``cutoff``.

        Used by the causality property tests: a causal function's output at or
        before ``cutoff`` must not change.
        """
        rng = np.random.default_rng(seed)

        def _bump(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            mask = out.index > cutoff
            n = int(mask.sum())
            if n == 0 or out.shape[1] == 0:
                return out
            noise = np.exp(rng.normal(0.0, scale, size=(n, out.shape[1])))
            out.loc[mask] = out.loc[mask].to_numpy() * noise
            return out

        earnings = {}
        for t, idx in self.earnings.items():
            keep = idx[idx <= cutoff]
            after = self.sessions[self.sessions > cutoff]
            extra = (
                pd.DatetimeIndex(rng.choice(after, size=min(3, len(after)), replace=False))
                if len(after)
                else pd.DatetimeIndex([])
            )
            earnings[t] = keep.union(extra)
        return replace(
            self,
            prices={t: _bump(df) for t, df in self.prices.items()},
            rates=_bump(self.rates),
            factors=_bump(self.factors),
            earnings=earnings,
        )
