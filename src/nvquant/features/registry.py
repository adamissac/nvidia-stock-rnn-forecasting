"""Feature registry: every feature group with its lookback, lag, and docs.

A feature function takes :class:`MarketData` and :class:`FeatureParams` and
returns a frame on ``market.sessions`` computed only from rows at or before each
date. The registry then applies the spec's ``lag`` (one session for exogenous
series by default). ``tests/property/test_feature_causality.py`` checks every
registered spec by perturbing the future and asserting the past is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nvquant.config.schema import FeaturesConfig
from nvquant.data.market import MarketData


@dataclass(frozen=True)
class FeatureParams:
    """Parameters every feature function may read (from ``cfg.features``)."""

    return_horizons: tuple[int, ...] = (1, 5, 21, 63)
    vol_windows: tuple[int, ...] = (21, 63)
    beta_window: int = 63
    zscore_window: int = 63
    earnings_cap: int = 20

    @classmethod
    def from_config(cls, cfg: FeaturesConfig) -> FeatureParams:
        """Build from the features section of the config."""
        return cls(
            return_horizons=tuple(cfg.return_horizons),
            vol_windows=tuple(cfg.vol_windows),
            beta_window=cfg.beta_window,
            zscore_window=cfg.zscore_window,
            earnings_cap=cfg.earnings_cap,
        )


FeatureFn = Callable[[MarketData, FeatureParams], pd.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one feature group.

    Attributes
    ----------
    name : str
        Group name (columns are prefixed or listed in the function docstring).
    group : str
        Documentation group (returns, volatility, technical, ...).
    func : FeatureFn
        Pure function of market data.
    lookback : int
        Sessions of history the longest column needs (warmup).
    exogenous : bool
        True if it reads anything besides the target's own OHLCV. Exogenous
        specs get ``cfg.features.exogenous_lag`` sessions of lag.
    known_ahead : int
        Sessions ahead the inputs are public (scheduled events such as earnings).
        The property test only perturbs data more than this far past the cutoff.
    description : str
        One line for the generated docs.
    """

    name: str
    group: str
    func: FeatureFn
    lookback: int
    exogenous: bool = False
    known_ahead: int = 0
    description: str = ""
    columns: tuple[str, ...] = field(default_factory=tuple)

    def lag(self, exogenous_lag: int) -> int:
        """Lag applied to this spec's output."""
        return exogenous_lag if self.exogenous else 0

    def compute(
        self, market: MarketData, params: FeatureParams, exogenous_lag: int = 1
    ) -> pd.DataFrame:
        """Run the function and apply the lag."""
        out = self.func(market, params)
        if not out.index.equals(market.sessions):
            raise ValueError(f"feature {self.name} must return rows on market.sessions")
        lag = self.lag(exogenous_lag)
        return out.shift(lag) if lag else out


_REGISTRY: dict[str, FeatureSpec] = {}


def register(
    name: str,
    group: str,
    lookback: int,
    exogenous: bool = False,
    known_ahead: int = 0,
    description: str = "",
) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator that adds a feature function to the registry."""

    def deco(func: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate feature {name}")
        _REGISTRY[name] = FeatureSpec(
            name=name,
            group=group,
            func=func,
            lookback=lookback,
            exogenous=exogenous,
            known_ahead=known_ahead,
            description=description or (func.__doc__ or "").strip().splitlines()[0],
        )
        return func

    return deco


def all_specs() -> list[FeatureSpec]:
    """Every registered feature spec (importing the feature modules registers them)."""
    from nvquant.features import calendar_feats, cross_asset, price  # noqa: F401

    return list(_REGISTRY.values())


def build_features(
    market: MarketData, cfg: FeaturesConfig, specs: list[FeatureSpec] | None = None
) -> pd.DataFrame:
    """Compute every (non-excluded) registered feature into one frame."""
    params = FeatureParams.from_config(cfg)
    frames = []
    for spec in specs or all_specs():
        if spec.name in cfg.exclude:
            continue
        frames.append(spec.compute(market, params, cfg.exogenous_lag))
    out = pd.concat(frames, axis=1)
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature columns: {dupes}")
    return out.astype("float64")


def column_metadata(market: MarketData, cfg: FeaturesConfig) -> pd.DataFrame:
    """One row per output column: spec, group, lookback, lag, description."""
    params = FeatureParams.from_config(cfg)
    rows = []
    for spec in all_specs():
        if spec.name in cfg.exclude:
            continue
        cols = spec.func(market, params).columns
        for c in cols:
            rows.append(
                {
                    "column": c,
                    "spec": spec.name,
                    "group": spec.group,
                    "lookback": spec.lookback,
                    "lag": spec.lag(cfg.exogenous_lag),
                    "known_ahead": spec.known_ahead,
                    "description": spec.description,
                }
            )
    return pd.DataFrame(rows)
