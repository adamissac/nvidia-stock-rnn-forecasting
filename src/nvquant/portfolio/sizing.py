"""Turn forecasts into positions (weights of capital, decided at the close of t).

Every rule is causal: scales and vols come from trailing windows or ex-ante
forecasts known at t. Long/flat rules clip at 0; long/short rules allow
negative weights. Gross exposure is capped at ``max_gross``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.config.schema import SizingConfig, StrategyConfig

ANN = 252.0


def _floor(sc: SizingConfig, max_gross: float) -> float:
    return -max_gross if sc.long_short else 0.0


def sign_position(forecast: pd.Series, sc: SizingConfig) -> pd.Series:
    """+1 above the threshold, -1 below minus the threshold (long/short only), else 0."""
    pos = (forecast > sc.threshold).astype(float)
    if sc.long_short:
        pos = pos - (forecast < -sc.threshold).astype(float)
    return pos.where(forecast.notna(), 0.0)


def scaled_position(forecast: pd.Series, sc: SizingConfig, st: StrategyConfig) -> pd.Series:
    """Forecast divided by twice its trailing std (so a 2-sigma forecast is a full position), clipped."""
    scale = forecast.rolling(st.scale_window, min_periods=min(63, st.scale_window)).std()
    pos = forecast / (2 * scale)
    return pos.clip(_floor(sc, st.max_gross), st.max_gross).fillna(0.0)


def vol_target_position(
    direction: pd.Series, var_forecast: pd.Series, sc: SizingConfig, st: StrategyConfig
) -> pd.Series:
    """Direction times ``target_vol / forecast_vol`` (annualized), capped at ``max_gross``."""
    vol = np.sqrt(var_forecast.reindex(direction.index) * ANN)
    lev = (st.target_vol / vol).clip(upper=st.max_gross)
    return (direction * lev).clip(_floor(sc, st.max_gross), st.max_gross).fillna(0.0)


def kelly_position(
    forecast: pd.Series, var_forecast: pd.Series, sc: SizingConfig, st: StrategyConfig
) -> pd.Series:
    """Fractional Kelly: ``f * mu / sigma^2`` with the return forecast and the variance forecast, capped."""
    raw = sc.kelly_fraction * forecast / var_forecast.reindex(forecast.index)
    return raw.clip(_floor(sc, st.max_gross), st.max_gross).fillna(0.0)


def apply_regime_filter(
    position: pd.Series, p_high: pd.Series, threshold: float = 0.5
) -> pd.Series:
    """Go flat when the filtered probability of the high-vol regime is above ``threshold``."""
    return position.where(p_high.reindex(position.index).fillna(0.0) <= threshold, 0.0)


def size(
    forecast: pd.Series,
    var_forecast: pd.Series,
    p_high: pd.Series,
    sc: SizingConfig,
    st: StrategyConfig,
) -> pd.Series:
    """Apply one sizing rule (and the optional regime filter) to a return forecast."""
    if sc.rule == "sign":
        pos = sign_position(forecast, sc)
    elif sc.rule == "scaled":
        pos = scaled_position(forecast, sc, st)
    elif sc.rule == "voltarget":
        pos = vol_target_position(sign_position(forecast, sc), var_forecast, sc, st)
    elif sc.rule == "kelly":
        pos = kelly_position(forecast, var_forecast, sc, st)
    else:
        raise ValueError(f"unknown sizing rule {sc.rule}")
    if sc.regime_filter:
        pos = apply_regime_filter(pos, p_high)
    return pos.rename(sc.name)
