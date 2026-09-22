"""Synthetic data with known properties.

Two kinds of generators live here:

- small generators (``gbm``, ``garch``, ``regime_switching``, ``planted_signal``)
  for unit tests that need a known answer, for example "a model recovers a
  planted signal" or "GARCH recovers its own parameters";
- ``synthetic_market``, a full :class:`MarketData` with the same schema as the
  real download, so the fast profile and CI can run the whole pipeline offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nvquant.config.schema import UniverseConfig
from nvquant.data.calendar import nyse_sessions
from nvquant.data.market import MarketData

TRADING_DAYS = 252


def gbm(
    n: int, mu: float = 0.08, sigma: float = 0.3, s0: float = 100.0, seed: int = 0
) -> pd.Series:
    """Geometric Brownian motion prices on a business-day index.

    Log returns are iid normal, so no feature can predict them. Used to check
    that models find no signal when there is none.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(s0 * np.exp(np.cumsum(r)), index=idx, name="price")


@dataclass(frozen=True)
class GarchPath:
    """Returns and true conditional variance of a simulated GARCH(1,1)."""

    returns: pd.Series
    variance: pd.Series


def garch(
    n: int,
    omega: float = 0.05,
    alpha: float = 0.08,
    beta: float = 0.9,
    mu: float = 0.0,
    seed: int = 0,
) -> GarchPath:
    """Simulate GARCH(1,1) returns in percent with Gaussian innovations."""
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for a stationary GARCH")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    ret[0] = mu + np.sqrt(var[0]) * z[0]
    for t in range(1, n):
        var[t] = omega + alpha * (ret[t - 1] - mu) ** 2 + beta * var[t - 1]
        ret[t] = mu + np.sqrt(var[t]) * z[t]
    idx = pd.bdate_range("2000-01-03", periods=n)
    return GarchPath(pd.Series(ret, index=idx), pd.Series(var, index=idx))


@dataclass(frozen=True)
class RegimePath:
    """Returns and the hidden state sequence of a Markov regime-switching model."""

    returns: pd.Series
    states: pd.Series


def regime_switching(
    n: int,
    transition: tuple[tuple[float, float], tuple[float, float]] = ((0.98, 0.02), (0.05, 0.95)),
    means: tuple[float, float] = (0.05, -0.1),
    stds: tuple[float, float] = (0.8, 2.5),
    seed: int = 0,
) -> RegimePath:
    """Two-state Gaussian Markov-switching returns (percent)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(transition)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=p[states[t - 1]])
    mu = np.asarray(means)[states]
    sd = np.asarray(stds)[states]
    idx = pd.bdate_range("2000-01-03", periods=n)
    return RegimePath(
        pd.Series(mu + sd * rng.standard_normal(n), index=idx), pd.Series(states, index=idx)
    )


def planted_signal(
    n: int, n_features: int = 10, strength: float = 0.1, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Tabular data where ``y = strength * x0 + noise`` and the rest is noise.

    With ``strength=0.1`` and unit-variance noise the population IC is about 0.1,
    roughly the size of a real (and optimistic) daily return signal.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    x = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        index=idx,
        columns=[f"x{i}" for i in range(n_features)],
    )
    y = pd.Series(strength * x["x0"].to_numpy() + rng.standard_normal(n), index=idx, name="y")
    return x, y


def _garch_vol(rng: np.random.Generator, n: int, ann_vol: float) -> np.ndarray:
    """Daily vol path from a GARCH(1,1)-like recursion with a given long-run level."""
    daily = ann_vol / np.sqrt(TRADING_DAYS)
    alpha, beta = 0.08, 0.9
    omega = daily**2 * (1 - alpha - beta)
    var = np.empty(n)
    var[0] = daily**2
    z = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = omega + alpha * var[t - 1] * z[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(var)


def _ohlcv_from_returns(
    rng: np.random.Generator,
    overnight: np.ndarray,
    intraday: np.ndarray,
    s0: float,
    base_volume: float,
) -> pd.DataFrame:
    n = len(overnight)
    opens = np.empty(n)
    closes = np.empty(n)
    prev_close = s0
    for t in range(n):
        opens[t] = prev_close * np.exp(overnight[t])
        closes[t] = opens[t] * np.exp(intraday[t])
        prev_close = closes[t]
    spread = np.abs(intraday) + np.abs(rng.normal(0, 0.006, n))
    high = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0, 0.5, n)) * spread)
    low = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0, 0.5, n)) * spread)
    move = np.abs(overnight + intraday)
    volume = base_volume * np.exp(rng.normal(0, 0.3, n)) * (1 + 20 * move)
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes, "volume": volume.round()}
    )


def synthetic_market(
    universe: UniverseConfig,
    n_sessions: int = 1500,
    start: str | pd.Timestamp = "2015-01-02",
    signal_strength: float = 0.0,
    seed: int = 0,
) -> MarketData:
    """A full synthetic market with the same schema as the real data.

    Structure: a GARCH market factor drives QQQ and SPY, a semiconductor factor
    (market beta 1.2 plus its own noise) drives SMH, SOXX, ^SOX and the stocks,
    and VIX tracks the market factor's conditional vol. If ``signal_strength``
    is non-zero, the target's intraday return on session t+1 gets a drift of
    ``signal_strength * sigma * s_t``, where ``s_t = -tanh(z-scored 5-day return
    at t)``. That drift shows up in the open(t+1) to open(t+2) return, which is
    exactly the 1-session label, so a causal short-term reversal feature can
    recover it.
    """
    rng = np.random.default_rng(seed)
    sessions = nyse_sessions(start, pd.Timestamp(start) + pd.Timedelta(days=int(n_sessions * 1.6)))
    sessions = sessions[:n_sessions]
    n = len(sessions)

    m_vol = _garch_vol(rng, n, 0.18)
    market = m_vol * rng.standard_normal(n) + 0.0003
    semis = 1.2 * market + _garch_vol(rng, n, 0.12) * rng.standard_normal(n)

    prices: dict[str, pd.DataFrame] = {}

    def split(
        total: np.ndarray, extra_intraday: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        overnight = 0.4 * total + rng.normal(0, 0.002, n)
        intraday = total - overnight
        if extra_intraday is not None:
            intraday = intraday + extra_intraday
        return overnight, intraday

    stocks = [universe.target, *universe.peers]
    for i, ticker in enumerate(stocks):
        beta = 1.0 + 0.15 * (i % 4)
        idio = _garch_vol(rng, n, 0.25 + 0.03 * (i % 5)) * rng.standard_normal(n)
        total = beta * semis + idio + 0.0002
        if ticker == universe.target and signal_strength != 0.0:
            vol = np.sqrt(np.convolve(total**2, np.ones(63) / 63, mode="full")[:n]) + 1e-4
            closes_log = np.cumsum(total)
            planted = np.zeros(n)
            for t in range(6, n - 1):
                ret5 = closes_log[t] - closes_log[t - 5]
                z = ret5 / (vol[t] * np.sqrt(5))
                planted[t + 1] = -signal_strength * vol[t] * np.tanh(z)
            overnight, intraday = split(total, planted)
        else:
            overnight, intraday = split(total)
        prices[ticker] = _ohlcv_from_returns(rng, overnight, intraday, 50.0 + 10 * i, 2e7)

    etf_loadings = {"SMH": (0.0, 1.0), "SOXX": (0.0, 1.02), "QQQ": (1.0, 0.1), "SPY": (0.8, 0.0)}
    for ticker in universe.etfs:
        lm, ls = etf_loadings.get(ticker, (1.0, 0.0))
        total = lm * market + ls * semis + rng.normal(0, 0.002, n)
        prices[ticker] = _ohlcv_from_returns(rng, *split(total), 100.0, 5e7)

    for ticker in universe.indices:
        if ticker == "^SOX":
            prices[ticker] = _ohlcv_from_returns(rng, *split(semis), 1000.0, 0.0)
            continue
        if ticker in ("^VIX", "^VIX3M"):
            smooth = 1.0 if ticker == "^VIX" else 0.6
            level = 100 * np.sqrt(TRADING_DAYS) * m_vol
            level = smooth * level + (1 - smooth) * level.mean() + rng.normal(0, 0.5, n)
            level = np.clip(level, 9.0, None)
        else:  # ^TNX, a yield in percent
            level = 2.5 + np.cumsum(rng.normal(0, 0.04, n))
            level = np.clip(level, 0.3, None)
        noise = np.abs(rng.normal(0, 0.01, n))
        prices[ticker] = pd.DataFrame(
            {
                "open": level * (1 + rng.normal(0, 0.005, n)),
                "high": level * (1 + noise),
                "low": level * (1 - noise),
                "close": level,
                "volume": np.zeros(n),
            }
        )

    for df in prices.values():
        df.index = sessions
        df.index.name = "date"

    dgs10 = 2.5 + np.cumsum(rng.normal(0, 0.04, n))
    rates = pd.DataFrame(
        {"DGS10": dgs10, "DGS2": dgs10 - 0.5 + np.cumsum(rng.normal(0, 0.02, n))},
        index=sessions,
    )
    rf = np.full(n, 0.0001)
    spy_ret = np.log(prices["SPY"]["close"]).diff().fillna(0.0).to_numpy()
    factors = pd.DataFrame(
        {
            "mkt_rf": spy_ret - rf,
            "smb": rng.normal(0, 0.005, n),
            "hml": rng.normal(0, 0.005, n),
            "rmw": rng.normal(0, 0.003, n),
            "cma": rng.normal(0, 0.003, n),
            "mom": rng.normal(0, 0.007, n),
            "rf": rf,
        },
        index=sessions,
    )
    earnings = {t: pd.DatetimeIndex(sessions[20 + (i % 10) :: 63]) for i, t in enumerate(stocks)}
    return MarketData(
        prices=prices, rates=rates, factors=factors, earnings=earnings, target=universe.target
    )
