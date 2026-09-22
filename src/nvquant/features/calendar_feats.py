"""Calendar features: earnings proximity, month end, options expiration.

These use scheduled information. The exchange calendar is published years
ahead. Earnings dates are announced roughly four weeks ahead, so
``days_to_earn`` is capped at ``earnings_cap`` (20 sessions) and the spec
declares ``known_ahead=20``: the causality test only perturbs earnings dates
more than 20 sessions past the cutoff. Yahoo's historical dates aren't
point-in-time (a date that moved would show where it ended up), which I list
as a limitation in docs/METHODOLOGY.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.data.calendar import nyse_sessions, third_fridays
from nvquant.data.market import MarketData
from nvquant.features.registry import FeatureParams, register

SINCE_CAP = 63


def sessions_to_next(sessions: pd.DatetimeIndex, events: pd.DatetimeIndex, cap: int) -> pd.Series:
    """Sessions from each date to the next event on or after it, capped at ``cap``."""
    pos_events = np.sort(sessions.get_indexer(events[events.isin(sessions)]))
    pos = np.arange(len(sessions))
    nxt = np.searchsorted(pos_events, pos, side="left")
    dist = np.full(len(sessions), float(cap))
    has = nxt < len(pos_events)
    dist[has] = np.minimum(pos_events[nxt[has]] - pos[has], cap)
    return pd.Series(dist, index=sessions)


def sessions_since_last(
    sessions: pd.DatetimeIndex, events: pd.DatetimeIndex, cap: int
) -> pd.Series:
    """Sessions since the most recent event strictly before each date, capped."""
    pos_events = np.sort(sessions.get_indexer(events[events.isin(sessions)]))
    pos = np.arange(len(sessions))
    prv = np.searchsorted(pos_events, pos, side="left") - 1
    dist = np.full(len(sessions), float(cap))
    has = prv >= 0
    dist[has] = np.minimum(pos[has] - pos_events[prv[has]], cap)
    return pd.Series(dist, index=sessions)


@register(
    "earnings",
    "calendar",
    lookback=0,
    known_ahead=20,
    description="Sessions to the next and since the last earnings announcement (capped).",
)
def earnings(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """``days_to_earn`` (cap 20), ``days_since_earn`` (cap 63), ``earn_next`` (1 if the
    announcement falls between tonight's close and tomorrow's open).
    """
    s = m.sessions
    ev = m.earnings.get(m.target, pd.DatetimeIndex([]))
    to_next = sessions_to_next(s, ev, p.earnings_cap)
    out = {
        "days_to_earn": to_next,
        "days_since_earn": sessions_since_last(s, ev, SINCE_CAP),
        "earn_next": (to_next == 0).astype(float),
    }
    return pd.DataFrame(out, index=s)


@register(
    "calendar",
    "calendar",
    lookback=0,
    description="Sessions to month end and options-expiration week flag.",
)
def calendar(m: MarketData, p: FeatureParams) -> pd.DataFrame:
    """``sessions_to_month_end`` (cap 5) and ``opex_week`` (1 in the week of the third Friday)."""
    s = m.sessions
    # Use the published exchange calendar, not the data's last row, so the value on the
    # final session of a truncated series is the same as it was in real time.
    cal = nyse_sessions(s.min(), s.max() + pd.Timedelta(days=45))
    month_ends = pd.DatetimeIndex(
        pd.Series(cal, index=cal).groupby(cal.to_period("M")).max().to_numpy()
    )
    pos_cal = cal.get_indexer(s)
    pos_end = np.searchsorted(cal.get_indexer(month_ends), pos_cal, side="left")
    to_end = pd.Series(
        np.minimum(cal.get_indexer(month_ends)[pos_end] - pos_cal, 5).astype(float), index=s
    )
    opex = third_fridays(cal)
    week = s.to_period("W-FRI")
    opex_weeks = set(opex.to_period("W-FRI"))
    out = {
        "sessions_to_month_end": to_end,
        "opex_week": pd.Series([float(w in opex_weeks) for w in week], index=s),
    }
    return pd.DataFrame(out, index=s)
