"""NYSE session calendar helpers (exchange_calendars XNYS)."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=4)
def _calendar(start: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start=start)


def nyse_sessions(start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
    """NYSE sessions between ``start`` and ``end`` inclusive, tz-naive, no freq."""
    cal = _calendar("1990-01-02")
    start_ts = max(pd.Timestamp(start), cal.first_session)
    end_ts = min(pd.Timestamp(end), cal.last_session)
    sessions = cal.sessions_in_range(start_ts, end_ts)
    return pd.DatetimeIndex(sessions.tz_localize(None) if sessions.tz else sessions, freq=None)


def third_fridays(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly options expiration dates (third Friday, or the prior session if closed)."""
    months = pd.period_range(sessions.min(), sessions.max(), freq="M")
    out = []
    for m in months:
        first = m.start_time
        offset = (4 - first.weekday()) % 7
        third = first + pd.Timedelta(days=offset + 14)
        prior = sessions[sessions <= third]
        if len(prior) and prior[-1].to_period("M") == m:
            out.append(prior[-1])
    return pd.DatetimeIndex(out)
