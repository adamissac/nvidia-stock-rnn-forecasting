"""Triple-barrier labels and meta-labels (Lopez de Prado 2018, ch. 3).

Entry is the open of t+1. The path is the sequence of later opens, so a touch
is detected at a price the strategy could actually trade. Barriers are
``+/- mult * sigma_t * sqrt(vertical)`` in log-return units, where ``sigma_t``
is the ex-ante daily vol at the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvquant.config.schema import TripleBarrierConfig
from nvquant.data.market import MarketData
from nvquant.labels.forward import ex_ante_vol


def triple_barrier(market: MarketData, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """Label every session with the first barrier its forward path touches.

    Returns
    -------
    DataFrame with columns
        ``tb_label`` in {-1, 0, 1} (lower, vertical with a zero return, upper; a
        vertical touch takes the sign of the return), ``tb_ret`` (log return at
        the touch), ``tb_t_end`` (touch session), ``tb_barrier`` (barrier width).
        Rows whose full vertical window isn't available are NaN.
    """
    px = market.ohlcv()
    sessions = market.sessions
    lo = np.log(px["open"]).to_numpy()
    sigma = ex_ante_vol(px["close"], cfg.vol_span).to_numpy()
    n = len(sessions)
    v = cfg.vertical
    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    t_end = np.full(n, -1, dtype=int)
    width = sigma * np.sqrt(v)
    for i in range(n):
        entry = i + 1
        if entry + v >= n or np.isnan(width[i]) or np.isnan(lo[entry]):
            continue
        path = lo[entry + 1 : entry + v + 1] - lo[entry]
        up = np.flatnonzero(path >= cfg.upper_mult * width[i])
        dn = np.flatnonzero(path <= -cfg.lower_mult * width[i])
        first_up = up[0] if len(up) else v
        first_dn = dn[0] if len(dn) else v
        if first_up < first_dn:
            k, label[i] = first_up, 1.0
        elif first_dn < first_up:
            k, label[i] = first_dn, -1.0
        else:
            k = v - 1
            label[i] = float(np.sign(path[k]))
        ret[i] = path[k]
        t_end[i] = entry + 1 + k
    ends = pd.Series(pd.NaT, index=sessions, dtype="datetime64[ns]")
    ok = t_end >= 0
    ends.iloc[np.flatnonzero(ok)] = sessions[t_end[ok]]
    return pd.DataFrame(
        {"tb_label": label, "tb_ret": ret, "tb_t_end": ends, "tb_barrier": width},
        index=sessions,
    )


def meta_labels(side: pd.Series, tb: pd.DataFrame) -> pd.Series:
    """1 if taking ``side`` (+1 long, -1 short) at t would have made money, else 0.

    NaN where the primary model is flat or the barrier outcome is unknown.
    """
    s = side.reindex(tb.index)
    out = (s * tb["tb_ret"] > 0).astype(float)
    return out.where((s != 0) & s.notna() & tb["tb_ret"].notna())
