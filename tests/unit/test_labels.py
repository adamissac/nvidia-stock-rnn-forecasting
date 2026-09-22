import numpy as np
import pandas as pd
import pytest

from nvquant.config.schema import TripleBarrierConfig
from nvquant.labels.forward import end_times, forward_returns, holding_log_return
from nvquant.labels.triple_barrier import meta_labels, triple_barrier
from nvquant.labels.weights import average_uniqueness


def test_forward_returns_align_with_next_open(market):
    L = forward_returns(market, [1, 5], 20)
    o = market.ohlcv()["open"]
    i = 30
    assert L["fwd_ret_1"].iloc[i] == pytest.approx(np.log(o.iloc[i + 2] / o.iloc[i + 1]))
    assert L["t_end_1"].iloc[i] == market.sessions[i + 2]
    assert L["t_end_5"].iloc[i] == market.sessions[i + 6]
    assert L["fwd_ret_1"].iloc[-2:].isna().all()
    r = holding_log_return(market)
    assert r.iloc[i] == pytest.approx(L["fwd_ret_1"].iloc[i])
    assert holding_log_return(market, "next_close").iloc[i] != r.iloc[i]


def test_end_times():
    s = pd.bdate_range("2020-01-01", periods=5)
    e = end_times(s, 2)
    assert e.iloc[0] == s[2] and e.iloc[-2:].isna().all()


def test_triple_barrier_touch_order(market):
    tb = triple_barrier(market, TripleBarrierConfig(vertical=5, vol_span=20))
    ok = tb.dropna(subset=["tb_label"])
    assert set(ok["tb_label"].unique()) <= {-1.0, 0.0, 1.0}
    assert (ok["tb_t_end"] > ok.index).all()
    pos = market.sessions.get_indexer(ok["tb_t_end"])
    assert (pos - market.sessions.get_indexer(ok.index) <= 5 + 1).all()


def test_meta_labels():
    idx = pd.bdate_range("2020-01-01", periods=4)
    tb = pd.DataFrame({"tb_ret": [0.01, -0.02, 0.03, np.nan]}, index=idx)
    side = pd.Series([1.0, 1.0, 0.0, 1.0], index=idx)
    m = meta_labels(side, tb)
    assert m.iloc[0] == 1.0 and m.iloc[1] == 0.0 and np.isnan(m.iloc[2]) and np.isnan(m.iloc[3])


def test_average_uniqueness_hand_example():
    s = pd.bdate_range("2020-01-01", periods=6)
    # label A covers periods 1..2, B covers 2..3 (period = session a label's interval starts on)
    t_start = pd.Series([s[0], s[1]], index=[s[0], s[1]])
    t_end = pd.Series([s[3], s[4]], index=[s[0], s[1]])
    u = average_uniqueness(t_start, t_end, s)
    # A: periods {1: 1 label, 2: 2 labels} -> (1 + 0.5) / 2 = 0.75; same for B
    assert u.iloc[0] == pytest.approx(0.75) and u.iloc[1] == pytest.approx(0.75)
