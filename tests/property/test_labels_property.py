"""Labels carry their end time, and nothing after it can change them."""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from nvquant.config.schema import TripleBarrierConfig, UniverseConfig
from nvquant.data.synthetic import synthetic_market
from nvquant.labels.forward import forward_returns
from nvquant.labels.triple_barrier import triple_barrier

MARKET = synthetic_market(UniverseConfig(), n_sessions=300, start="2018-01-02", seed=12)
N = len(MARKET.sessions)


@settings(max_examples=50, deadline=None)
@given(i=st.integers(min_value=30, max_value=N - 30), seed=st.integers(0, 10_000))
def test_label_depends_only_on_data_through_t_end(i, seed):
    L = forward_returns(MARKET, [1, 5, 21], 20)
    tb = triple_barrier(MARKET, TripleBarrierConfig(vertical=5, vol_span=20))
    t = MARKET.sessions[i]
    for h in (1, 5, 21):
        end = L[f"t_end_{h}"].iloc[i]
        if pd.isna(end):
            continue
        assert end > t
        L2 = forward_returns(MARKET.perturb_after(end, seed=seed), [h], 20)
        assert L2[f"fwd_ret_{h}"].iloc[i] == L[f"fwd_ret_{h}"].iloc[i]
    end = tb["tb_t_end"].iloc[i]
    if not pd.isna(end):
        tb2 = triple_barrier(
            MARKET.perturb_after(end, seed=seed), TripleBarrierConfig(vertical=5, vol_span=20)
        )
        assert tb2["tb_label"].iloc[i] == tb["tb_label"].iloc[i]
