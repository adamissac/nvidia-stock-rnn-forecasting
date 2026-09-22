import numpy as np
import pandas as pd

from nvquant.config.schema import FeaturesConfig, TripleBarrierConfig
from nvquant.features.registry import build_features
from nvquant.labels.triple_barrier import triple_barrier
from nvquant.labels.weights import average_uniqueness
from nvquant.portfolio.meta_labeling import bet_size, meta_label_positions, trend_side


def test_meta_label_positions(market):
    feats = build_features(market, FeaturesConfig()).fillna(0.0)
    side = trend_side(market.ohlcv()["close"], 50)
    tb = triple_barrier(market, TripleBarrierConfig(vertical=5, vol_span=20))
    uniq = average_uniqueness(pd.Series(tb.index, index=tb.index), tb["tb_t_end"], market.sessions)
    dates = market.sessions[400:]
    pos, prob = meta_label_positions(feats, side, tb, uniq, dates, 50, seed=0)
    assert ((pos >= 0) & (pos <= 1)).all()
    assert (pos[side.reindex(dates) == 0] == 0).all()  # never trades against a flat primary
    # outcomes that resolve after a block's refit date can't change that block's probabilities
    first = dates[50]
    tb2 = tb.copy()
    late = (tb2["tb_t_end"] >= first).to_numpy()
    tb2.loc[late, "tb_ret"] = -tb2.loc[late, "tb_ret"]
    _, prob2 = meta_label_positions(feats, side, tb2, uniq, dates, 50, seed=0)
    block = dates[(dates >= first) & (dates < dates[100])]
    np.testing.assert_allclose(prob.loc[block].to_numpy(), prob2.loc[block].to_numpy())


def test_bet_size_is_symmetric_and_monotone():
    p = np.linspace(0.5, 0.99, 50)
    b = bet_size(p)
    assert b[0] == 0 and np.all(np.diff(b) > 0) and b[-1] < 1
    assert (bet_size(1 - p[1:]) == 0).all()
