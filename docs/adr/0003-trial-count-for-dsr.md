# ADR 0003: What counts as a trial in the Deflated Sharpe Ratio

**Status:** accepted (2026-09-21)

**Context.** The DSR corrects the best Sharpe for how many configurations were tried. Too small an N makes the best result look significant. Inner tuning trials are fits too, but they're scored on training folds, not on the out-of-sample returns I pick from.

**Decision.** N is the number of strategy configurations whose out-of-sample returns the backtest computed (every forecast times every sizing rule, plus meta-labeling), and the variance term is their per-period Sharpe variance. I also report the DSR with N equal to every logged fit (models, tuning trials, strategies, vol models, v1), as a sensitivity check.

**Consequences.** Adding a sizing rule or a model increases N automatically, because the registry counts it.
