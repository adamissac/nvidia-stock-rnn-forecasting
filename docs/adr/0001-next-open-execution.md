# ADR 0001: Decide at the close, fill at the next open

**Status:** accepted (2026-09-21)

**Context.** A daily strategy has to say when it knows what, and when it trades. Trading at the close of the same bar it just observed (same-bar execution) is the most common way a backtest gets a return it could never have earned.

**Decision.** Decisions use data through the close of session t. Orders fill at the open of t+1, and the position earns log(O[t+2] / O[t+1]). The 1-session label is defined as exactly that return, so the forecast target, the label, and the PnL are the same quantity. `next_close` exists as a config option for a sensitivity check only.

**Consequences.** The overnight gap into t+1 is lost to the strategy, which is conservative for a stock that moves a lot overnight on earnings. Labels resolve two sessions after the decision, so purging uses `t_end = t + 2` for the 1-session label.
