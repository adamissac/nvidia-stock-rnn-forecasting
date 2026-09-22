---
name: backtest-protocol
description: The rules every nvquant backtest, strategy, benchmark, and performance number must follow. Use this whenever you build or change a strategy, sizing rule, cost model, benchmark, metric, or statistical test, whenever you report a Sharpe or any performance number, and before touching anything related to the lockbox or the README results table.
---

# Backtest protocol

Short version. Details and formulas are in [references/protocol.md](references/protocol.md).

1. **Timing.** Decide after the close of session t with data through t. Fill at
   the open of t+1. The position earns log(open[t+2] / open[t+1]). Configurable
   in `backtest.execution`, but the default is the only one reported.
2. **Costs.** Half-spread + commission + vol-scaled slippage + square-root
   impact at the configured AUM, charged on |change in position|, plus borrow on
   shorts. Always report the 0 to 20 bps per side sensitivity.
3. **Benchmarks.** Every strategy table includes buy-and-hold NVDA, vol-targeted
   buy-and-hold NVDA, SMH, QQQ, the 200-day trend rule, and the random-signal
   null with matched turnover. The vol-targeted one is the key ablation.
4. **Two engines.** `nvquant.backtest.vectorized` and `nvquant.backtest.event`
   must agree to 1e-10 on every strategy. If you change one, run
   `uv run pytest tests/unit/test_backtest_engines.py`.
5. **Metrics and tests.** Net Sharpe with a stationary-bootstrap CI, PSR, DSR
   (trial count and Sharpe variance from the registry, per-period Sharpe), PBO
   via CSCV, SPA and White's RC against the benchmarks, MinTRL.
6. **Register every trial.** Every strategy configuration whose out-of-sample
   return series is computed is written to the registry
   (`nvquant.experiments.registry`). DSR's N is that count. Never delete trials.
7. **Lockbox.** 2025-01-01 onward. Only `make lockbox` touches it, once, after
   docs/PREREGISTRATION.md is committed. Development code loads data with
   `mode="dev"`, which truncates before the cutoff.
8. **Skepticism.** A net Sharpe above about 1.5 for a daily single-name strategy
   is a bug until proven otherwise. Check the leakage-guard checklist, beta
   exposure, and whether the PnL is all 2023 to 2024.

Where the quantitative-trading plugin's backtesting skills disagree with this, this wins.
