# Lockbox preregistration

Written and committed before the lockbox (2025-01-01 onward) is opened. `make lockbox` refuses to run unless this file and `configs/preregistration.yaml` are committed and unchanged, and it runs exactly once.

## What gets evaluated

The code, configs, and data pipeline at the commit that adds this file. Nothing may change between that commit and the lockbox run.

| Configuration | Why it's included |
|---|---|
| `hist_mean` + voltarget | The best configuration on the development period. It's a naive baseline, so including it tests whether the development ranking holds. |
| `lgbm` + voltarget | The best configuration built on a machine-learning forecast. |
| `ridge` + voltarget | The plainest learned model with the same sizing, to show whether a linear signal carries over. |
| Buy and hold NVDA, vol-targeted buy and hold NVDA, buy and hold SMH, buy and hold QQQ, 200-day trend rule | Benchmarks. Vol-targeted buy and hold is the key comparison. |

The procedure is the development procedure, extended in time. The feature store is rebuilt on the full history (every feature is causal, so development-period values don't change). Each model is refit walk-forward on every labeled sample before each refit date, with the first out-of-sample decision on the first session of 2025. The volatility forecast used for sizing is GJR-GARCH-t, as in development. Costs, sizing, and execution timing are unchanged.

## Metrics to report

For every configuration and benchmark, over the lockbox dates:

1. Net Sharpe with a 95% stationary-bootstrap confidence interval
2. Total return and maximum drawdown
3. PSR against zero
4. The net Sharpe difference from vol-targeted buy and hold

Plus the two 2025 stress windows, reported from this run only: the DeepSeek selloff (2025-01-24 to 2025-02-07) and the April 2025 tariff shock (2025-04-02 to 2025-04-21), with each configuration's return and maximum drawdown inside the window.

## What I expect, written down before looking

The development result is that no forecast adds value beyond vol targeting. I expect the lockbox to agree: both learned configurations should land within their bootstrap uncertainty of vol-targeted buy and hold, with no systematic edge. The lockbox covers less than two years, so its confidence intervals will be wide. A single lockbox Sharpe above the benchmark, inside those intervals, wouldn't overturn the development conclusion, and I'll say so. A lockbox that clearly contradicts development (a learned configuration beating vol-targeted buy and hold by more than its interval width) would be reported as the headline and investigated as a possible bug before being believed.

## What I won't do

- Change any configuration, parameter, or cost after seeing lockbox numbers.
- Run the lockbox a second time. A forced rerun needs `--force --reason`, is recorded in `reports/lockbox/SENTINEL.json`, and would be disclosed in the README.
- Report 2025 stress numbers from anywhere other than this run.
