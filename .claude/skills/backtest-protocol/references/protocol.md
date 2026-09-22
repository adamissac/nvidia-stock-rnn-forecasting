# Backtest protocol details

## Execution and returns

- Decision time: after close of session t. Inputs: anything with index <= t
  (exogenous inputs at <= t-1 through their lag).
- Fill: open of t+1. Holding-period return for the position chosen at t:
  `r_oo[t] = log(open[t+2] / open[t+1])`. That is the same quantity as the
  1-day label, so forecasts, labels, and PnL line up by construction.
- Gross PnL at t: `pos[t] * (exp(r_oo[t]) - 1)` (simple return on capital).
- Costs at t: `cost_rate[t] * |pos[t] - pos[t-1]|`, charged at the fill.

## Cost model (per unit of traded notional)

```
cost_rate = half_spread + commission + slip_k * sigma_daily
            + impact_k * sigma_daily * sqrt(|trade_notional| / ADV_dollar)
borrow    = borrow_bps_annual / 252 * max(-pos, 0)   (per day held short)
```

`ADV_dollar` is the trailing 20-session average dollar volume, lagged one
session. Defaults are in `configs/base.yaml` under `costs`. The capacity study
sweeps AUM from 1e6 to 1e10 and reports where net Sharpe halves.

## Benchmarks

| name | definition |
|------|-----------|
| bh_nvda | position 1 always |
| bh_nvda_voltarget | position = min(target_vol / garch_vol[t], max_gross) |
| bh_smh, bh_qqq | buy-and-hold the ETF (open-to-open, same timing) |
| trend_200 | long when close[t] > SMA200[t], else flat |
| random_null | 1,000 random long/flat signals with the strategy's turnover; report the strategy's percentile |

## Statistics

- Sharpe: per-period mean / std of daily net simple returns; annualized with sqrt(252) only for display.
- PSR(SR*) = Phi(((SR - SR*) sqrt(T - 1)) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)), with SR per period, g3 skew, g4 kurtosis (not excess).
- DSR: PSR with SR* = sqrt(V[SR_n]) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))), gamma = Euler-Mascheroni.
- MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (z_alpha / (SR - SR*))^2.
- PBO: CSCV with S = 16 blocks, logits of the out-of-sample rank of the in-sample winner.
- SPA and White RC: `arch.bootstrap.SPA` on losses = -returns, stationary bootstrap, 1,000 reps.
- Bootstrap CIs: stationary bootstrap with `arch.bootstrap.optimal_block_length`.

## Trial registry

Registry file: `reports/registry/trials.jsonl`, one JSON object per trial with
`trial_id, kind (model|strategy|tuning|lockbox), name, config, config_hash,
git_sha, data_hash, fold_metrics, wall_time_s, created_at`. DSR uses the
strategy-kind trials of the development run.

## Lockbox

- Dates >= `lockbox.start` (2025-01-01) are the lockbox.
- `nvquant lockbox` writes `reports/lockbox/SENTINEL.json` and refuses to run
  again without `--force --reason "..."`, which is appended to the sentinel.
- The 2025 stress windows (2025-01-27 DeepSeek selloff, April 2025 tariff shock)
  are reported from the lockbox run only.
