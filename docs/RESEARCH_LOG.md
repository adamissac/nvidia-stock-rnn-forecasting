# Research log

One dated entry per phase: Hypothesis, What ran, Result, Decision. Numbers come from artifacts under `reports/`.

## 2026-09-21: Phase 0, audit and plan

**Hypothesis.** The v1 notebook's headline result (RMSE 0.08) doesn't show that NVDA prices are forecastable, and the repo can't reproduce it.

**What ran.** I re-ran each audit check at commit a686217 on Python 3.12.14 with yfinance 1.7.0 and pandas 3.0.6 (commands are in docs/V1_AUDIT.md). I computed no performance numbers on 2025 dates, since those belong to the lockbox.

**Result.** All 8 findings from the spec reproduce, and I found 8 more. The most important is finding 4: the saved run trained on a newest-first CSV, so the model learned to predict backwards in time. The README metric isn't computed anywhere in the notebook.

**Decision.** Rebuild as a package with the controls listed at the end of docs/V1_AUDIT.md. The owner lifted the Phase 0 approval gate, so I recorded the plan decisions (long/flat headline, CPU only, $10M AUM, 1-session horizon with next-open fills) in docs/PLAN.md and went straight on to Phase 1.

## 2026-09-21: Phase 1, foundation

**Hypothesis.** A config-driven package with synthetic data that has known properties lets me test every later stage offline, including the claim "the models find signal when it exists and nothing when it doesn't".

**What ran.** I set up a uv project on Python 3.12 with a src layout, the typer CLI `nvquant`, pydantic configs (`base.yaml` plus `fast` and `full` profiles, unknown keys rejected), seeding and determinism helpers, synthetic generators (GBM, GARCH(1,1), two-state Markov switching, a planted-signal regression, and a full synthetic market with the real schema), the Makefile, pre-commit, and CI. The v1 files moved to `legacy/` with `git mv`. Two environment findings: this Mac has no Homebrew, so LightGBM couldn't load `libomp`. `scripts/macos_libomp.py` points it at the copy PyTorch ships instead, and `make setup` runs it. The installed stack is pandas 3.0 (copy-on-write, so `to_numpy()` can be read-only), arch 8, optuna 5, and torch 2.14. I checked each API I used against the installed package before writing against it.

**Result.** The synthetic tests pass. For example, the planted signal with strength 0.1 has a sample correlation between 0.07 and 0.13 over 20,000 rows, and GBM returns show no autocorrelation.

**Decision.** Keep the fast profile synthetic and under a minute, so CI can run the whole pipeline end to end on every push.

## 2026-09-21: Phase 2, data

**Hypothesis.** Everything downstream is only as honest as the data, so the data layer should fail loudly on schema problems, prove the split adjustment is right, and fix the modeling start by rule instead of by choice.

**What ran.** `make data` downloaded 19 Yahoo tickers (with `multi_level_index=False` and `auto_adjust=True`), FRED DGS10 and DGS2 through the keyless CSV endpoint, Ken French FF5 and momentum daily files, and earnings dates for the 11 stocks, into a parquet cache with a SHA256 manifest. `make data-report` then checked every series against the NYSE calendar. Two fixes came out of the first run: Yahoo caps `get_earnings_dates` at `limit=100`, and Yahoo stamps unconfirmed future earnings at 15:00, so I classify an announcement as before the open only if it's before noon.

**Result.** From `reports/data_quality.json`: 0 manifest mismatches, and both NVDA split checks pass (`split_checks_passed: true`; the close ratio across 2021-07-20 is 0.991 and across 2024-06-10 is 1.007). The modeling start is 2006-07-17, set by ^VIX3M, the latest first-valid required input (`modeling_start`). The volume ratio across the 2021 split is 0.509, which is inside the tolerance but close to it. Unadjusted volume would show a ratio near 4, so the check still separates the two cases, but it's worth knowing the margin.

**Decision.** Keep ^VIX3M as a required input even though it moves the start to mid-2006. The consequence is that the 2008 crisis has no out-of-sample model coverage (the first OOS date lands in 2009), so that stress window is reported for benchmarks only.

## 2026-09-21: Phase 3, features and labels

**Hypothesis.** If every feature is a pure, trailing function of past data and the test suite proves it by perturbing the future, then no later result can come from lookahead in the features.

**What ran.** 11 registered feature groups (returns, five realized-vol estimators and vol-of-vol, momentum and reversal, causal RSI/MACD/Bollinger/ATR, volume, rolling beta and correlation to SMH and QQQ, peer relative strength and dispersion, ETF returns, VIX and rate state, earnings and calendar), plus two fitted features computed walk-forward: fracdiff of log price with the smallest `d` that passes ADF on the training window, and the forward-filtered HMM high-vol probability. Labels: forward open-to-open log returns at 1, 5, and 21 sessions, vol-normalized versions, triple-barrier labels on the open path, meta-labels, and average-uniqueness weights. Each label has its end time. The hypothesis test perturbs everything after a random cutoff, 50 examples per feature group.

**Result.** The store has 57 columns over 4,647 sessions (`reports/features_info.json`: `n_features`, `n_rows`). Across the yearly refits, the chosen fracdiff `d` ranged from 0.1 to 0.4 (`fracdiff_d`), so the log price needs well under a full difference to pass ADF. The causality test passes for every group. While writing it I found two causality problems of my own. The month-end feature used the last row of the data as "the last session of the month", which is wrong at the end of a truncated series, so it now reads the published exchange calendar. And `days_to_earn` needs a declared `known_ahead` of 20 sessions, because the date isn't public earlier than that.

**Decision.** Fitted features are computed walk-forward in the store. The rows before their first refit are burn-in, and the harness starts out-of-sample dates strictly after the first refit, so no test row ever sees an in-sample fitted value.

## 2026-09-21: Phase 4, validation framework

**Hypothesis.** If every model goes through one walk-forward harness that purges on label end times, tunes only inside the training window, and logs every fit, then out-of-sample numbers mean what they say, and the trial count for the DSR is complete.

**What ran.** `WalkForward.blocks` (expanding or rolling, with a configurable retrain frequency; training rows need `t_end < refit date`), `PurgedKFold` with embargo, and `CombinatorialPurgedCV` with path assembly. The lockbox sentinel refuses a second run unless it gets `--force` and a written reason, and every forced rerun is appended. There's a JSONL registry, and Optuna tuning for LightGBM runs on purged folds of each training window. Property tests generate random label horizons and check that no training label interval overlaps a test span and that the embargo holds.

**Result.** The property tests pass. In the first full training run, the registry recorded 14 model runs and 320 tuning trials, which is 20 trials at each of 16 yearly retunes (`reports/registry/trials.jsonl`, kinds `model` and `tuning`). LightGBM's walk-forward has 185 monthly refits, the first on 2009-07-22 (its `fold_metrics`).

**Decision.** The DSR's trial count is the number of strategy configurations whose out-of-sample returns I computed (decision D8 in the plan). Tuning trials are logged but reported separately, since they never see out-of-sample returns. The report also shows a DSR that counts every logged fit, to show how much this choice matters.

## 2026-09-22: Phase 5, models

**Hypothesis.** If daily NVDA returns are forecastable from these features, at least one model (linear, boosted trees, or a small sequence model) should beat the naive baselines out of sample, and the gain should survive a Diebold-Mariano test and the model confidence set.

**What ran.** Walk-forward over 3,884 decision dates (2009-07-22 to 2024-12-31) for 14 return forecasters: zero, historical mean, AR(p), ridge, elastic net, LightGBM (nested Optuna), LSTM and GRU with Gaussian and quantile heads, TCN, PatchTST-lite, and equal-weight and stacked ensembles. Also six volatility models and the v1 replica. Wall times are in the registry: 34 to 52 s for each deep model except TCN at 344 s, 53 s for LightGBM with tuning, and 657 s for the v1 replica (`reports/registry/trials.jsonl`, `wall_time_s`). The whole development pipeline took about 21 minutes against the 3-hour budget.

**Result.** From `reports/results.json` (`forecasts`, `mcs_pvalues`, `vol_models`, `v1_replica`):
- The strongest IC is the AR baseline's (0.056, NW t = 3.35), then elastic net (0.036, t = 2.24) and ridge (0.033, t = 1.99). Out-of-sample R2 against the zero forecast is under half a percent for every model, and no model beats the zero forecast in a Diebold-Mariano test at 5%. The deep models all have negative R2 (-2.01% to -0.10%), and the model confidence set at 10% drops PatchTST, TCN, and both GRUs.
- The Gaussian heads' 90% intervals cover 90.6% to 91.9% of outcomes, but the PIT KS test rejects uniformity. The quantile heads' 80% intervals cover only about 70%, so they're too narrow.
- Volatility: LightGBM has the lowest QLIKE (0.331) but under-forecasts (MZ beta 1.38). HAR is the best calibrated (MZ beta 1.04). GJR-GARCH-t, the sizing model I picked before seeing any results, has QLIKE 0.491.
- The v1 replica is worse than persistence on 92.2% of days. Its forecast blew up (above 10 times the training maximum) on 10.6% of days, because once prices leave the MinMax range the ReLU cells diverge. The training loss was small at every refit, so this only shows up out of sample.
- The stacking weights jump between members from refit to refit (`reports/forecasts_stack_weights.parquet`), which is what you'd expect when the members' edges are this small.

**Decision.** Keep GJR-GARCH-t for sizing even though HAR and LightGBM score better. Switching after seeing the scores would be one more untracked selection step. Move on to strategies: a small IC can still matter as a trading signal, so Phase 6 tests that directly.

## 2026-09-22: Phase 6, strategies and backtest

**Hypothesis.** Even a small IC can pay after costs if the sizing is right. The test that matters is whether any forecast beats holding NVDA with the same volatility targeting, since vol targeting alone is known to raise the Sharpe of a volatile stock.

**What ran.** 80 strategy configurations (13 forecasts times 6 sizing rules, plus meta-labeling raw and vol-targeted) and 5 benchmarks over 3,884 decision dates, with next-open fills, the full cost model at $10M, both engines, a 0 to 20 bps cost sweep, and a capacity sweep up to $30B. Every configuration is registered as a strategy trial (`reports/registry/trials.jsonl`, kind `strategy`: 80 rows).

**Result.** From `reports/results.json`:
- The engines agree to 1.4e-17 (`meta.engine_max_abs_diff`).
- The best configuration is the historical-mean baseline with vol targeting (net Sharpe 1.18, `headline.best_sharpe`), below vol-targeted buy and hold (1.21) and the 200-day trend rule (1.24). The best strategy on an ML forecast is LightGBM with vol targeting at 1.14.
- The regime filter lowered the Sharpe for all 13 vol-targeted strategies that used it. Meta-labeling on the trend rule reached 0.45, against 1.24 for the trend rule it was meant to improve.
- Costs aren't what decides the ranking. The best configuration goes from 1.19 at 0 bps to 1.16 at 20 bps per side (`cost_sweep`), and square-root impact barely matters until $10B (`capacity`), because its turnover is low.

**Decision.** Nothing so far beats the key ablation. I'll run the full statistics (DSR, PBO, SPA, attribution, peers) before calling it, but no parameter will be changed to chase the benchmark.
