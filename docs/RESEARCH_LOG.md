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

## 2026-09-22: Phase 7, evaluation and risk

**Hypothesis.** If the best configuration has real skill, it should survive trial adjustment (DSR, PBO), beat the benchmarks under Hansen's SPA, show alpha that isn't just NVDA's own run, and work on the peers too.

**What ran.** Metrics with stationary-bootstrap CIs, PSR, DSR (N = 80 strategy trials, and N = 421 logged fits as a sensitivity check), MinTRL, PBO over 12,870 CSCV combinations, SPA and White's RC against buy and hold and vol-targeted buy and hold, a 1,000-path random null, factor attribution, four VaR methods with Kupiec and Christoffersen backtests, stress windows, regimes, calendar years, a block-bootstrap Monte Carlo, CPCV paths for ridge and LightGBM, feature importance, and the same pipeline on all 10 peers.

**Result.** From `reports/results.json`:
- The best configuration's DSR is 0.99 (0.95 counting every fit). That only says its Sharpe beats what luck across 80 trials would produce. It doesn't say it beats the benchmark: its Sharpe is 0.03 below vol-targeted buy and hold, and the SPA p-value against that benchmark is 0.153 (`headline`). PBO is 0.29.
- FF5 plus momentum alpha is 32.8% a year (t = 3.98), but vol-targeted buy and hold shows about the same (`attribution.bh_voltarget`), so the "alpha" is NVDA beating the factors in this sample, not the model.
- The random null (matched to the best configuration's exposure, which is long 97.8% of the time) is beaten by the best configuration on 98.0% of paths. That measures vol targeting against unscaled exposure, not forecasting skill.
- 28.8% of the summed daily PnL comes from 2023 to 2024 (`pnl_share_2023_2024`). The calendar-year table shows 2016 and 2021 contributed as much.
- Peers: with the vol-target sizing, ridge beat the peer's vol-targeted buy and hold on 1 of 10 peers, and LightGBM and the historical mean on 0 (`peers.n_beating_voltarget`).
- 99% VaR: historical and Gaussian VaR are rejected or borderline by Kupiec (exception rates 1.35% and 1.43%). Cornish-Fisher and GARCH VaR pass (p = 0.215 and 0.422).
- Feature importance is unstable: the mean rank correlation of MDA across purged folds is 0.13 (`importance.stability.mda`), which is what you'd expect when there isn't much signal to rank.
- The 2008 stress window has no out-of-sample coverage, because the first out-of-sample date is 2009-07-22.

**Decision.** The development-period conclusion is a null result. No forecast adds value beyond vol targeting after costs, and the method doesn't transfer to the peers. For the lockbox I'll preregister the best configuration, the best ML configuration, and the benchmarks, so 2025 can confirm or contradict this without any new selection.

## 2026-09-22: Phase 8, reporting and docs

**Hypothesis.** If every number in the README and docs is rendered from `reports/results.json`, and a check fails when a block drifts from a fresh render, then no document can quietly disagree with the artifacts.

**What ran.** `make report` builds `reports/results.json`, the static tear sheet, the figures, and the app data. It writes docs/RESULTS.md in full and fills the marked blocks in the README (`RESULTS`, `WHAT_DIDNT_WORK`), V1_POSTMORTEM (`V1`), and INTERVIEW_NOTES (`KEY_NUMBERS`, `RESUME`). `check_readme_block.py` re-renders every block and fails on any difference, and a docs-style test rejects em dashes and the banned hype words. The Streamlit app reads `reports/` only, and a headless AppTest renders it in the integration run.

**Result.** The check passes on the committed artifacts. Writing the v1 block turned up something RMSE hides: the replica's RMSE is dominated by a handful of blown-up forecasts, so the postmortem leads with median absolute error and the share of days worse than persistence (`v1_replica.median_abs_error_model`, `v1_replica.share_days_worse_than_persistence`).

**Decision.** Outside this research log (where every number names its source key), prose in the docs describes methods and doesn't quote result numbers. Those only appear inside generated blocks.

## 2026-09-22: Phase 9, review and hardening

**Hypothesis.** Independent reviews of the full diff will find mistakes I can't see, most likely in the statistics that decide what I claim.

**What ran.** Five reviews of `main...HEAD`: leakage-auditor, quant-reviewer, silent-failure-hunter, pr-test-analyzer, and a /code-review pass. I fixed every finding in a separate commit, added the tests they asked for, and reran the full pipeline (same data hash).

**Result.** The leakage audit passed, with four low-severity notes, all fixed. The quant review failed on the reporting, and it was right:
- The random null held a constant position, so it measured vol targeting, not timing. Now it's sized with the same vol-target path and costs. Random long/flat timing beats the best configuration on 78.4% of paths (`random_null.best_percentile` = 0.216), and the null's median Sharpe equals vol-targeted buy and hold's.
- The Fama-French regression mixed open-to-open strategy returns with close-to-close factors, which inflated alpha. Monthly, the best configuration's alpha is 24.1% a year (t = 2.41) against 25.3% (t = 2.49) for vol-targeted buy and hold (`attribution.*.FF5_MOM`). So it's NVDA's alpha, not the model's.
- 2023 and 2024 account for 82.5% of the compounded dollar gain (`headline.gain_share_2023_2024`), much more than the 28.8% share of summed daily returns suggested.
- With every strategy rescaled to the benchmark's volatility, SPA against vol-targeted buy and hold gives p = 0.991 (`headline.spa_vs_bh_voltarget_vol_matched`).

The other reviews found real bugs, all fixed:
- The peer study would crash if meta-labeling won.
- A NaN Sharpe could win best-strategy selection and turn every DSR into NaN.
- The GARCH VaR backtest was off by a day, and the rolling VaRs need a two-session lag.
- The conformal update used an outcome one step early.
- Missing cost inputs fell back to a full-sample median, which is lookahead.
- Stacking fell back to equal weights when NNLS rejected every member.
- The lockbox sentinel was written only after the evaluation.
- `make report` had written generated results into SPEC.md, because the spec quotes the marker strings. SPEC.md is restored, and generated blocks now go only to an explicit list of files.

**Corrections to earlier entries.** The Phase 7 entry quotes numbers from before these fixes:
- The FF5 alpha was 32.8% (t = 3.98) and is now 24.1% (t = 2.41). The comparison with vol-targeted buy and hold still holds, with new numbers (25.3%, t = 2.49).
- The null percentile was 98.0% and is now 21.6%, from a null that finally tests timing.
- The 99% VaR Kupiec p-values moved slightly after the lag fix: Cornish-Fisher 0.214 and GARCH 0.423 (they were 0.215 and 0.422), and historical is 0.044 (`var.0.99.backtests`). The pass and reject calls don't change.

For a while the every-fit trial count doubled, because the registry holds both full runs. Distinct configurations are now counted once, so it's back to 421, with a DSR of 0.95 (`meta.n_all_fits`, `headline.best_dsr_all_fits`). The conclusion doesn't change, and the corrected numbers make it stronger.

**Decision.** Preregister the lockbox: the best development configuration (the historical-mean baseline with vol targeting), the best ML configuration (LightGBM with vol targeting), plain ridge with vol targeting, and the five benchmarks.

## 2026-09-22: Phase 10, lockbox

**Hypothesis (preregistered in docs/PREREGISTRATION.md, commit 8f119f6).** No forecast adds value beyond vol targeting. Both learned configurations should land within their bootstrap uncertainty of vol-targeted buy and hold on 2025 onward.

**What ran.** `make lockbox`, once, on the preregistration commit. The feature store was rebuilt on the full history, the three preregistered configurations were refit walk-forward from the first session of 2025, and the five benchmarks were backtested with unchanged costs and timing. The sentinel shows one completed, unforced run (`reports/lockbox/SENTINEL.json`, `n_runs` = 1).

**Result.** 430 sessions, 2025-01-02 to 2026-09-21 (`lockbox`):
- Vol-targeted buy and hold: net Sharpe 0.82 [-0.56, 2.27].
- The historical-mean configuration matches it exactly, since it's long every day and sized the same way.
- LightGBM reaches 0.85 [-0.60, 2.22] and ridge 0.98 [-0.43, 2.49].
- Both learned configurations are well inside the benchmark's interval, as the preregistration expected, and SMH was the best benchmark (1.46).

After the run I noticed that the stress windows were dated by decision date, which misses moves at a window's first open. The 2025-01-27 DeepSeek gap is earned by the 2025-01-23 decision. I didn't re-run anything. The report re-dates the same saved returns by holding period and shows both tables. Re-dated, the DeepSeek window cost buy and hold 12.3% and the preregistered configurations 11.1% to 14.7%. The April tariff window cost buy and hold 7.9%, while LightGBM gained 6.4% (`lockbox_stress_holding_dated`). The development stress windows use the same holding-period dating now.

**Decision.** The conclusion stands on both the development period and the lockbox: on these features and models, nothing beats holding NVDA with vol targeting after costs. That's the result I report.

**Fresh-clone reproduction (same day).** `git clone`, `make setup`, and `make reproduce` in an empty directory completed with exit code 0, including a full re-download. Yahoo's adjusted prices came back with the same rows but differences up to 1.4e-6 relative (for example NVDA's close, largest on 2002-05-02), which changed 17 file hashes and the data hash (`ebe58cb94301` against the committed `e2fdd3b2b961`). The best configuration, the vol-targeted buy-and-hold Sharpe, the FF5 alpha, the null percentile, and the 2023 to 2024 gain share all matched to at least five significant figures. The two resampling statistics moved slightly: PBO 0.29 to 0.30, and SPA against vol-targeted buy and hold 0.153 to 0.158. The committed artifacts stay on the original data hash, and the conclusions are the same on both.
