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

**What ran.** `make data` downloaded 18 Yahoo tickers (with `multi_level_index=False` and `auto_adjust=True`), FRED DGS10 and DGS2 through the keyless CSV endpoint, Ken French FF5 and momentum daily files, and earnings dates for the 11 stocks, into a parquet cache with a SHA256 manifest. `make data-report` then checked every series against the NYSE calendar. Two fixes came out of the first run: Yahoo caps `get_earnings_dates` at `limit=100`, and Yahoo stamps unconfirmed future earnings at 15:00, so I classify an announcement as before the open only if it's before noon.

**Result.** From `reports/data_quality.json`: 0 manifest mismatches, and both NVDA split checks pass (`split_checks_passed: true`; the close ratio across 2021-07-20 is 0.991 and across 2024-06-10 is 1.007). The modeling start is 2006-07-17, set by ^VIX3M, the latest first-valid required input (`modeling_start`). The volume ratio across the 2021 split is 0.509, which is inside the tolerance but close to it. Unadjusted volume would show a ratio near 4, so the check still separates the two cases, but it's worth knowing the margin.

**Decision.** Keep ^VIX3M as a required input even though it moves the start to mid-2006. The consequence is that the 2008 crisis has no out-of-sample model coverage (the first OOS date lands in 2009), so that stress window is reported for benchmarks only.
