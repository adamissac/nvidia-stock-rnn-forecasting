# v2 plan

Source of truth for scope: [SPEC.md](SPEC.md). This file turns the spec's target design into phases with checkboxes and acceptance criteria. A box gets checked only when its acceptance criteria were verified by a command whose output I looked at.

## Decisions

The owner asked me to run every phase without approval gates (see the SPEC addenda), so I made these calls myself. Each one can be changed in `configs/base.yaml` or by a follow-up ADR.

| # | Question | Decision | Why |
|---|----------|----------|-----|
| D1 | Long/flat or long/short? | **Long/flat is the headline.** Long/short runs as registered variants with a 30 bps/yr borrow cost. | NVDA had a strong positive drift in the sample, so shorting mostly measures that drift. Long/flat is the honest comparison to buy-and-hold. The variants still count as trials in DSR. |
| D2 | GPU? | **CPU only.** CUDA and MPS are detected and logged, but `device: cpu` is the default. | The machine is an Apple M5 Pro (15 cores, 24 GB). The models are tiny, CPU runs are deterministic, and CI has no GPU. |
| D3 | Rename the repo? | **No.** The repo keeps its URL and the package is `nvquant`. | Existing links keep working. The README title changes instead. |
| D4 | Assumed AUM for impact | **$10M** base, with a capacity sweep from $1M to $10B. | This is small next to NVDA's dollar volume, and the sweep shows where impact starts to matter. |
| D5 | Forecast horizon and timing | **1 session.** Decide after the close of t, fill at the open of t+1, hold to the open of t+2. The 5- and 21-day labels are used for IC decay and triple-barrier labels. | Labels, forecasts, and PnL then all measure the same open-to-open return. |
| D6 | Lockbox | **2025-01-01 to the latest date.** Development runs load data truncated at 2024-12-31. | This is what the spec asks for. |
| D7 | Retrain cadence | Tabular and linear models monthly (21 sessions), deep models every 126 sessions, HMM and fracdiff `d` every 252 sessions, GARCH every 21 sessions. | This keeps the full run near the 3-hour budget (see below). |
| D8 | DSR trial count | N = the number of distinct strategy configurations whose out-of-sample returns I computed in the development run. Inner Optuna trials are logged but not counted, because they never see development OOS returns. I also report DSR with N including every logged fit. | Selection bias comes from choosing among OOS results. The second number shows how sensitive DSR is to this choice. |
| D9 | Exogenous lag | Every series other than the target's own OHLCV is lagged one session by default, including peers, ETFs, VIX, rates, and factors. | This is conservative. FRED H.15 yields really do publish a day late, and using the same lag everywhere keeps the rule simple to check. |
| D10 | Ship target | PR from `quant-overhaul` into `main`, then merged with `gh pr merge` (SPEC addendum 2). | The owner asked for the work on main. A merge keeps the "no direct push to main" rule intact. |
| D11 | Deep model framework | Plain PyTorch, including the v1 replica (with a custom ReLU LSTM cell to match Keras `activation='relu'`). | This avoids pulling in TensorFlow for one model ("no unused frameworks"). |
| D12 | SHAP | LightGBM's native TreeSHAP (`pred_contrib=True`) instead of the `shap` package. | It computes the same exact values with one less dependency. |

## Compute budget (CPU laptop, full profile)

These are estimates. After Phase 7, the RESEARCH_LOG compares them with measured wall times from the registry.

| Step | Estimate | Main lever if over budget |
|------|----------|---------------------------|
| data download + cache | 3 min | cached after the first run |
| data-quality report, features, labels | 2 min | none needed |
| baselines, linear, LightGBM (monthly refits, 12 years OOS) | 10 min | refit every 63 sessions |
| Optuna nested tuning for LightGBM (20 trials per yearly retune) | 10 min | 10 trials |
| deep models: LSTM, GRU (Gaussian and quantile heads), TCN, PatchTST, 3 seeds, refit every 126 sessions | 60 min | 2 seeds, then drop PatchTST |
| v1 replica (yearly refit on 252-day windows) | 5 min | none needed |
| volatility models (GARCH family refit monthly, HAR, LightGBM vol) | 8 min | refit every 63 sessions |
| HMM (yearly refit) + forward filter | 2 min | none needed |
| ensembles, stacking | 2 min | none needed |
| backtests (all strategy configs, both engines, cost sweep, capacity) | 5 min | none needed |
| evaluation: bootstrap CIs, SPA/RC, MCS, PBO, random null (1,000 paths) | 15 min | 500 reps |
| peer study (finalist configs on 10 peers) | 35 min | drop deep models from peers |
| explainability (MDA under purged CV, clustered MDA) | 10 min | fewer permutations |
| reporting | 2 min | none needed |
| **total** | **~2.8 h** | |

Trade-offs made to fit the budget:
- The peer study runs the **pre-registered finalist configurations** unchanged on every peer, not every exploratory model. This is still the identical pipeline for anything I actually claim.
- Deep models refit twice a year with 3 seeds instead of monthly.
- The fast profile (CI) uses synthetic data, 2 years of it, a tiny deep model, and 50 bootstrap reps. Its target is under 5 minutes.

## Risks

| Risk | Mitigation |
|------|-----------|
| yfinance or Yahoo changes format or rate-limits | parquet cache, retries with backoff, pandera schemas that fail loudly, tests never use the network |
| ^VIX3M or AVGO history starts late and pushes the modeling start past 2008 | the start date is the latest first-valid date across required inputs, as the spec says. Stress windows without OOS model coverage are reported for benchmarks only, and I say so. |
| pandas 3.0 breaks a dependency | pin the pandas range in pyproject, run the full test suite in CI |
| deep models take longer than budgeted | the levers in the table above, cut in the order the spec allows |
| a result looks too good | the leakage-auditor and quant-reviewer gates, and the random-signal null |
| Ken French data lags a month or two | used only for attribution, which runs on the overlap |
| earnings dates aren't point-in-time | `days_to_earnings` capped at 20 sessions (about when the date is announced), caveat documented |

## Phases

### Phase 0: audit, tooling, plan
- [x] Branch `quant-overhaul` from an up-to-date `main`; docs/SPEC.md saved verbatim
- [x] docs/V1_AUDIT.md: the 8 spec findings verified at a686217, plus 8 more
- [x] CLAUDE.md, docs/RESEARCH_LOG.md
- [x] Skills: leakage-guard (with scripts/leakage_lint.py), backtest-protocol, add-model, refresh-results
- [x] Agents: leakage-auditor, quant-reviewer, experiment-runner
- [x] Hooks and permissions merged into .claude/settings.json (plugins and marketplaces kept)
- [x] This plan

Acceptance: the files exist, `python3 .claude/skills/leakage-guard/scripts/leakage_lint.py` runs, and the hook exits 0 on a non-Python payload. The branch is pushed.

### Phase 1: foundation
Files: `pyproject.toml`, `uv.lock`, `src/nvquant/{__init__,cli,logging_utils}.py`, `src/nvquant/config/{__init__,schema}.py`, `src/nvquant/experiments/repro.py`, `src/nvquant/data/{market,calendar,synthetic}.py`, `configs/{base,fast,full}.yaml`, `Makefile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `scripts/macos_libomp.py`, `legacy/` (git mv of the notebook, download script, and requirements), `tests/unit/test_{config,synthetic,repro,leakage_lint}.py`.
- [x] uv project (Python 3.12), src layout, typer CLI `nvquant`, CPU torch wheels on Linux CI
- [x] pydantic config with base plus profile merge, and a config hash
- [x] Seeding for python, numpy, and torch, deterministic torch, device detection, git SHA, file hashing
- [x] Synthetic generators: GBM, GARCH(1,1), 2-state regime switching, planted weak signal, and a full synthetic `MarketData` for the fast profile
- [x] Makefile targets: setup, data, data-report, features, train, backtest, evaluate, report, reproduce, lockbox, test, lint, typecheck, ci, app (PROFILE=fast|full)
- [x] CI: uv sync, ruff, mypy, pytest with coverage >= 85%, end-to-end fast-profile smoke run. `make ci` runs the same steps and passes locally (149 tests, 97% coverage). The workflow file itself is parked at `.github/ci-workflow-pending.yml`: the push token lacks GitHub's `workflow` scope, so it can't be written into `.github/workflows/`. Moving it there is one command, listed in that file.
- [x] pre-commit (ruff, ruff-format, end-of-file, leakage lint)
- [x] legacy/ with README, v1 files moved with `git mv`, v1 notebook unmodified

Acceptance: `make ci` passes locally and GitHub Actions is green on the branch.

### Phase 2: data
Files: `src/nvquant/data/{sources,schemas,manifest,loader,quality}.py` (the universe lives in `config/schema.py`; `sources.py` holds the Yahoo, FRED, Ken French, and earnings downloaders), `reports/data_quality.{json,md}`, `tests/unit/test_data.py`.
- [x] yfinance download with `multi_level_index=False`, `auto_adjust=True`, retries with backoff, parquet cache
- [x] FRED DGS10 and DGS2 from the keyless CSV endpoint
- [x] Ken French FF5 plus momentum daily zips
- [x] NVDA earnings dates with the point-in-time caveat
- [x] NYSE session alignment (exchange_calendars), pandera schemas
- [x] Manifest with source, download time, and SHA256 per file; data hash
- [x] Data-quality report: missing sessions, stale prices, outliers, split checks on 2021-07-20 and 2024-06-10 for prices and volume
- [x] Modeling start date computed and documented; dev loader truncates at the lockbox cutoff

Acceptance: `make data data-report` works from an empty cache, the split checks pass, and the loader tests (offline, on fixtures) pass.

### Phase 3: features and labels
Files: `src/nvquant/features/{registry,price,cross_asset,calendar_feats,fracdiff,fitted,store,docs}.py`, `src/nvquant/models/regime.py` (the HMM behind the regime feature), `src/nvquant/labels/{forward,triple_barrier,weights}.py` (meta-labels live in `triple_barrier.py`), `docs/FEATURES.md` (generated), `tests/property/test_{feature_causality,labels_property}.py`, `tests/unit/test_{features,labels}.py`.
- [x] All spec feature groups with lookback and lag metadata
- [x] Fracdiff (fixed width) with the minimum `d` passing ADF, chosen on training data only
- [x] Labels: forward 1/5/21 log returns, vol-normalized, triple barrier, meta-labels, uniqueness weights; every label has `t_end`
- [x] Causality property test over every registered feature
- [x] Feature store (parquet) and generated feature docs

Acceptance: the property tests pass with at least 50 hypothesis examples per feature, and `make features` writes the store and docs/FEATURES.md.

### Phase 4: validation framework
Files: `src/nvquant/cv/{splits,lockbox}.py` (purging lives in `splits.py`), `src/nvquant/experiments/{registry,tuning,harness}.py`, `tests/property/test_cv_purging.py`, `tests/unit/test_{cv,registry}.py`.
- [x] Expanding and rolling walk-forward with configurable retrain frequency
- [x] Purged k-fold with embargo; CPCV with path reconstruction
- [x] Lockbox guard and sentinel with `--force --reason`
- [x] JSONL registry with config, hashes, git SHA, fold metrics, wall time
- [x] Nested Optuna inside training folds with a bounded budget; every trial logged
- [x] Property tests: no training label interval overlaps a test fold, and the embargo holds

Acceptance: the property tests pass, and a harness run on synthetic data writes registry rows.

### Phase 5: models
Files: `src/nvquant/models/{base,factory,baselines,linear,trees,volatility,ensemble,v1_replica}.py`, `src/nvquant/models/deep/{nets,train}.py` (heads live in `nets.py`), `src/nvquant/evaluation/forecast.py`, `docs/V1_POSTMORTEM.md`, `tests/unit/test_{models,volatility}.py`.
- [x] Baselines: zero, historical mean, AR(p); ridge, elastic net; LightGBM
- [x] Deep: LSTM and GRU (Gaussian NLL and quantile heads), TCN, PatchTST-lite; one training loop with early stopping, gradient clipping, and seed ensembles
- [x] v1 replica against a persistence forecast
- [x] Volatility: GARCH, GJR-GARCH-t, EGARCH, HAR-RV, LightGBM vol; QLIKE, MSE, Mincer-Zarnowitz
- [x] Regimes: walk-forward Gaussian HMM plus my own forward filter
- [x] Ensembles: equal weight and purged-CV stacking
- [x] Forecast evaluation: R2_oos, DM-HLN, IC with Newey-West, IC decay, Pesaran-Timmermann, PIT and coverage, adaptive conformal, MCS
- [x] Synthetic recovery tests: planted signal gives positive IC, GBM gives none

Acceptance: `make train` finishes within budget, every model has OOS forecasts over the whole dev OOS range, and the recovery tests pass.

### Phase 6: strategy and backtest
Files: `src/nvquant/portfolio/{sizing,meta_labeling}.py`, `src/nvquant/backtest/{costs,vectorized,event,benchmarks,strategies}.py`, `tests/unit/test_backtest_engines.py`.
- [x] Threshold and scaled signals, vol targeting, fractional Kelly with caps, regime filter, meta-labeling
- [x] Next-open execution; the full cost model; 0 to 20 bps sweep; capacity sweep
- [x] Vectorized and event-driven engines agree to 1e-10 (tested)
- [x] Invariants: zero signal earns 0, always-long equals buy-and-hold minus entry cost, costs are monotone, positions are lagged
- [x] Benchmarks, including vol-targeted buy-and-hold and the random null

Acceptance: the engine agreement and invariant tests pass, and `make backtest` writes the strategy return matrix and registers every config.

### Phase 7: evaluation and risk
Files: `src/nvquant/evaluation/{metrics,significance,attribution,risk,importance}.py`, `src/nvquant/experiments/pipeline.py` (peer study, CPCV paths, and every stage), `tests/unit/test_{significance,evaluation}.py`, `tests/integration/test_pipeline_fast.py`.
- [x] Performance metrics, including rolling Sharpe and IC
- [x] PSR, DSR, PBO (CSCV), SPA, White RC, bootstrap CIs, MinTRL; hand-computed unit tests
- [x] Attribution on QQQ, SMH, and FF5 plus momentum with Newey-West
- [x] VaR and CVaR (historical, parametric, Cornish-Fisher, GARCH) with Kupiec and Christoffersen
- [x] Stress windows before 2025, regime-conditional results, block-bootstrap Monte Carlo
- [x] Peer study on all 10 peers
- [x] Explainability: TreeSHAP, MDA under purged CV, MDI, clustered MDA, stability

Acceptance: `make evaluate` writes reports/evaluation/*.json, and the significance tests match the hand-computed examples.

### Phase 8: reporting and docs
Files: `src/nvquant/reporting/{results,tearsheet,readme,figures}.py`, `app/streamlit_app.py`, `README.md`, `docs/{METHODOLOGY,RESULTS,V1_POSTMORTEM,INTERVIEW_NOTES}.md`, `docs/adr/*.md`, `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`.
- [x] `make report` writes reports/results.json, reports/tearsheet.html, and the README block
- [x] README rewrite with a mermaid diagram, "What didn't work", and "From v1 to v2"
- [x] METHODOLOGY, RESULTS, V1_POSTMORTEM, ADRs, CHANGELOG, CITATION.cff, CONTRIBUTING
- [x] INTERVIEW_NOTES with resume bullets that use only numbers from results.json
- [x] Streamlit app that reads reports/ only

Acceptance: the refresh-results check script passes, the app imports and renders headless, and docs contain no em dashes or hype words (tested).

### Phase 9: review and hardening
- [x] leakage-auditor PASS on `main...HEAD`
- [x] quant-reviewer PASS
- [x] silent-failure-hunter and pr-test-analyzer findings fixed or documented
- [x] /code-review findings fixed or documented

Acceptance: every finding has a commit that fixes it or a line in docs/RESEARCH_LOG.md that explains why it wasn't fixed.

### Phase 10: lockbox and ship
- [x] docs/PREREGISTRATION.md committed before the lockbox run
- [x] `make lockbox` run exactly once; sentinel committed
- [x] 2025 stress windows reported from the lockbox run only
- [x] `make reproduce` from a fresh clone
- [ ] results refreshed, PR opened with gh, then merged into main

Acceptance: the sentinel shows one run, its timestamp is after the preregistration commit, and the PR is merged.

### Stretch (after Phase 10, behind config flags)
- [ ] Relative value (NVDA vs AMD, NVDA vs SMH) with a Kalman hedge ratio, rolling cointegration, and OU half-life
- [ ] GitHub Pages workflow for the tear sheet
