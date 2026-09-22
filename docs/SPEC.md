# Mission

You are working in a local clone of my GitHub repo adamissac/nvidia-stock-rnn-forecasting (confirm with `gh repo view`; if you're not inside it, clone it with `gh repo clone`, tell me to relaunch Claude Code inside it, and stop). Right now it's a tutorial-style Keras LSTM that predicts NVDA's next opening price from the previous 60 opens. Rebuild it into a research-grade quantitative finance project that I can defend line by line in quant research and ML engineering interviews.

North star: honest, reproducible, out-of-sample evidence. Complexity is welcome, but every component has to earn its place by producing better out-of-sample evidence, preventing a known bias, or making results reproducible. No dead code, no unused frameworks, no hand-typed numbers in docs. If nothing beats the baselines after costs, that is a valid result: report it plainly and rigorously. A clean null result is worth more than a suspicious Sharpe of 3. When a result looks too good, assume a bug and hunt for it before believing it.

Primary methods reference: López de Prado, Advances in Financial Machine Learning (2018), chapters 3 to 8 and 11 to 14 (labeling, sample weights, fractional differentiation, cross-validation, feature importance, backtest statistics).

ultrathink before you plan.

# Ground rules

1. This message is Phase 0 only: audit, tooling, and a plan. Do Phase 0, commit, push the branch, then stop and wait for my approval. After I approve, I'll restart you so the new skills, agents, and hooks load, and I'll set a /goal to execute the plan.
2. First actions: pull main, create branch `quant-overhaul`, and save this entire message verbatim to docs/SPEC.md so it survives context compaction. CLAUDE.md must point to it. docs/SPEC.md is the source of truth for scope (the "stop after Phase 0" instruction applies only to this first session).
3. Before writing code against any third-party library (yfinance, arch, lightgbm, torch, optuna, pandera, hmmlearn, exchange_calendars, statsmodels, streamlit), pull current docs through the context7 plugin. APIs drift: current yfinance returns MultiIndex columns by default, which is one of the bugs in this repo.
4. Keep the main context clean: use Explore for research, the experiment-runner agent (below) for training runs and verbose logs, and the reviewer agents at the end of each phase. When modules are independent and their interfaces are pinned, you may parallelize implementation with subagents in separate git worktrees and merge after review.
5. Commit at the end of every phase with conventional commit messages and push the branch. Never push to main, never force-push. Use `git mv` when moving files so history survives.
6. Author metadata is "Adam Issac" everywhere (pyproject authors, CITATION.cff, docs).
7. Depth over breadth. If time or compute runs short, cut stretch items first, then the most exotic deep model. Never cut the validation, statistics, or testing layers.

# Phase 0

## 0.1 Audit v1: write docs/V1_AUDIT.md

These findings come from a snapshot at commit a686217. Verify each against HEAD, quantify where you can, and add anything else you find:

1. download_data.py is saved as UTF-16LE without a BOM, so Python fails with "source code cannot contain null bytes". This also fails CI's `python -m compileall` step.
2. CI and `make test` run `python -m unittest discover`, but tests/test_smoke.py uses bare pytest-style functions, so zero tests run, and Python 3.12 exits with code 5 when no tests run.
3. Even with the encoding fixed, current yfinance returns MultiIndex columns from `yf.download`, so the `str(c).replace(" ", "_")` renaming produces headers like `('Close',_'NVDA')` and the notebook's `[['Open']]` lookup fails.
4. The notebook's saved outputs came from a different CSV than the download script writes: MM/DD/YYYY dates sorted newest first, and Volume as a comma-formatted string. The windowing assumes ascending time, so each label was the day before its 60-day window, meaning the model learned to predict backwards in time. If the test CSV had the same format, the train/test concatenation also stitched the oldest training days onto the test period.
5. README claims (RMSE 0.08, trained on 1,257 days, "strong trend accuracy") aren't produced by the notebook. No RMSE or directional metric is computed anywhere, and the saved output shows 252 training rows and 192 windows (the code comments say 1257 and 1197, left over from a tutorial).
6. The test loop hard-codes `range(60, 80)`, so only 20 of the 33 test days get predictions, and the plot draws 33 actuals against 20 predictions.
7. Methodology: the target is a non-stationary price level; there's no naive baseline (a persistence forecast would likely match it); MinMax scaling of price levels can't extrapolate once test prices leave the training range; the LSTM layers use ReLU (can be unstable and rules out the cuDNN kernel); 100 epochs with no validation or early stopping; one train/test split; tutorial leftovers like `fb_test_features`; a junk markdown cell at the end.
8. The README has mojibake (UTF-8 em dashes decoded as cp1252).

Keep the tone factual and neutral. This becomes the "v1 to v2" story later.

## 0.2 Project memory and agent tooling

Read the Claude Code docs on skills, subagents, and hooks before writing these (ask the claude-code-guide agent or fetch the docs), and use the skill-creator plugin to author the skills (skip its full eval loop). Merge into the existing .claude/settings.json, since the plugin installs created it. Skills may reference modules that later phases create; update them when those interfaces land.

CLAUDE.md (under 120 lines, rules not essays):
- One paragraph on the project's purpose, plus pointers to docs/SPEC.md, docs/PLAN.md, and docs/RESEARCH_LOG.md.
- The five non-negotiables:
  1. No lookahead. Every value used for a decision at time t is computable from data available at t's decision time. Exogenous series are lagged one trading day by default.
  2. Out-of-sample only. Walk-forward or purged CV with embargo. The lockbox is evaluated exactly once.
  3. Baselines and costs. Every model and strategy is compared to naive baselines, net of transaction costs.
  4. Log every trial. Every fitted configuration goes to the experiment registry, and trial counts feed the Deflated Sharpe Ratio.
  5. Generated numbers only. Numbers in README and docs come from artifacts in reports/, never typed by hand.
- Make targets, a directory map, coding conventions (type hints, numpy-style docstrings, pure feature functions that never mutate inputs, config-driven, seeded), and git rules.
- Append a dated entry to docs/RESEARCH_LOG.md at the end of every phase: hypothesis, what ran, result, decision.
- Writing rules for every doc: plain, direct, first person, sounding like me (a CS and math student) explaining my own work. No em dashes. No hype words ("cutting-edge", "robust", "leverage", "seamless", "state-of-the-art", "powerful"). Every claim is backed by a generated number or a citation.

Skills in .claude/skills/ (descriptions a little pushy so they trigger; each SKILL.md short, with details in references/ and scripts/):
1. leakage-guard: reference skill with `paths` scoped to the features, labels, cv, backtest, and models packages. The lookahead checklist: negative shifts outside labels, centered rolling windows, backfill, scalers or feature selection fit on full data, full-sample normalization, HMM smoothed posteriors or Viterbi paths used as features, tuning on test folds, same-bar execution, ex-post universe selection, and revised macro data. It also documents the property-test pattern from Phase 3. Bundle scripts/leakage_lint.py, a fast static check for these patterns that allows negative shifts only under labels/ or on lines marked `# leakage-ok: <reason>`.
2. backtest-protocol: reference skill covering execution timing, the cost model, required benchmarks, required metrics and statistical tests, trial registration, and the lockbox rule.
3. add-model: user-invocable task skill that takes the model name as an argument. Scaffold a model on the shared interface, register it, add unit tests, run walk-forward through the experiment-runner, log trials, and update the results table.
4. refresh-results: task skill with `disable-model-invocation: true`. Run `make report`, regenerate the README block between `<!-- RESULTS:START -->
Out-of-sample decisions from 2009-07-22 to 2024-12-31 (3884 sessions), after costs, next-open execution. 80 strategy configurations were backtested, and all of them count as trials in the Deflated Sharpe Ratio.

| Strategy | Net Sharpe [95% CI] | CAGR | Vol | Max DD | Turnover/yr | DSR |
|---|---|---|---|---|---|---|
| `hist_mean` + voltarget | 1.18 [0.68, 1.64] | 43.3% | 36.0% | -46.8% | 4.4 | 0.99 |
| `ridge` + voltarget | 1.00 [0.51, 1.49] | 33.0% | 34.5% | -61.3% | 22.8 | 0.94 |
| `lgbm` + voltarget | 1.14 [0.64, 1.61] | 39.3% | 34.1% | -53.2% | 18.0 | 0.98 |
| `lstm_gauss` + voltarget | 0.83 [0.38, 1.33] | 22.2% | 29.1% | -56.4% | 26.4 | 0.80 |
| `gru_quant` + kelly | 0.95 [0.45, 1.40] | 38.3% | 44.6% | -62.8% | 51.3 | 0.91 |
| `tcn` + voltarget | 0.84 [0.38, 1.31] | 22.5% | 29.0% | -55.3% | 46.5 | 0.82 |
| `patchtst` + voltarget | 1.01 [0.52, 1.47] | 28.0% | 28.5% | -36.1% | 36.4 | 0.94 |
| `ens_equal` + voltarget | 0.92 [0.46, 1.42] | 26.7% | 31.0% | -41.2% | 28.8 | 0.88 |
| `ens_stack` + voltarget | 1.03 [0.54, 1.49] | 33.1% | 33.0% | -45.8% | 18.0 | 0.95 |
| `meta` + voltarget | 0.45 [-0.05, 0.93] | 4.0% | 9.8% | -31.5% | 12.6 | 0.25 |
| Vol-targeted buy and hold | 1.21 [0.69, 1.67] | 45.2% | 36.3% | -46.8% | 4.3 | benchmark |
| Buy and hold NVDA | 1.10 [0.61, 1.57] | 48.9% | 46.0% | -67.2% | 0.1 | benchmark |
| Buy and hold SMH | 0.89 [0.43, 1.39] | 22.7% | 27.2% | -46.9% | 0.1 | benchmark |
| Buy and hold QQQ | 0.98 [0.50, 1.50] | 19.4% | 20.1% | -36.7% | 0.1 | benchmark |
| 200-day trend rule | 1.24 [0.77, 1.71] | 48.5% | 37.4% | -53.3% | 3.8 | benchmark |

- Best development strategy: `hist_mean` + voltarget. Its Sharpe minus the vol-targeted buy and hold's is -0.03. DSR 0.99 (with every logged fit counted as a trial: 0.95).
- Probability of backtest overfitting (CSCV): 0.29. Hansen SPA p-value against vol-targeted buy and hold: 0.15; against buy and hold: 0.27.
- Random long/flat signals with the same turnover beat it 2.0% of the time.
- Fama-French 5 + momentum alpha: 32.8% a year (t = 3.98). Share of its summed daily PnL earned in 2023 and 2024: 29%.

| Forecast model | IC | IC t (NW) | R2 OOS vs zero | DM p vs zero | Hit rate |
|---|---|---|---|---|---|
| `ar` | 0.056 | 3.35 | 0.49% | 0.077 | 52.9% |
| `elastic_net` | 0.036 | 2.24 | 0.21% | 0.163 | 52.9% |
| `ens_equal` | 0.010 | 0.60 | 0.17% | 0.601 | 50.9% |
| `ens_stack` | 0.000 | 0.00 | -0.18% | 0.534 | 52.1% |
| `gru_gauss` | -0.008 | -0.48 | -0.96% | 0.060 | 50.0% |
| `gru_quant` | 0.016 | 0.91 | -1.45% | 0.034 | 50.1% |
| `hist_mean` | 0.035 | 2.27 | 0.23% | 0.126 | 52.8% |
| `lgbm` | 0.021 | 1.29 | 0.12% | 0.528 | 53.1% |
| `lstm_gauss` | 0.021 | 1.25 | -0.10% | 0.828 | 51.1% |
| `lstm_quant` | 0.031 | 1.80 | -0.22% | 0.684 | 51.6% |
| `patchtst` | -0.000 | -0.00 | -2.01% | 0.000 | 50.7% |
| `ridge` | 0.033 | 1.99 | 0.42% | 0.066 | 52.1% |
| `tcn` | -0.014 | -0.86 | -1.21% | 0.029 | 49.8% |

v1 replica (the original stacked ReLU LSTM on price levels, refit yearly): median absolute error 0.75 vs 0.02 for persistence; worse than persistence on 92% of days; on 10.6% of days its forecast exceeded 10x the training range (ReLU blow-up once prices left the MinMax range). Direction accuracy 49.1% vs 52.9% up days. Details: [docs/V1_POSTMORTEM.md](docs/V1_POSTMORTEM.md).

Lockbox (2025-01-01 onward): not evaluated yet.

<sub>Generated by `make report` from `reports/results.json` (git f3530da-dirty, data e2fdd3b2b961, config ba8c37786f54).</sub>
<!-- RESULTS:END -->` from reports/results.json, then run quant-reviewer on the diff.

Subagents in .claude/agents/:
1. leakage-auditor: read-only (Read, Grep, Glob, and Bash for tests and the lint), preloads leakage-guard, `memory: project`. Audits a diff or module for lookahead and data snooping. Returns PASS or FAIL with file:line findings and a proposed fix for each.
2. quant-reviewer: read-only, preloads backtest-protocol, plays a skeptical PM. Flags a net Sharpe above about 1.5 for a daily single-name strategy, PnL concentrated in one regime (the 2023 to 2024 AI rally especially), beta passing as alpha, cost fragility, weak trial-adjusted significance (DSR, PBO), and any README claim that doesn't match reports/results.json. Returns PASS or FAIL.
3. experiment-runner: Bash, Read, Grep, Glob only, never edits source. Runs make targets and training jobs, keeps logs out of the main context, and returns a compact metrics summary with artifact paths.

Hooks and permissions in .claude/settings.json (write hook scripts in Python so they work on macOS, Linux, and Windows; check the hooks reference for the exact stdin schema):
- PostToolUse on Edit|Write for .py files: `ruff format` and `ruff check --fix` on the edited file, then the leakage lint. On lint violations, exit 2 with the findings on stderr so you see them and fix them. Exit 0 quietly if ruff or the project environment isn't installed yet.
- Deny rules for pushing to main, force pushes, and deleting data/raw.
- Keep the enabledPlugins and extraKnownMarketplaces entries so the plugin setup is reproducible.

The quantitative-trading plugin's backtesting-frameworks and risk-metrics-calculation skills are useful references. Where they conflict with our skills, ours win because they're stricter.

## 0.3 Write docs/PLAN.md, then stop

Turn the target design below into phases with checkboxes, concrete acceptance criteria, the files each phase creates, a compute budget for a CPU laptop (the full pipeline should finish in about 3 hours; if your estimate is higher, shrink models, retrain frequency, or seed counts and note the trade-off), risks, and the decisions you need from me listed at the top (for example long/flat versus long/short, GPU availability, whether to rename the repo). Commit as "chore: v1 audit, agent tooling, and v2 plan", push, print a short summary, and stop.

# Target design (goes into PLAN.md; don't build it yet)

## Repo shape
- Python 3.12 managed with uv (pyproject.toml plus a committed uv.lock). src layout: src/nvquant/ with config, data, features, labels, cv, models, portfolio, backtest, evaluation, reporting, and experiments subpackages, plus a typer CLI.
- configs/ (YAML validated by pydantic), tests/ (unit, property, integration), notebooks/ (thin, importing from the package, never the source of truth), reports/ (generated), docs/, app/ (Streamlit), legacy/ (original notebook preserved unmodified via git mv, with a short README).
- Make targets: setup, data, data-report, features, train, backtest, evaluate, report, reproduce, lockbox, test, lint, typecheck, ci, app. Two profiles: fast (synthetic or tiny data, under 5 minutes, used by CI) and full.
- CI on GitHub Actions: uv sync, ruff, mypy on src/nvquant (strict where practical, with pandas-stubs), pytest with at least 85% coverage on src/, and an end-to-end smoke run of the whole pipeline on synthetic data. Tests never touch the network. Add pre-commit. pytest replaces unittest.
- Reproducibility: global seeding (python, numpy, torch), deterministic torch settings where possible, a data manifest (source, download time, SHA256 per file), and every artifact tagged with git SHA, config hash, and data hash. Detect CUDA or MPS, default to CPU.
- Raw data never goes in git (data/ is already ignored). Only small synthetic fixtures are committed.

## Phase 1: foundation
Packaging, config, CLI, logging, Makefile, CI, pre-commit, and the legacy/ move. Synthetic data generators with known properties (GBM, GARCH, regime-switching, and a series with a planted weak signal) so tests can prove models recover signal when it exists and find none when it doesn't. Acceptance: CI green on the branch.

## Phase 2: data
- Universe: NVDA; peers AMD, AVGO, TSM, INTC, MU, QCOM, TXN, AMAT, LRCX, ASML; plus SMH, SOXX, QQQ, SPY, ^SOX, ^VIX, ^VIX3M, ^TNX.
- yfinance with explicit `multi_level_index=False` and `auto_adjust=True`, retries with backoff, a parquet cache in data/raw, and the manifest.
- FRED DGS10 and DGS2 through FRED's keyless CSV endpoint. Market-observed series only; no revised macro data unless it comes from ALFRED vintages.
- Ken French daily Fama-French 5 factors plus momentum, downloaded as CSV zips (no pandas-datareader).
- Historical NVDA earnings dates, with point-in-time caveats documented.
- NYSE session alignment via exchange_calendars, pandera schemas, and a data-quality report covering missing sessions, stale prices, outliers, and explicit checks that no split discontinuities remain in prices or volume (including the 4-for-1 on 2021-07-20 and the 10-for-1 on 2024-06-10).
- Modeling start date is the latest first-valid date across required inputs, documented.

## Phase 3: features and labels
- Features, grouped, each with lookback and lag metadata: log returns at several horizons; realized vol (close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) and vol-of-vol; momentum and short-term reversal; causal RSI, MACD, Bollinger %B, and ATR; volume z-score, Amihud illiquidity, dollar volume; rolling beta and correlation to SMH and QQQ, relative strength versus peers, peer dispersion; VIX level and change, VIX/VIX3M term structure, 10y minus 2y slope, rate changes; days to and from earnings, month-end, options-expiration week; fixed-width fractionally differentiated price with the minimum d that passes ADF, chosen on training data only.
- Labels: forward log returns at 1, 5, and 21 days; vol-normalized forward returns; triple-barrier labels with volatility-scaled barriers and a vertical barrier; meta-labels; sample weights from average label uniqueness.
- The signature test: a hypothesis property test for every feature function. For random cutoffs t, perturb all data strictly after t and assert every feature value up to t is unchanged. Also test that every label carries its end time and that purging uses it.
- A materialized feature store (parquet) and auto-generated feature docs.

## Phase 4: validation framework
- Expanding and rolling walk-forward with configurable retrain frequency; purged k-fold with embargo; combinatorial purged CV producing multiple backtest paths.
- Lockbox: 2025-01-01 to the latest date, untouched until Phase 10. `make lockbox` writes a timestamped sentinel and refuses to rerun without `--force` and a logged reason. Nothing in Phases 5 to 9 computes performance on lockbox dates, including stress windows and peer runs; data-quality checks are the only exception.
- Experiment registry: every fit logs config, config hash, git SHA, data hash, fold metrics, and wall time. Optuna tuning happens only inside training folds (nested), with bounded budgets and every trial logged.
- Tests prove no training label window overlaps a test fold and that the embargo holds.

## Phase 5: models (one interface, one evaluation harness)
- Return baselines: zero return (random walk), historical mean, AR(p). Linear: ridge, elastic net. Trees: LightGBM.
- Deep models in plain PyTorch with one shared training loop (inner validation split, early stopping, gradient clipping, seed ensembles): LSTM and GRU on stationary features predicting returns, with a heteroscedastic Gaussian NLL head and a quantile (pinball) head; a TCN; a small PatchTST-style Transformer. Keep them small; the signal-to-noise ratio is low.
- The v1 replica: the original stacked LSTM on price levels, run through the new harness against a persistence forecast, written up honestly in docs/V1_POSTMORTEM.md.
- Volatility: GARCH(1,1), GJR-GARCH with Student-t errors, EGARCH (arch package), HAR-RV on range-based realized vol, and a LightGBM vol model, scored with QLIKE, MSE, and Mincer-Zarnowitz regressions. Vol forecasts drive position sizing and VaR.
- Regimes: Gaussian HMM fit walk-forward. Features use forward-filtered state probabilities computed only from data up to t (implement the forward filter yourself). Never use smoothed posteriors or full-sample Viterbi paths.
- Ensembles: equal weight, and stacking fit with purged CV.
- Forecast evaluation: out-of-sample R2 versus the zero forecast, Diebold-Mariano with the Harvey-Leybourne-Newbold correction, Spearman IC with Newey-West t-stats and IC decay by horizon, the Pesaran-Timmermann directional test, calibration for probabilistic models (PIT histograms, interval coverage), adaptive conformal intervals, and a Model Confidence Set (arch.bootstrap.MCS).

## Phase 6: strategy and backtest
- Signal to position: thresholded and scaled forecasts, vol targeting with the GARCH forecast, fractional Kelly with leverage caps, an optional regime filter, and meta-labeling (a primary model picks the side, a secondary classifier sizes the bet by its predicted probability of success).
- Execution: decisions use data through the close of day t and fill at the open of t+1 (configurable). Costs: half-spread, commission, volatility-scaled slippage, square-root market impact at an assumed AUM (show where capacity starts to bite), and borrow cost on shorts. Cost sensitivity from 0 to 20 bps per side.
- Two independent engines, vectorized and event-driven, that must agree within a tight tolerance on every strategy (tested).
- Invariant tests: a zero signal earns exactly zero; always-long matches buy-and-hold minus entry cost; higher costs never raise PnL; positions are always lagged.
- Benchmarks: buy-and-hold NVDA; vol-targeted buy-and-hold NVDA (the key ablation: does the model add anything beyond vol targeting?); SMH; QQQ; a 200-day moving-average trend rule; a null distribution from random signals with matched turnover.

## Phase 7: evaluation and risk
- Metrics: CAGR, annualized vol, Sharpe, Sortino, Calmar, max drawdown and duration, hit rate, profit factor, turnover, exposure, skew, kurtosis, tail ratio, rolling Sharpe, rolling IC.
- Significance: Probabilistic and Deflated Sharpe Ratio (Bailey and López de Prado 2014) using the registry's trial count and the variance of Sharpe across trials, with per-period rather than annualized Sharpe in the formulas; Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado, and Zhu 2017); Hansen's SPA and White's Reality Check against the benchmarks (arch.bootstrap); stationary block bootstrap confidence intervals with optimal block length; minimum track record length. Unit-test PSR, DSR, and PBO against hand-computed examples.
- Attribution: regressions on QQQ, SMH, and Fama-French 5 plus momentum with Newey-West t-stats. Report alpha and betas.
- Risk: historical, parametric, Cornish-Fisher, and GARCH VaR and CVaR, with Kupiec and Christoffersen backtests; drawdown analysis; named stress windows before 2025 (2008 crisis, Q4 2018, February to March 2020, the 2022 drawdown); regime-conditional performance; block-bootstrap Monte Carlo of equity paths.
- Selection bias: NVDA was picked with hindsight. Run the identical pipeline, unchanged, on every peer (winners and losers, INTC included) over the development period and report the distribution. If the method only works on NVDA, say so.
- Explainability: SHAP for LightGBM, MDA permutation importance under purged CV, MDI, clustered feature importance, and importance stability across folds.

## Phase 8: reporting and docs
- `make report` builds a static HTML tear sheet and reports/results.json, then injects the README results block.
- README rewrite: short summary, results table (out-of-sample, net of costs, with DSR and PBO), a mermaid architecture diagram, methodology summary, "What didn't work", "From v1 to v2", reproduction steps, limitations, and a not-investment-advice disclaimer.
- docs/METHODOLOGY.md (math in GitHub LaTeX), docs/RESULTS.md, docs/V1_POSTMORTEM.md, short ADRs in docs/adr/, CHANGELOG, CITATION.cff replacing CITATION.md, and an updated CONTRIBUTING.
- docs/INTERVIEW_NOTES.md, written for me: for each component, a plain-English explanation, why it exists, which failure it prevents, likely interview questions with answers, and three resume bullets that use only numbers from reports/results.json and emphasize rigor over returns.
- A Streamlit app in app/ that reads reports/ artifacts (no training at runtime): equity curves, drawdowns, rolling Sharpe, regime overlay, forecast versus realized vol, feature importance, cost sensitivity.

## Phase 9: review and hardening
Run leakage-auditor and quant-reviewer on the full diff against main, the pr-review-toolkit agents (at least silent-failure-hunter and pr-test-analyzer), and /code-review. Fix every finding or document why not.

## Phase 10: lockbox and ship
Write docs/PREREGISTRATION.md naming the final configurations and the metrics to report, commit it, then evaluate the lockbox exactly once. Report the 2025 stress windows (the 2025-01-27 DeepSeek selloff and the April 2025 tariff shock) only from this lockbox run. Run `make reproduce` from a fresh clone, refresh results, and open a PR into main with gh: summary, key results table, the v1 to v2 story, and a checklist of the five non-negotiables.

## Stretch (only after Phase 10, each behind a config flag)
Relative value: NVDA versus AMD and versus SMH with a Kalman-filter hedge ratio, rolling cointegration tests, and Ornstein-Uhlenbeck half-life, run through the same backtest and statistics. A GitHub Pages workflow that publishes the tear sheet.

## Out of scope
Intraday data, news or alternative data (no free point-in-time source), live or paper trading.

---

# Addendum (2026-09-21, from the owner, after Phase 0 was requested)

"do all phases and dont ask for my approval for anything"

So ground rule 1's stop after Phase 0 is lifted: Phase 0 through Phase 10 run in one go, with no approval gates. Decisions that PLAN.md would have asked about are made with the defaults recorded at the top of docs/PLAN.md. All other rules above still apply (no pushes to main, no force-pushes, lockbox evaluated exactly once, PR into main at the end).

# Addendum 2 (2026-09-21, from the owner)

"continue and finish it all and make sure all tasks are pushed to main and everything is completed"

So the finished work lands on main. Work still happens on `quant-overhaul`, the PR is still opened, and then the PR is merged into main with `gh pr merge` (a regular merge commit, never a force-push, never a direct `git push` to main). The deny rules in .claude/settings.json stay as specified.
