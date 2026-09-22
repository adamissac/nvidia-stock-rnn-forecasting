# Project rules

The rules I hold myself to in this repo. [PLAN.md](PLAN.md) has the phase plan and the decisions behind it, and [RESEARCH_LOG.md](RESEARCH_LOG.md) is the dated log.

## Five non-negotiables

1. **No lookahead.** Every value used for a decision at time t is computable from data available at t's decision time. Exogenous series are lagged one trading day by default.
2. **Out-of-sample only.** Walk-forward or purged CV with embargo. The lockbox (2025-01-01 onward) is evaluated exactly once.
3. **Baselines and costs.** Every model and strategy is compared to naive baselines, net of transaction costs.
4. **Log every trial.** Every fitted configuration goes to the experiment registry, and trial counts feed the Deflated Sharpe Ratio.
5. **Generated numbers only.** Numbers in the README and docs come from artifacts in `reports/`, never typed by hand.

## Make targets

`PROFILE=fast` (synthetic, under a minute, what CI runs) or `PROFILE=full` (the default for data targets).

| target | what it does |
|--------|--------------|
| setup | `uv sync`, the macOS libomp fix for LightGBM, pre-commit install |
| data / data-report | download and cache raw data + manifest / data-quality report |
| features | build the feature and label store, regenerate docs/FEATURES.md |
| train | walk-forward forecasts for every enabled model |
| backtest | strategies, benchmarks, cost and capacity sweeps |
| evaluate | forecast tests, significance, risk, attribution, peers, importance |
| report | reports/results.json, tear sheet, README results block |
| reproduce | data through report, end to end |
| lockbox | the one-time lockbox evaluation (refuses to rerun) |
| test / lint / typecheck / ci | pytest / ruff + leakage lint / mypy / all of them |
| app | Streamlit app over reports/ |

## Directory map

- `src/nvquant/`: config, data, features, labels, cv, models, portfolio, backtest, evaluation, reporting, experiments, `cli.py`
- `configs/`: `base.yaml` plus `fast.yaml` and `full.yaml` overrides (pydantic-validated)
- `tests/`: `unit/`, `property/` (hypothesis), `integration/` (no network, ever)
- `reports/`: generated artifacts (committed), `reports/registry/trials.jsonl`
- `data/`: raw cache and feature store (git-ignored, never committed)
- `tools/`: the leakage lint; `docs/`; `app/`: Streamlit; `legacy/`: v1, untouched

## Coding conventions

- Python 3.12, type hints everywhere, numpy-style docstrings on public functions.
- Feature and label functions are pure: they never mutate inputs and return new frames.
- Anything fit on data (scalers, fracdiff d, HMM, GARCH, tuning) is fit inside a
  walk-forward training window by the harness, never on the full sample.
- Config-driven: no magic numbers in code paths that affect results; put them in `configs/`.
- Seeded: call `nvquant.experiments.repro.seed_everything` at every entry point.
- ruff (format + lint), mypy on `src/nvquant`, and `tools/leakage_lint.py` on the package.
  Deliberate exceptions need `# leakage-ok: <reason>`.
- New features must be registered in `nvquant.features.registry` so the causality
  property test covers them. New models go through `Forecaster` and the factory, with
  a planted-signal test and a pure-noise test.

## Review checklist before trusting a result

- Run `make ci`, and `tools/leakage_lint.py` on anything touching features, labels, cv,
  models, portfolio, or backtest.
- Walk the lookahead checklist in [LEAKAGE_CHECKLIST.md](LEAKAGE_CHECKLIST.md).
- Check the result against the backtest protocol in [METHODOLOGY.md](METHODOLOGY.md):
  next-open execution, full costs, the vol-targeted benchmark, trial-adjusted statistics.
- A net Sharpe above about 1.5 for a daily single-name strategy is a bug until proven
  otherwise. So is PnL concentrated in one regime, beta passing as alpha, or a result
  that only survives at zero costs.

## Git rules

- Work on a branch. Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Commit and push at the end of every phase. Never force-push.
- `git mv` for moves. Never commit raw data; only small synthetic fixtures live in git.

## Research log

At the end of every phase, append a dated entry to [RESEARCH_LOG.md](RESEARCH_LOG.md) with
four headings: Hypothesis, What ran, Result, Decision. Numbers in it come from `reports/`
artifacts, with the key named.

## Writing rules for every doc

- Plain, direct, first person, the way I'd explain my own work.
- No em dashes. No hype words: cutting-edge, robust, leverage, seamless,
  state-of-the-art, powerful. `tests/unit/test_docs_style.py` enforces both.
- Every claim is backed by a generated number (from `reports/`) or a citation.
