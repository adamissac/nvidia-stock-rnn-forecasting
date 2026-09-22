# nvquant: NVDA quant research

This repo started as a tutorial Keras LSTM that predicted NVDA's next open price
(preserved in `legacy/`). v2 rebuilds it as a quant research project whose only
goal is honest, reproducible, out-of-sample evidence about whether daily NVDA
returns are forecastable and tradeable after costs. A clean null result is a
valid outcome. When something looks too good, assume a bug.

- Scope, source of truth: [docs/SPEC.md](docs/SPEC.md) (including its addenda)
- Phases, checkboxes, decisions: [docs/PLAN.md](docs/PLAN.md)
- Dated research log: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)

## Five non-negotiables

1. **No lookahead.** Every value used for a decision at time t is computable from
   data available at t's decision time. Exogenous series are lagged one trading
   day by default.
2. **Out-of-sample only.** Walk-forward or purged CV with embargo. The lockbox
   (2025-01-01 onward) is evaluated exactly once.
3. **Baselines and costs.** Every model and strategy is compared to naive
   baselines, net of transaction costs.
4. **Log every trial.** Every fitted configuration goes to the experiment
   registry, and trial counts feed the Deflated Sharpe Ratio.
5. **Generated numbers only.** Numbers in README and docs come from artifacts in
   `reports/`, never typed by hand.

## Make targets

`PROFILE=fast` (synthetic, under 5 min, used by CI) or `PROFILE=full` (default for data targets).

| target | what it does |
|--------|--------------|
| setup | `uv sync --all-extras` and pre-commit install |
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

- `src/nvquant/`: config, data, features, labels, cv, models, portfolio,
  backtest, evaluation, reporting, experiments, `cli.py`
- `configs/`: `base.yaml` plus `fast.yaml` and `full.yaml` overrides (pydantic-validated)
- `tests/`: `unit/`, `property/` (hypothesis), `integration/` (no network, ever)
- `reports/`: generated artifacts (committed), `reports/registry/trials.jsonl`
- `data/`: raw cache and feature store (git-ignored, never committed)
- `docs/`: spec, plan, log, methodology, results, ADRs; `app/`: Streamlit; `legacy/`: v1

## Coding conventions

- Python 3.12, type hints everywhere, numpy-style docstrings on public functions.
- Feature and label functions are pure: they never mutate inputs and return new frames.
- Anything fit on data (scalers, fracdiff d, HMM, GARCH, tuning) is fit inside a
  walk-forward training window by the harness, never on the full sample.
- Config-driven: no magic numbers in code paths that affect results; put them in `configs/`.
- Seeded: call `nvquant.experiments.repro.seed_everything` at every entry point.
- ruff (format + lint), mypy on `src/nvquant`, leakage lint on every edit (hook).
  Deliberate exceptions need `# leakage-ok: <reason>`.
- New features must be registered in `nvquant.features.registry` so the causality
  property test covers them. New models go through the `add-model` skill.
- Use context7 (or the installed package source) to check current library APIs
  before writing against yfinance, arch, lightgbm, torch, optuna, pandera,
  hmmlearn, exchange_calendars, statsmodels, or streamlit.

## Git rules

- Work on `quant-overhaul`. Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Commit and push at the end of every phase. Never push to main, never force-push.
  Landing on main happens only by merging the PR with `gh pr merge`.
- `git mv` for moves. No attribution trailers of any kind in commits or PRs.
- Never commit raw data. Only small synthetic fixtures live in git.

## Research log

At the end of every phase, append a dated entry to docs/RESEARCH_LOG.md with
four headings: Hypothesis, What ran, Result, Decision. Numbers in it must come
from `reports/` artifacts (quote the key path).

## Writing rules for every doc

- Plain, direct, first person, like a CS and math student explaining their own work.
- No em dashes. No hype words: cutting-edge, robust, leverage, seamless,
  state-of-the-art, powerful. `tests/unit/test_docs_style.py` enforces both.
- Every claim is backed by a generated number (from `reports/`) or a citation.

## Agents and skills

- `leakage-auditor` and `quant-reviewer` gate the end of each phase (both must PASS).
- `experiment-runner` runs long jobs and returns a compact summary.
- Skills: `leakage-guard`, `backtest-protocol`, `add-model <name>`, `refresh-results`.
