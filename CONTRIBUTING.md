# Contributing

## Setup

```bash
make setup          # uv sync, the macOS libomp fix for LightGBM, pre-commit hooks
make ci             # ruff, leakage lint, mypy, pytest (coverage >= 85%), and the fast end-to-end run
```

`make setup` needs [uv](https://docs.astral.sh/uv/). On macOS without Homebrew's `libomp`, `scripts/macos_libomp.py` points LightGBM at the OpenMP runtime that PyTorch ships.

## Rules

The rules are in [CLAUDE.md](CLAUDE.md). The short version:

1. No lookahead. New features go in `nvquant.features.registry`, so the causality property test covers them.
2. Out-of-sample only: walk-forward or purged CV. Don't touch lockbox dates (2025-01-01 onward).
3. Compare against the naive baselines, net of costs.
4. Every fit goes through the harness, which logs it to `reports/registry/trials.jsonl`.
5. Numbers in docs come from `reports/`. Run `make report` instead of editing numbers by hand.

## Adding a model

Follow `.claude/skills/add-model/SKILL.md`: implement `Forecaster`, register it in `nvquant.models.factory`, add a config block, add tests (including the planted-signal and pure-noise checks), run `make train backtest evaluate report`, and add a dated entry to docs/RESEARCH_LOG.md.

## Commits

Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change each.
