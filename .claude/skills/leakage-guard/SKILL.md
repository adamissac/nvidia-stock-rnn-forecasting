---
name: leakage-guard
description: Lookahead and data-snooping checklist for nvquant. Use this whenever you write, edit, or review anything in the features, labels, cv, backtest, or models packages, including a one-line change to a rolling window, a shift, a scaler, a train/test split, an HMM, or execution timing. Also use it before trusting any result that looks better than expected, since the usual cause is leakage.
paths:
  - "src/nvquant/features/**"
  - "src/nvquant/labels/**"
  - "src/nvquant/cv/**"
  - "src/nvquant/backtest/**"
  - "src/nvquant/models/**"
  - "src/nvquant/portfolio/**"
  - "src/nvquant/experiments/**"
---

# Leakage guard

The rule: every value used for a decision at time t is computable from data
available at t's decision time (after the close of session t; fills at the open
of t+1). Exogenous series (anything that isn't the target ticker's own OHLCV)
are lagged one session by default through `FeatureSpec.lag`.

## Before you finish an edit in these packages

1. Run the lint: `uv run python .claude/skills/leakage-guard/scripts/leakage_lint.py src/nvquant`.
   The PostToolUse hook runs it too. A deliberate exception needs
   `# leakage-ok: <reason>` on the same line.
2. Walk the checklist in [references/checklist.md](references/checklist.md) for the
   patterns the lint can't see (fitting on the wrong rows, tuning on test folds,
   universe chosen with hindsight, revised macro data).
3. If you added or changed a feature function, make sure it is registered in
   `nvquant.features.registry` so the causality property test covers it. The
   pattern is in [references/property_tests.md](references/property_tests.md).
4. If you touched labels or splitters, run `uv run pytest tests/property tests/unit/test_cv.py -q`.

## Where fitted things live

- Anything that is fit (scalers, fracdiff `d`, HMM parameters, GARCH parameters,
  Optuna trials) is fit inside a walk-forward training window by
  `nvquant.experiments.harness`, never at feature-store build time.
- Features in the store are pure functions of past data. Fitted features
  (`nvquant.features.fitted`) implement `fit(train_end)` then `transform`.
- Labels carry `t_end`; purging in `nvquant.cv` uses it. Never drop that column.
