---
name: add-model
description: Scaffold, register, test, and evaluate a new forecasting model in nvquant end to end. Use this when asked to add, try, or compare a new model (for example "add a random forest", "try an XGBoost model", "add a Mamba model"), so the model goes through the shared interface, the walk-forward harness, and the trial registry instead of a one-off script.
argument-hint: "[model-name]"
arguments: [model]
---

# Add a model: $model

Follow these steps in order. The interface and an example are in
[references/interface.md](references/interface.md).

1. **Implement** `$model` in the right module under `src/nvquant/models/`
   (tabular models in `linear.py`/`trees.py`, torch models in `deep/`). It must
   subclass `Forecaster` and implement `fit(X, y, sample_weight)` and
   `predict(X, index)`. All fitting (scalers included) happens inside `fit`.
2. **Register** it in `nvquant.models.factory.MODEL_REGISTRY` and add a config
   block under `models:` in `configs/base.yaml` (keep it disabled in `fast`
   unless it trains in under a few seconds).
3. **Test** it in `tests/unit/test_models.py`: it fits and predicts on synthetic
   data, predictions have the right index, it is deterministic under a fixed
   seed, and on `synthetic.planted_signal` it gets positive out-of-sample IC
   while on `synthetic.gbm` its IC is not significant.
4. **Run walk-forward** through the experiment-runner agent:
   `uv run nvquant train --config configs/full.yaml --only $model`. Don't read
   the full logs in the main context; ask for the metric summary.
5. **Log trials.** The harness registers the run automatically; confirm a new
   `kind: model` row for `$model` in `reports/registry/trials.jsonl`.
6. **Update results.** Run `make backtest evaluate report` (through the
   experiment-runner) so the model shows up in `reports/results.json`, and add a
   dated entry in docs/RESEARCH_LOG.md with the hypothesis and the result.
7. Use leakage-guard's checklist before you call it done.
