# Lookahead checklist

The patterns I check for before trusting anything in features, labels, cv, models,
portfolio, or backtest. `tools/leakage_lint.py` catches the mechanical ones (the LK codes);
the rest need reading the code.

Each item: what it looks like, why it leaks, what to do instead.

| # | Pattern | Why it leaks | Do this instead |
|---|---------|--------------|-----------------|
| 1 | `shift(-k)` outside `labels/` | reads k rows into the future | only labels look forward; everything else uses `shift(k)` with k >= 0 (lint LK001) |
| 2 | `rolling(..., center=True)` | the window includes future rows | trailing windows only (LK002) |
| 3 | `bfill()` / `fillna(method="bfill")` | copies a future value into the past | `ffill()` with a limit, or leave NaN and drop warmup rows (LK003) |
| 4 | scaler, PCA, or feature selector fit on the full sample | test-period mean/variance shapes training inputs | fit inside `Forecaster.fit` on training rows only (LK005) |
| 5 | `(x - x.mean()) / x.std()` in a feature | full-sample normalization | rolling or expanding z-score (LK004) |
| 6 | HMM `predict_proba`, `predict`, `decode` | smoothed posteriors and Viterbi paths use the whole sequence, including the future | `nvquant.models.regime.forward_filter`, which only uses observations up to t (LK006) |
| 7 | tuning hyperparameters on the walk-forward test block | the test block becomes training data for the choice | Optuna inside `nvquant.experiments.tuning`, on purged folds of the training window only |
| 8 | position at t times return at t | same-bar execution: you trade on the close you just saw | positions are decided after close t and earn the open(t+1) to open(t+2) return; the engines apply the lag, tests check it (LK008) |
| 9 | picking the universe or the ticker because it did well | ex-post selection | the peer study runs the same config on every peer; report the distribution |
| 10 | revised macro series (GDP, payrolls) | values published later than the date they are stamped with | only market-observed series (Treasury yields, VIX); FRED series are lagged one session because H.15 publishes the next business day |
| 11 | dividend-adjusted levels used as a level feature | the adjustment factor depends on future dividends | use returns or ratios; the fracdiff feature uses log price, which is only shifted by a constant between dividends (documented in docs/METHODOLOGY.md) |
| 12 | `train_test_split`, `KFold`, `cross_val_score` | shuffles or ignores label overlap | `nvquant.cv` splitters with purging and embargo (LK007) |
| 13 | using lockbox dates (>= 2025-01-01) before Phase 10 | the lockbox stops being out-of-sample | `nvquant.data.loader.load_market_data(mode="dev")` truncates at the cutoff; don't bypass it |
| 14 | earnings dates used further ahead than they are announced | the next date isn't public until about four weeks before | `days_to_earnings` is capped at 20 sessions |

When a backtest Sharpe looks too good, go through this table before believing it.

## The signature test

For every registered feature function f and random cutoff t: build market data D, build D'
by perturbing every value strictly after t, and assert `f(D).loc[:t]` equals `f(D').loc[:t]`.
If a feature ever reads a row after t, some perturbation changes its value at or before t,
and hypothesis finds it. `tests/property/test_feature_causality.py` iterates over
`nvquant.features.registry.all_specs()`, so a new feature is covered as soon as it's
registered. The same idea covers labels, the fitted features, the HMM forward filter, the
splitters, and the walk-forward harness (`tests/unit/test_harness.py`).
