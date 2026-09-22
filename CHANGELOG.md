# Changelog

## 2.0.0 (2026-09-22)

A rebuild from the ground up. See [docs/V1_AUDIT.md](docs/V1_AUDIT.md) for why.

- Package `nvquant` (Python 3.12, uv, src layout) with a typer CLI and pydantic configs.
- Data layer: Yahoo, FRED, Ken French, and earnings dates, with a SHA256 manifest, pandera schemas, split checks, and a data-quality report.
- Causal feature store with walk-forward fitted features (fracdiff, HMM regimes), a causality property test over every feature, triple-barrier and meta-labels.
- Purged walk-forward, purged k-fold with embargo, CPCV, nested Optuna tuning, an experiment registry, and a lockbox guard.
- Baselines, linear models, LightGBM, LSTM/GRU/TCN/PatchTST with Gaussian and quantile heads, ensembles, GARCH-family and HAR volatility models, and a v1 replica.
- Two backtest engines that must agree, a cost model with square-root impact, benchmarks including vol-targeted buy and hold, and a random-signal null.
- PSR, DSR, MinTRL, PBO, SPA and White's RC, bootstrap CIs, attribution, VaR backtests, stress windows, a peer study, and feature importance.
- Generated reports: results.json, a tear sheet, the README results block, and a Streamlit app.
- The v1 notebook and scripts are preserved in `legacy/`.

## 1.x

- Renamed notebook (fixed "Predication" typo).
- Added requirements, license, citation, and disclaimer.
