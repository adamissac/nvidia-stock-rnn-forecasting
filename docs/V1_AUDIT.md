# v1 audit

This is an audit of the original project as it stood at commit `a686217` (which was still the tip of `main` when I started v2). Each finding below was re-checked against that commit on Python 3.12.14 with yfinance 1.7.0 and pandas 3.0.6. The commands I used are in the "How I checked" column so anyone can re-run them.

I kept this factual. The point is not that v1 was bad (it was a first project built from a tutorial), but that each problem maps to something v2 has to prevent.

## Summary

| # | Finding | Severity | Status at a686217 |
|---|---------|----------|-------------------|
| 1 | `download_data.py` is UTF-16LE without a BOM | Blocks everything | Confirmed |
| 2 | Test runner finds zero tests | CI is meaningless | Confirmed |
| 3 | yfinance MultiIndex columns break the CSV schema | Blocks the notebook | Confirmed |
| 4 | Notebook outputs come from a newest-first CSV, so the model learned to predict backwards in time | Invalidates results | Confirmed from saved outputs |
| 5 | README metrics are not produced anywhere | Unsupported claims | Confirmed |
| 6 | Test loop predicts 20 of 33 test days | Wrong evaluation | Confirmed |
| 7 | Methodology gaps (price-level target, no baseline, scaling, ReLU LSTM, no validation, one split) | Invalidates conclusions | Confirmed |
| 8 | README mojibake | Cosmetic | Confirmed |
| 9 to 16 | Additional findings (below) | Mixed | Confirmed |

## Findings from the original checklist

### 1. The download script cannot be parsed

`download_data.py` starts with the bytes `22 00 22 00 22 00 0a 00`, which is UTF-16LE with no byte order mark. `file` reports it as plain `data`.

How I checked: `python download_data.py` fails with `SyntaxError: source code cannot contain null bytes` (exit 1), and `python -m compileall -q .` also exits 1. CI runs `compileall`, so CI would fail at this step on every push. Decoding the file as UTF-16LE shows ordinary Python, so the content is fine and only the encoding is wrong.

### 2. The test suite runs zero tests

`tests/test_smoke.py` defines two bare functions (`test_requirements_file_exists`, `test_download_helper_exists`) in pytest style. Both `make test` and CI call `python -m unittest discover -s tests -v`, and `unittest` only collects `TestCase` subclasses.

How I checked: the run prints `Ran 0 tests` and `NO TESTS RAN`, and exits with code 5 on Python 3.12. So even after fixing finding 1, CI would stay red, and if it were green it would prove nothing.

### 3. Current yfinance breaks the CSV schema

With yfinance 1.7.0, `yf.download("NVDA", period="max", auto_adjust=True)` returns MultiIndex columns such as `('Close', 'NVDA')`. The script's renaming `str(c).replace(" ", "_")` turns them into headers like `('Close',_'NVDA')` and `('Date',_'')`.

How I checked: after renaming, `df[['Open']]` raises `KeyError: "None of [Index(['Open'], dtype='str')] are in the [columns]"`, which is exactly the lookup the notebook does in its second cell. v2 passes `multi_level_index=False` explicitly and validates the schema.

### 4. The notebook was trained on a newest-first CSV

The saved output of the first notebook cell shows the training CSV it actually used:

- dates in `MM/DD/YYYY` form, sorted newest first (`12/31/2024`, `12/30/2024`, `12/27/2024`, ...);
- Volume stored as a comma-formatted string (`155,659,203`).

`download_data.py` cannot produce that file: it writes ISO dates in ascending order with numeric volume. So the saved outputs came from a different, hand-made CSV.

The windowing code assumes ascending time: row `i` is the label and rows `i-60 .. i-1` are the inputs. With newest-first rows, rows `i-60 .. i-1` are the 60 sessions after row `i`, so every label is the day before its window. The model was trained to predict the past from the future.

The saved shapes back this up. The training file has 252 rows, which is exactly the number of NYSE sessions in calendar 2024, and its first row is 2024-12-31. The test file has 33 rows. If the test CSV had the same newest-first format, then `pd.concat((train['Open'], test['Open']))` followed by taking the last `len(test) + 60` values joins the 60 oldest training days (early 2024) directly onto the test period, so the first test windows mix January to March 2024 with 2025 prices.

Note: a 33-session test file that follows calendar 2024 is consistent with early 2025, which sits inside the v2 lockbox (2025-01-01 onward). I did not compute any performance on those dates during the audit. The v1 replica in v2 is evaluated on the development period only (see docs/V1_POSTMORTEM.md).

### 5. The README metrics do not come from the notebook

The README claims "an RMSE of 0.08 on unseen test data", training on "over 1,257 days", and "strong trend accuracy".

How I checked: I searched every notebook cell. No cell computes RMSE, MAE, or any directional metric. The saved outputs show 252 training rows (`len(nv_training_scaled)` prints `252`) and 192 training windows (`X_train.shape` prints `(192, 60)`). The comments next to those lines say `# 1257` and `# (1197, 60)`, which are left over from the tutorial the notebook was adapted from (the test variable is even called `fb_test_features`, from a Facebook stock version). "Trend accuracy" is stated but never measured.

### 6. The test loop covers 20 of 33 days

The test windows are built with `for i in range(60, 80)`, a hard-coded upper bound. The saved output shows `test_inputs.shape == (93, 1)`, so the correct loop is `range(60, 93)`, which gives 33 windows. Only 20 of the 33 test days (61%) get a prediction.

The plot then draws `nv_testing_processed` (33 actual values) and `y_pred` (20 predictions) on the same integer x-axis, labelled "Date". The two lines are different lengths and the axis is not dates.

### 7. Methodology gaps

- **Price-level target.** The model predicts the next open price. Price levels are non-stationary, so a small RMSE mostly reflects that tomorrow's price is close to today's.
- **No baseline.** There is no persistence forecast (tomorrow's open equals today's open). On price levels, persistence is very hard to beat, and without it an RMSE number has no reference point.
- **MinMax scaling of levels.** The scaler is fit on training prices only (correct), but it maps them into [0, 1]. Test prices above the training maximum map above 1, and the network never saw inputs in that range. With the download script's own 80/20 split of history through 2024, the training maximum open is 7.149 (split-adjusted) and the test maximum is 148.950, 20.8 times higher.
- **ReLU inside the LSTM cells.** `LSTM(100, activation='relu')` replaces the tanh cell activations. Unbounded activations in a recurrent cell can blow up, and Keras only uses the cuDNN kernel when the activation is tanh.
- **No validation and no early stopping.** It trains for 100 epochs on all 192 windows, with nothing held out.
- **One split.** There is a single train/test split, so there is no way to tell a real effect from a lucky period.
- **Tutorial leftovers.** `fb_test_features`, the stale shape comments, and a final markdown cell that holds nothing but about 200 lines of `---`.

### 8. README mojibake

The README contains `â€"` in 4 places. These are UTF-8 em dashes (`e2 80 94`) that were decoded as cp1252 and then saved again as UTF-8.

## Additional findings

9. **Overparameterized.** The four-layer, 100-unit LSTM has 282,101 trainable parameters (40,800 in the first layer, 80,400 in each of the next three, and 101 in the dense head). It is trained on 192 windows, which is about 1,469 parameters per training example.
10. **The download script and the notebook do not describe the same experiment.** The script splits the whole history 80/20 (6,528 sessions from 1999-01-22 through 2024-12-31, so it would train on 1999-01-22 to 2019-10-22 and test on 2019-10-23 onward). The notebook's saved run used one year of training data. Even with every bug fixed, running the documented steps would not reproduce the documented run.
11. **Nondeterminism.** The notebook seeds numpy and TensorFlow, but it doesn't enable op determinism, so repeated runs can give different weights.
12. **Unpinned dependencies.** `requirements.txt` has lower bounds only and there is no lock file. The README says Python 3.10+ while CI uses 3.12, and CI installs all of TensorFlow just to run `compileall`.
13. **The notebook never runs in CI.** Nothing checks that the documented pipeline executes.
14. **Unused imports.** `Activation` and `Flatten` are imported and never used, and numpy and pandas are imported several times.
15. **Split and dividend adjustment was never checked.** `auto_adjust=True` is used, but nothing verifies there is no discontinuity at NVDA's splits (4-for-1 on 2021-07-20 and 10-for-1 on 2024-06-10). The hand-made CSV in the notebook has post-split 2024 prices, but nothing records where it came from.
16. **No transaction costs, no trading rule, and no statistics.** v1 is a forecasting demo. It says nothing about whether the forecast is tradeable, and it has no significance test of any kind.

## What each finding turns into in v2

| Finding | v2 control |
|---------|-----------|
| 1, 2, 13 | uv-managed package, pytest, CI with an end-to-end synthetic run |
| 3, 15 | explicit yfinance arguments, pandera schemas, a data-quality report with split checks |
| 4 | ascending-time schema checks and property tests that perturbing the future never changes a feature |
| 5 | README numbers are injected from `reports/results.json` |
| 6 | the evaluation harness scores every out-of-sample date |
| 7 | stationary targets, naive baselines, walk-forward and purged CV, Diebold-Mariano tests |
| 9, 11 | small models, seed ensembles, deterministic torch settings |
| 10, 12 | one config-driven pipeline and a committed `uv.lock` |
| 16 | a cost-aware backtest, Deflated Sharpe Ratio, and PBO |
