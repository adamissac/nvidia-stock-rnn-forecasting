# v1 postmortem

v1 was a four-layer stacked LSTM that read the last 60 opening prices, MinMax-scaled to [0, 1], and predicted the next open. Its README reported "an RMSE of 0.08" and "strong trend accuracy". [V1_AUDIT.md](V1_AUDIT.md) shows that the notebook never computed those numbers, and that the saved run was trained on a newest-first CSV. This file answers the next question: if I take the same model and evaluate it properly, what does it actually do?

## What I ran

`nvquant.models.v1_replica` rebuilds the model in PyTorch with the same architecture: four 100-unit LSTM layers whose candidate and cell-output activations are ReLU (what Keras's `LSTM(activation='relu')` does), dropout of 0.2 between layers, a dense output, Adam, MSE, 100 epochs, and batch size 32. It has the same 282,101 parameters. PyTorch's built-in LSTM is tanh-only, so the cell is written by hand.

The only things I changed are the ones the audit found broken:

- time runs forward (the schema check rejects descending dates);
- the MinMax scaler is fit on each training window only;
- every out-of-sample day gets a forecast (not 20 of 33);
- the model is refit once a year on the trailing 252 sessions, and each refit only uses targets that were known at the time;
- the forecast is compared with persistence (tomorrow's open equals today's open), with a Diebold-Mariano test.

The out-of-sample period is the development period shared by every v2 model. It stops at 2024-12-31, so the v1 test window (early 2025) stays in the lockbox.

## Results

<!-- V1:START -->
| Metric | v1 replica | Persistence |
|---|---|---|
| Out-of-sample days | 3887 | 3887 |
| Median absolute error ($) | 0.754 | 0.023 |
| RMSE ($) | 1.91e+09 | 0.987 |
| RMSE in v1's scaled units | 1.39e+08 | 0.0577 |

- Worse than persistence on 92.2% of days (Diebold-Mariano stat 3.97, p = 0.0001).
- On 49.0% of days the latest open was above the training window's maximum, so the MinMax-scaled input was above 1.
- On 10.6% of days the forecast was more than 10 times the training maximum. The largest absolute forecast was 3.58e+10 for a stock whose highest open in the period was 148.95.
- Direction accuracy 49.1%, while 52.9% of days were up. The IC of the implied return is -0.009.
- Parameters: 282,101.

| Refit | Training windows | Training range of the open ($) | Final training MSE (scaled) |
|---|---|---|---|
| 2009-07-22 | 252 | 0.14 to 0.57 | 0.0024 |
| 2010-07-22 | 252 | 0.19 to 0.43 | 0.0227 |
| 2011-07-21 | 252 | 0.20 to 0.59 | 0.0207 |
| 2012-07-20 | 252 | 0.26 to 0.47 | 0.0371 |
| 2013-07-24 | 252 | 0.26 to 0.35 | 0.0615 |
| 2014-07-24 | 252 | 0.31 to 0.47 | 0.0051 |
| 2015-07-24 | 252 | 0.40 to 0.56 | 0.1626 |
| 2016-07-25 | 252 | 0.46 to 1.35 | 0.0392 |
| 2017-07-25 | 252 | 0.85 to 4.15 | 0.1151 |
| 2018-07-25 | 252 | 2.53 to 6.54 | 0.0388 |
| 2019-07-26 | 252 | 3.13 to 7.15 | 0.0115 |
| 2020-07-27 | 252 | 3.37 to 10.53 | 0.0037 |
| 2021-07-27 | 252 | 6.98 to 20.76 | 0.0164 |
| 2022-07-27 | 252 | 13.50 to 33.37 | 0.0597 |
| 2023-07-28 | 252 | 10.93 to 47.32 | 0.0050 |
| 2024-07-30 | 252 | 27.56 to 139.41 | 0.0056 |
<!-- V1:END -->

## Why it fails

1. **Scaling.** MinMax maps the training window to [0, 1]. NVDA's price trends, so the next year's opens are often above the training maximum, and the network sees inputs it was never trained on. The table above shows how often that happened.
2. **ReLU in a recurrent cell.** With tanh the cell state is bounded. With ReLU, an input above the usual range makes the cell state grow step by step across 60 steps, and four stacked layers compound it. The training loss looks fine in every refit (the last column above), because training inputs are inside [0, 1]. The blow-ups only appear out of sample.
3. **Price levels.** Even where the forecast stays finite, predicting a price level mostly means predicting "about today's price". Persistence does that exactly, with no parameters.

## What v2 does instead

- Targets are returns (vol-normalized), which are close to stationary, and features are returns, ratios, or trailing z-scores, so nothing has to extrapolate a level.
- The recurrent models are small (hidden size 16), use standard tanh cells, and are trained with early stopping and gradient clipping.
- Every model is compared with naive baselines (zero, historical mean, AR) and, as a strategy, with vol-targeted buy and hold, net of costs.
