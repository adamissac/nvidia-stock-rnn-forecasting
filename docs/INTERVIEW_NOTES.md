# Interview notes

These are my own notes for explaining this project. For each component: what it is in plain English, why it exists, which failure it prevents, and the questions I expect, with my answers. The numbers are generated from `reports/results.json`.

## Key numbers

<!-- KEY_NUMBERS:START -->
| Quantity | Value |
|---|---|
| Out-of-sample period | 2009-07-22 to 2024-12-31 (3,884 sessions) |
| Strategy configurations (DSR trials) | 80 |
| Every logged fit | 842 |
| Best configuration | `hist_mean` + voltarget |
| Its net Sharpe [95% CI] | 1.18 [0.68, 1.64] |
| Vol-targeted buy and hold | 1.21 |
| Buy and hold | 1.10 |
| DSR (strategy trials / all fits) | 0.99 / 0.93 |
| PBO | 0.29 |
| SPA p vs vol-targeted buy and hold | 0.15 |
| FF5 + momentum alpha (t) | 24.1% (2.41) |
| Share of the compounded gain from 2023 and 2024 | 83% |
| Share of summed daily returns from 2023 and 2024 | 29% |
<!-- KEY_NUMBERS:END -->

## Resume bullets

<!-- RESUME:START -->
- Built a leakage-tested research pipeline for daily NVDA returns in Python: 57 causal features checked by a perturb-the-future property test, purged walk-forward validation over 3,884 out-of-sample sessions, and two independent backtest engines that agree to 1e-17.
- Evaluated 80 strategy configurations (linear, gradient boosting, LSTM/GRU/TCN/Transformer) net of costs with trial-adjusted statistics (Deflated Sharpe, PBO 0.29, Hansen SPA). Reported the null result: no model beat vol-targeted buy and hold (SPA p = 0.15).
- Audited my earlier LSTM price model: re-run with walk-forward it was worse than a persistence forecast on 92% of days, and I traced the failure to MinMax-scaled inputs leaving the training range.
<!-- RESUME:END -->

## The one-minute version

I rebuilt a tutorial LSTM stock "predictor" into a research pipeline and used it to test whether any model can beat holding NVDA, out of sample, after costs, and after correcting for how many things I tried. None could. The best configuration was a naive baseline with volatility targeting, and none of the machine-learning forecasts beat vol-targeted buy and hold. I report that as the result, with the trial-adjusted statistics that back it up.

## 1. Timing and lookahead

**Plain English.** At every date I only use what I'd have known after that day's close, and I trade at the next morning's open.

**Why it exists.** Most "great" backtests trade on information they couldn't have had.

**Failure it prevents.** Same-bar execution, features built from the future, and labels that overlap the features.

**Likely questions.**
- *How do you know there's no lookahead?* Every feature is registered, and a hypothesis test perturbs all data after a random cutoff and checks that no feature value at or before the cutoff changes. There's also a static lint for negative shifts, centered windows, backfill, and HMM smoothing, which runs on every edit.
- *What's your label exactly?* log(O[t+2] / O[t+1]) for a decision at the close of t. The PnL and the label are the same number, which removes a whole class of alignment bugs.
- *Why lag exogenous data by a day?* FRED Treasury yields really are published a day late. For VIX and ETFs it's conservative, and it keeps the rule easy to audit.

## 2. Validation

**Plain English.** Train on the past, test on the next block, move forward, repeat. Never test on anything the model or its tuning saw.

**Failure it prevents.** Overlapping labels leaking between train and test, and tuning on the test period.

**Likely questions.**
- *What's purging?* Each sample is an interval from the decision to its label's end. A training sample whose interval overlaps the test period is dropped, and so are samples just after it (the embargo), because their features overlap the test labels.
- *Why not normal k-fold?* Shuffled folds put tomorrow in the training set, and overlapping labels leak across fold boundaries.
- *What's CPCV for?* It gives many backtest paths through the same history, so I can see how much the Sharpe varies across paths. I don't use it as the headline, because some of its test groups come before their training data.

## 3. Features

**Plain English.** Returns, five kinds of realized volatility, momentum and reversal, classic technicals, volume, peers and ETFs, VIX and rates, earnings timing, a fractionally differenced price, and an HMM regime probability.

**Likely questions.**
- *What is fractional differentiation?* A way to make a price series stationary while keeping more memory than returns do. I pick the smallest d that passes an ADF test, on training data only.
- *Why write your own HMM forward filter?* Library "posterior probabilities" are smoothed: they use future observations. The forward filter only uses data up to t.
- *Which features mattered?* See the importance section in RESULTS.md. The more honest answer is that importance was unstable across folds, which is what you'd expect when there isn't much signal.

## 4. Models

**Likely questions.**
- *Why so many models if none worked?* To show the null result isn't one model's fault. Linear, trees, and four deep architectures all fail the same benchmark.
- *Why did the deep models do worst?* Daily returns have a tiny signal-to-noise ratio. Flexible models fit noise, and early stopping on a validation split this noisy doesn't fully prevent it.
- *What's the Gaussian NLL head for?* It predicts a mean and a variance, so the model can say when it's less sure. Its intervals were close to their nominal coverage, but the PIT test still rejects exact calibration.

## 5. Backtest and costs

**Likely questions.**
- *How do you know the backtest is right?* Two engines, one vectorized and one an event loop, written separately, must agree to 1e-10 on every strategy. Invariant tests check that a zero signal earns exactly zero, always-long equals buy-and-hold minus one entry cost, higher costs never raise PnL, and positions are lagged.
- *What costs?* Half spread, commission, slippage scaled by vol, square-root impact at $10M AUM, and borrow on shorts. The cost sweep shows how the Sharpe changes from 0 to 20 bps per side.
- *What's the vol-targeted buy and hold for?* It's the key ablation. Many "model" strategies are really just volatility timing. If a model can't beat holding the stock scaled by the same vol forecast, the model isn't adding anything.

## 6. Statistics

**Likely questions.**
- *What's the Deflated Sharpe Ratio?* The probability that the true Sharpe beats the best Sharpe you'd expect from pure luck, given how many strategies I tried and how much their Sharpes varied. The trial count comes from the registry, not from memory.
- *Your best strategy has a high DSR. Isn't that significant?* It's significantly better than zero skill, because NVDA went up a lot. It isn't better than the benchmark that matters. That's what SPA against vol-targeted buy and hold tests, and it doesn't reject.
- *What's PBO?* Split the history in half many ways. How often does the in-sample winner land below the median out of sample? A high PBO means the selection process overfits.
- *Where does the alpha come from?* The factor regression shows a large "alpha" for anything that holds NVDA, because NVDA beat the factors in this sample. That's the selection-bias problem, and it's why the peer study exists.

## 7. Selection bias

**Plain English.** I picked NVDA because I knew it went up. The peer study runs the identical pipeline on ten other chip stocks, including ones that did badly.

**What I'd say.** The method rarely beats vol-targeted buy and hold on the peers either. The count per model is in the "What didn't work" list, and the full table is in RESULTS.md. NVDA's high Sharpe is the stock, not the method.

## 8. The lockbox

**Plain English.** I held back 2025 onward, wrote down in advance which strategies I'd test and which numbers I'd report, committed that, and then ran the lockbox exactly once. The code refuses a second run unless I force it with a written reason, and a forced rerun would be recorded.

## 9. From v1 to v2

v1 predicted price levels with MinMax scaling and a ReLU LSTM, trained on a newest-first CSV, with no baseline and a metric that wasn't computed anywhere. Re-run honestly, it's worse than "tomorrow equals today" on most days, and it blows up once prices leave the training range. The lesson I'd tell an interviewer: the fix wasn't a better model, it was better evaluation.
