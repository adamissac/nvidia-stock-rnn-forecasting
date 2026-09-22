# Methodology

This is how I built and tested every part of v2, with the math written out. Result numbers are not in this file; they're in [RESULTS.md](RESULTS.md), which `make report` generates from `reports/results.json`. The main reference throughout is López de Prado, *Advances in Financial Machine Learning* (2018), cited as AFML with a chapter number.

## 1. Timing: what is known when

Every decision happens after the close of session $t$ and uses only data available then. The order fills at the open of $t+1$ and is held until the open of $t+2$. So the return a decision earns, the 1-session label, and the backtest PnL are all the same quantity:

$$
r_t \;=\; \log\frac{O_{t+2}}{O_{t+1}}, \qquad \text{label end time } \tau_t = t+2 .
$$

Anything that isn't the target's own OHLCV (peers, ETFs, VIX, Treasury yields, factors) is lagged one extra session. For FRED yields this is required: the H.15 release for day $t$ comes out on day $t+1$. For the others it's a conservative default that keeps the rule simple to check.

Data after 2024-12-31 is the lockbox. `load_market_data(mode="dev")` cuts every series at that date before anything else runs, so the development pipeline can't see it even by accident.

## 2. Data

- Yahoo Finance daily bars with `auto_adjust=True` (split and dividend adjusted) and `multi_level_index=False`, for NVDA, ten semiconductor peers, SMH, SOXX, QQQ, SPY, ^SOX, ^VIX, ^VIX3M, and ^TNX.
- FRED DGS10 and DGS2. These are market-observed series that don't get revised, so no ALFRED vintages are needed.
- Ken French daily Fama-French five factors and momentum, used only for attribution.
- Earnings dates from Yahoo. An after-close announcement on day $d$ maps to session $d$, and a before-open one maps to the session before $d$. That way the gap it causes shows up in the next open.

Every series is aligned to the NYSE calendar (`exchange_calendars`) and validated with pandera. The index must be strictly increasing, which catches v1's newest-first bug, and prices must be positive. The data-quality report checks both NVDA splits (4-for-1 on 2021-07-20 and 10-for-1 on 2024-06-10) for continuity in price and volume.

**Modeling start.** The latest first-valid date across the required inputs (NVDA, SMH, QQQ, SPY, ^VIX, ^VIX3M, DGS10, DGS2). ^VIX3M sets it.

**Known data caveats.**
- Dividend adjustment rescales past price *levels* using dividends paid later. Returns and ratios aren't affected (except on ex-dates, and NVDA's dividends are tiny), but a level feature would be. The only level-based feature is fracdiff of log price. Between dividends, the adjustment only adds a constant to log price, and the fracdiff weights sum to a small number, so the effect is negligible.
- Yahoo's earnings dates aren't point-in-time: a date that moved shows up where it ended up. `days_to_earn` is capped at 20 sessions, about how far ahead NVDA announces its date, and the causality test treats it as known 20 sessions ahead.
- The universe is today's list of large semiconductor names. It has survivorship bias, which is one reason the peer study reports every peer, not only the good ones.

## 3. Features

Every feature is a pure function of the data up to $t$, using trailing windows only. The registry records each feature's lookback and lag. `docs/FEATURES.md` is generated from it.

**Realized volatility** over a window of $n$ days, annualized by $\sqrt{252}$. With $o, h, l, c$ as log open, high, low, and close:

$$
\sigma^2_{\text{Parkinson}} = \frac{1}{4n\ln 2}\sum (h-l)^2, \qquad
\sigma^2_{\text{GK}} = \frac{1}{n}\sum \Big[\tfrac12 (h-l)^2 - (2\ln 2 - 1)(c-o)^2\Big],
$$

$$
\sigma^2_{\text{RS}} = \frac{1}{n}\sum \big[(h-o)(h-c) + (l-o)(l-c)\big], \qquad
\sigma^2_{\text{YZ}} = \sigma^2_{\text{overnight}} + k\,\sigma^2_{\text{open-close}} + (1-k)\,\sigma^2_{\text{RS}},
$$

with $k = 0.34 / (1.34 + (n+1)/(n-1))$ (Yang and Zhang 2000).

**Fractional differentiation** (AFML ch. 5). The weights of $(1-B)^d$ are $w_0 = 1$ and $w_k = -w_{k-1}(d-k+1)/k$, truncated when $|w_k| < 10^{-4}$ or after 252 terms (fixed width). At each yearly refit I pick the smallest $d$ on a grid from 0 to 1 whose transformed log price rejects a unit root (ADF $p < 0.05$) on the training window only. The feature then uses that $d$ until the next refit.

**Regimes.** A two-state Gaussian HMM on the daily return and the log Parkinson range, fit by Baum-Welch (`hmmlearn`) on the training window. The feature is the *filtered* probability, which I compute with my own forward recursion:

$$
\alpha_t(j) \;\propto\; p(x_t \mid s_t=j)\sum_i \alpha_{t-1}(i)\,A_{ij}, \qquad \sum_j \alpha_t(j) = 1 .
$$

$\alpha_t$ uses $x_1,\dots,x_t$ only. Smoothed posteriors ($\text{P}(s_t \mid x_{1..T})$) and Viterbi paths use the whole sample, so they would leak the future, and nothing in the package calls them.

**The signature test.** For every registered feature group and a random cutoff $t$, I perturb every value strictly after $t$ and assert that every feature value at or before $t$ is unchanged. Hypothesis runs 50 cutoffs and perturbations per group. The same test covers the walk-forward fitted features and the forward filter.

## 4. Labels and sample weights

- Forward returns $\log(O_{t+1+h}/O_{t+1})$ for $h \in \{1, 5, 21\}$, with end time $t+1+h$.
- Vol-normalized: divided by $\hat\sigma_t\sqrt h$, where $\hat\sigma_t$ is an EWMA (span 63) of daily log returns known at $t$. The return models are trained on the normalized 1-day label, so days with the same number of standard deviations count the same, and their forecasts are multiplied by $\hat\sigma_t$ to get a return.
- Triple barrier (AFML ch. 3): enter at $O_{t+1}$, barriers at $\pm \hat\sigma_t\sqrt{10}$ on the path of later opens, vertical barrier after 10 sessions. The label is which barrier the path touches first. Meta-labels (AFML 3.6) mark whether a primary model's side made money.
- Average uniqueness (AFML ch. 4): with $c_u$ the number of labels covering period $u$, label $i$'s weight is $\bar u_i = \frac{1}{|T_i|}\sum_{u \in T_i} 1/c_u$. The 1-day labels don't overlap, so their weights are 1. The triple-barrier labels overlap heavily, and their weights matter for the meta-labeling classifier.

## 5. Validation

**Walk-forward.** At each refit date $r$ the training set is every sample with $\tau_i < r$, so the label resolved before the first decision in the block. The test block is every decision date from $r$ to the next refit. Linear and tree models refit every 21 sessions, deep models every 126, and the HMM and fracdiff every 252. Every model shares the same first out-of-sample date. That date is at least 756 samples in, and strictly after the first refit of every fitted feature.

**Purged k-fold with embargo** (AFML ch. 7), for inner tuning and feature importance. A training sample $[t_i, \tau_i]$ is dropped if it overlaps the test span, and if it starts within 5 sessions after the test span ends.

**Combinatorial purged CV** (AFML ch. 12). $N=6$ groups with $k=2$ in each test set gives $\binom{6}{2} = 15$ splits and $\varphi = \frac{k}{N}\binom{N}{k} = 5$ complete backtest paths. CPCV trains on data after some of its test groups by design, so I report it as a check on how much the Sharpe varies across paths, never as the headline.

**Nested tuning.** Optuna TPE with 20 trials at each yearly retune. The objective is the mean MSE over 3 purged folds of the training window. The walk-forward test block isn't used.

## 6. Models

All return models share one interface: `fit(X, y, w)` and `predict(X, index)`. Scalers and penalties are fit inside `fit`, on training rows only.

- **Baselines:** zero (random walk), the historical mean (Goyal and Welch 2008), and AR($p$) on lagged daily returns with $p \le 5$ chosen by BIC.
- **Linear:** ridge and elastic net, with the penalty chosen by purged inner CV.
- **Trees:** LightGBM, shallow and heavily regularized, tuned as above.
- **Deep (PyTorch):** one-layer LSTM and GRU (hidden size 16, 20-session input sequence), each with two heads: a Gaussian head trained with the negative log-likelihood $\tfrac12[\log\sigma^2 + (y-\mu)^2/\sigma^2]$, and a quantile head trained with the pinball loss $\sum_q \max(q e, (q-1)e)$ for $q \in \{0.1, 0.5, 0.9\}$, with the three quantiles built so they can't cross. Also a TCN (dilated causal convolutions with residual blocks) and a small PatchTST-style encoder (overlapping time patches, a linear embedding, one Transformer layer). One training loop serves all of them: the last 15% of the training window is the validation set (after a 2-sample gap), early stopping has patience 5, gradients are clipped at norm 1, and 3 seeds are averaged.
- **Ensembles:** the equal-weight mean of the member forecasts, and walk-forward non-negative least-squares stacking on past out-of-sample forecasts whose labels have resolved.
- **v1 replica:** the original architecture (four stacked 100-unit LSTMs with ReLU cell activations and 0.2 dropout, 282,101 parameters) on MinMax-scaled windows of 60 opens, refit yearly on the trailing 252 sessions and compared with persistence. See [V1_POSTMORTEM.md](V1_POSTMORTEM.md).

**Volatility.** The one-step variance forecast for the next session, from EWMA ($\lambda = 0.94$), GARCH(1,1), GJR-GARCH(1,1,1) with Student-$t$ errors, EGARCH(1,1,1), HAR-RV (Corsi 2009, on logs), and LightGBM. GARCH parameters are refit every 21 sessions and held fixed in between, while the variance recursion keeps filtering new returns. The realized proxy is the overnight gap squared plus the Garman-Klass variance of the session. Scoring uses $\text{QLIKE} = \overline{\text{rv}/h - \log(\text{rv}/h) - 1}$, which ranks forecasts consistently even with a noisy proxy (Patton 2011), plus MSE and a Mincer-Zarnowitz regression $\text{rv} = a + b h$ with a HAC Wald test of $(a,b) = (0,1)$. GJR-GARCH-$t$ drives position sizing and VaR.

## 7. Forecast evaluation

- $R^2_{\text{OOS}} = 1 - \sum (y-\hat y)^2 / \sum (y - \bar y_{\text{bench}})^2$ against the zero forecast and against the historical mean (Campbell and Thompson 2008).
- Diebold-Mariano on squared errors with the Harvey-Leybourne-Newbold correction $\text{DM}\cdot\sqrt{(n+1-2h+h(h-1)/n)/n}$, using a $t_{n-1}$ reference.
- Spearman IC. The t-stat is Newey-West on the product of standardized ranks, whose mean equals the Spearman correlation. IC decay is computed against the 1, 5, and 21-session forward returns.
- Pesaran-Timmermann test of directional accuracy.
- Calibration: the PIT histogram with a KS test against uniform, and interval coverage (90% for the Gaussian heads, 80% for the quantile heads).
- Adaptive conformal intervals (Gibbs and Candès 2021), using only residuals whose labels have resolved: $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$ with $\gamma = 0.005$.
- Model confidence set (Hansen, Lunde, and Nason 2011) on squared-error losses, via `arch.bootstrap.MCS`.

## 8. Strategies and backtest

**Sizing rules** (all long/flat unless marked):
- sign: $p_t = \mathbb 1[\hat r_t > 0]$ (and $-1$ below zero for the long/short variant)
- scaled: $p_t = \text{clip}\big(\hat r_t / (2\,\text{sd}_{252}(\hat r)), 0, 1.5\big)$
- vol target: $p_t = \text{sign}(\hat r_t)\cdot\min(0.35/\hat\sigma^{\text{ann}}_t, 1.5)$
- fractional Kelly: $p_t = \text{clip}(0.25\,\hat r_t/\hat\sigma^2_t, 0, 1.5)$
- the regime filter sets $p_t = 0$ when the filtered high-vol probability is above 0.5

**Meta-labeling.** The primary rule goes long when the close is above its 200-day average. A LightGBM classifier, trained walk-forward on the triple-barrier outcomes of the primary's past bets with uniqueness weights, gives $\hat p$. The bet size is $\max(0, 2\Phi(z)-1)$ with $z = (\hat p - 0.5)/\sqrt{\hat p(1-\hat p)}$ (AFML ch. 10).

**Costs** per unit of traded notional:

$$
c_t = \underbrace{1\text{ bp}}_{\text{half spread}} + \underbrace{0.5\text{ bp}}_{\text{commission}} + 0.02\,\hat\sigma_t + 0.1\,\hat\sigma_t\sqrt{\frac{|\Delta p_t|\cdot \text{AUM}}{\text{ADV}_t}},
$$

with AUM of \$10M and ADV the trailing 20-session dollar volume, lagged one session. The last term is the square-root impact law (Almgren et al. 2005, Tóth et al. 2011). Shorts pay 30 bps a year in borrow. Net return:

$$
\text{net}_t = p_t\,(e^{r_t} - 1) - c_t\,|p_t - p_{t-1}| - \tfrac{30\text{ bp}}{252}\max(-p_t, 0).
$$

**Two engines.** A vectorized engine and an event-driven loop are written separately, and every run checks that they agree to $10^{-10}$. The invariant tests check that a zero signal earns exactly zero, always-long equals buy-and-hold minus one entry cost, higher costs never raise PnL, and a position decided at $t$ earns $r_t$ and nothing else.

**Benchmarks:** buy-and-hold NVDA, vol-targeted buy-and-hold NVDA (the ablation that separates the model from plain vol targeting), SMH, QQQ, a 200-day trend rule, and 1,000 random long/flat signals from a two-state Markov chain matched to the best strategy's exposure and switching rate.

## 9. Significance

Sharpe ratios in these formulas are per period (daily), $\widehat{SR} = \bar r / s_r$, with skewness $\gamma_3$ and non-excess kurtosis $\gamma_4$.

- **PSR** (Bailey and López de Prado 2012): $\text{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \gamma_3\widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right)$.
- **DSR** (Bailey and López de Prado 2014): the PSR with $SR^* = \sqrt{V[\widehat{SR}_n]}\left((1-\gamma)\Phi^{-1}(1-\tfrac1N) + \gamma\,\Phi^{-1}(1-\tfrac{1}{Ne})\right)$, where $N$ is the number of strategy configurations I backtested, $V$ is the variance of their Sharpe ratios, and $\gamma$ is the Euler-Mascheroni constant. I report it twice: with $N$ as the strategy count, and with $N$ as every logged fit.
- **MinTRL:** $1 + \left(1 - \gamma_3\widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2\right)\left(\frac{z_{0.95}}{\widehat{SR} - SR^*}\right)^2$ observations.
- **PBO** via CSCV (Bailey, Borwein, López de Prado, and Zhu 2017): split the $T \times N$ return matrix into 16 blocks, and for each of the $\binom{16}{8}$ ways to choose half of them as in-sample, find where the in-sample best lands out of sample. With relative rank $\omega$, $\lambda = \log\frac{\omega}{1-\omega}$, and $\text{PBO} = \Pr(\lambda \le 0)$.
- **Hansen's SPA and White's Reality Check** against buy-and-hold and vol-targeted buy-and-hold, with losses $= -$returns and a stationary bootstrap (1,000 reps).
- **Bootstrap CIs:** stationary bootstrap with the Politis-White optimal block length, 1,000 reps.

## 10. Risk, attribution, and selection bias

- VaR at 95% and 99% four ways: historical, Gaussian, Cornish-Fisher, and GARCH (position times $\hat\sigma_t$ times a unit-variance $t_5$ quantile). Each is backtested with Kupiec's unconditional coverage test and Christoffersen's independence and conditional-coverage tests.
- Named stress windows before 2025 (the 2008 crisis, Q4 2018, February to March 2020, the 2022 drawdown). Any window before the first out-of-sample date is marked as not covered.
- Performance split by the filtered HMM regime and by calendar year, including the share of total return earned in 2023 and 2024.
- Monte Carlo of the equity path with a stationary block bootstrap: quantiles of terminal wealth and drawdown.
- Attribution: OLS of the strategy's excess return on QQQ, on SMH, and on FF5 plus momentum, with Newey-West errors. The factor return for session $t+1$ stands in for the holding period, which straddles two sessions. That biases betas toward zero, not toward false alpha.
- **Peer study:** the same pipeline, with nothing changed, on each of the ten peers over the development period, using the best strategy's sizing rule and the pre-chosen peer models.

## 11. Feature importance

MDA (the increase in out-of-fold MSE when a feature is permuted) on purged 5-fold CV; clustered MDA, where Ward clusters on $\sqrt{(1-\rho)/2}$ are permuted together (López de Prado 2020); MDI as LightGBM gain; exact TreeSHAP via LightGBM's `pred_contrib`; and stability as the mean pairwise Spearman correlation of importance ranks across folds.

## 12. The lockbox

The final configurations and the metrics to report are written in [PREREGISTRATION.md](PREREGISTRATION.md) and `configs/preregistration.yaml` and committed before the lockbox runs. `make lockbox` refuses to run unless both files are committed and unchanged. It writes a sentinel with the timestamp, git SHA, and data hash, and it refuses to run a second time without `--force` and a written reason, which would be appended to the sentinel. The 2025 stress windows (the DeepSeek selloff around 2025-01-27 and the April 2025 tariff shock) come only from this run.
