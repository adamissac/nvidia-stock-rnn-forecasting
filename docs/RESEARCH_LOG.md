# Research log

One dated entry per phase: Hypothesis, What ran, Result, Decision. Numbers come from artifacts under `reports/`.

## 2026-09-21: Phase 0, audit and plan

**Hypothesis.** The v1 notebook's headline result (RMSE 0.08) doesn't show that NVDA prices are forecastable, and the repo can't reproduce it.

**What ran.** I re-ran each audit check at commit a686217 on Python 3.12.14 with yfinance 1.7.0 and pandas 3.0.6 (commands are in docs/V1_AUDIT.md). I computed no performance numbers on 2025 dates, since those belong to the lockbox.

**Result.** All 8 findings from the spec reproduce, and I found 8 more. The most important is finding 4: the saved run trained on a newest-first CSV, so the model learned to predict backwards in time. The README metric isn't computed anywhere in the notebook.

**Decision.** Rebuild as a package with the controls listed at the end of docs/V1_AUDIT.md. The owner lifted the Phase 0 approval gate, so I recorded the plan decisions (long/flat headline, CPU only, $10M AUM, 1-session horizon with next-open fills) in docs/PLAN.md and went straight on to Phase 1.
