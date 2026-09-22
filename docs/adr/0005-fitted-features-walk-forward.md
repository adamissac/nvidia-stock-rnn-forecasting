# ADR 0005: Fitted features are computed walk-forward in the feature store

**Status:** accepted (2026-09-21)

**Context.** Fracdiff's `d` and the HMM parameters are estimated from data. Estimating them once on the full sample would leak the future into every row.

**Decision.** They're refit every 252 sessions on an expanding window. The values after refit r come from parameters fit on data up to r and observations up to t. Rows before the first refit use the first fit (burn-in), and every model's first out-of-sample date is strictly after the first refit, so those rows are only ever training rows.

**Consequences.** The feature store depends on the refit schedule, which is part of the config hash. The causality property test covers these features too.
