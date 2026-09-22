# ADR 0002: A lockbox from 2025-01-01, enforced by truncation

**Status:** accepted (2026-09-21)

**Context.** Every look at data I later report as out-of-sample makes it less out-of-sample. Walk-forward alone doesn't stop me from iterating on the whole history.

**Decision.** Every series is cut at 2024-12-31 when the development pipeline loads data (`mode="dev"`). The lockbox run is its own stage. It refuses to run unless docs/PREREGISTRATION.md and configs/preregistration.yaml are committed and unchanged, writes a sentinel, and refuses a second run without `--force` and a written reason. The data-quality report is the only other code that reads post-2024 rows, and it computes no performance.

**Consequences.** Design choices made during development can't be tuned to 2025. The 2025 stress windows are reported only from the single lockbox run.
