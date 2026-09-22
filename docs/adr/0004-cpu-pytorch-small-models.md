# ADR 0004: Small PyTorch models on CPU, no TensorFlow

**Status:** accepted (2026-09-21)

**Context.** Daily single-stock returns have a very low signal-to-noise ratio, and the laptop has no CUDA GPU. v1 used Keras.

**Decision.** Every deep model, the v1 replica included, is plain PyTorch with one shared training loop (early stopping, gradient clipping, seed ensembles), and hidden sizes stay at 16 or below. The v1 replica reimplements Keras's `LSTM(activation='relu')` as a custom cell. The default device is CPU, for determinism; CUDA and MPS are detected and logged.

**Consequences.** Adding TensorFlow for one model would break the "no unused frameworks" rule. The full pipeline stays within the laptop budget.
