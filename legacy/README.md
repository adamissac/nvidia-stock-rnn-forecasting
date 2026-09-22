# v1 (legacy)

These are the original project files, moved here with `git mv` so their history is preserved. They are not modified:

- `nvidia_stock_rnn.ipynb`: the Keras stacked-LSTM notebook, with its saved outputs
- `download_data.py`: the download script (still saved as UTF-16LE, which is audit finding 1)
- `requirements.txt`: v1's unpinned dependencies

Nothing in v2 imports from this folder. It's excluded from ruff, mypy, and pytest.

What's wrong with v1 is in [docs/V1_AUDIT.md](../docs/V1_AUDIT.md). The v1 model is re-run through the v2 harness (as a PyTorch replica with the same architecture) and written up in [docs/V1_POSTMORTEM.md](../docs/V1_POSTMORTEM.md).
