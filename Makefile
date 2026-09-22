# nvquant make targets. PROFILE=fast runs on synthetic data (CI); PROFILE=full uses real data.
PROFILE ?= full
CONFIG  := configs/$(PROFILE).yaml
RUN     := uv run
NV      := $(RUN) nvquant

.PHONY: setup data data-report features train backtest evaluate report reproduce lockbox \
        test lint typecheck ci app smoke clean-fast

setup:
	uv sync
	$(RUN) python scripts/macos_libomp.py
	$(RUN) pre-commit install

data:
	$(NV) data --config $(CONFIG)

data-report:
	$(NV) data-report --config $(CONFIG)

features:
	$(NV) features --config $(CONFIG)

train:
	$(NV) train --config $(CONFIG)

backtest:
	$(NV) backtest --config $(CONFIG)

evaluate:
	$(NV) evaluate --config $(CONFIG)

report:
	$(NV) report --config $(CONFIG)

reproduce:
	$(NV) reproduce --config $(CONFIG)

lockbox:
	$(NV) lockbox --config configs/full.yaml

test:
	$(RUN) pytest

lint:
	$(RUN) ruff format --check .
	$(RUN) ruff check .
	$(RUN) python tools/leakage_lint.py src/nvquant

typecheck:
	$(RUN) mypy

smoke:
	$(MAKE) reproduce PROFILE=fast

ci: lint typecheck test smoke

app:
	$(RUN) streamlit run app/streamlit_app.py

clean-fast:
	rm -rf data/fast reports/fast
