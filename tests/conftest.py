"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nvquant.config import Config, load_config
from nvquant.config.schema import UniverseConfig
from nvquant.data.market import MarketData
from nvquant.data.synthetic import synthetic_market

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fast_cfg() -> Config:
    return load_config(ROOT / "configs" / "fast.yaml")


@pytest.fixture(scope="session")
def market() -> MarketData:
    return synthetic_market(
        UniverseConfig(), n_sessions=700, start="2018-01-02", signal_strength=0.3, seed=3
    )


@pytest.fixture()
def tmp_cfg(tmp_path: Path) -> Config:
    """Fast profile writing into a temporary directory."""
    return load_config(
        ROOT / "configs" / "fast.yaml",
        overrides={
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "reports_dir": str(tmp_path / "reports"),
                "docs_dir": str(tmp_path / "docs"),
                "readme": str(tmp_path / "README.md"),
            }
        },
    )


def business_index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)
