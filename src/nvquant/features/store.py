"""Materialized feature and label store (parquet) under ``<data_dir>/features``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from nvquant.config import config_hash
from nvquant.config.schema import Config
from nvquant.data.loader import LoadedData, resolve
from nvquant.experiments.repro import git_sha
from nvquant.features.fitted import FracDiffFeature, RegimeFeature, walk_forward_fitted
from nvquant.features.registry import build_features, column_metadata
from nvquant.labels.forward import forward_returns
from nvquant.labels.triple_barrier import triple_barrier
from nvquant.labels.weights import average_uniqueness
from nvquant.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FeatureStore:
    """Everything the models need, on one session index."""

    features: pd.DataFrame
    labels: pd.DataFrame
    metadata: pd.DataFrame
    info: dict[str, object]

    @property
    def feature_columns(self) -> list[str]:
        """Model input columns."""
        return list(self.features.columns)


def store_dir(cfg: Config, mode: str = "dev") -> Path:
    """``<data_dir>/features`` (dev) or ``<data_dir>/features_lockbox``."""
    name = "features" if mode == "dev" else "features_lockbox"
    return resolve(cfg.paths.data_dir) / name


def build_store(cfg: Config, loaded: LoadedData) -> FeatureStore:
    """Compute static features, walk-forward fitted features, and labels."""
    market = loaded.market
    start = loaded.modeling_start
    static = build_features(market, cfg.features)

    fitted_frames = []
    fit_records = []
    fd = FracDiffFeature(cfg.features.fracdiff)
    frame, recs = walk_forward_fitted(
        market, fd, start, cfg.features.fracdiff.min_train, cfg.features.fracdiff.refit_every
    )
    fitted_frames.append(frame)
    fit_records += recs
    rg = RegimeFeature(cfg.regime, seed=cfg.seed)
    frame, recs = walk_forward_fitted(
        market, rg, start, cfg.regime.min_train, cfg.regime.refit_every
    )
    fitted_frames.append(frame)
    fit_records += recs
    features = pd.concat([static, *fitted_frames], axis=1).loc[start:]

    fwd = forward_returns(market, cfg.labels.horizons, cfg.labels.vol_span)
    tb = triple_barrier(market, cfg.labels.triple_barrier)
    labels = pd.concat([fwd, tb], axis=1).loc[start:]
    labels["uniq_1"] = average_uniqueness(
        pd.Series(labels.index, index=labels.index), labels["t_end_1"], market.sessions
    )
    labels["uniq_tb"] = average_uniqueness(
        pd.Series(labels.index, index=labels.index), labels["tb_t_end"], market.sessions
    )

    meta = column_metadata(market, cfg.features)
    fitted_meta = pd.DataFrame(
        [
            {"column": "fracdiff_logp", "spec": "fracdiff", "group": "fitted", "lookback": cfg.features.fracdiff.max_width,
             "lag": 0, "known_ahead": 0, "description": "Fixed-width fracdiff of log close; d = min grid value passing ADF on the training window."},
            {"column": "regime_p_high", "spec": "regime", "group": "fitted", "lookback": 0, "lag": 0, "known_ahead": 0,
             "description": "Forward-filtered probability of the high-volatility Gaussian HMM state."},
        ]
    )  # fmt: skip
    meta = pd.concat([meta, fitted_meta], ignore_index=True)
    first_complete = features.dropna().index.min()
    info: dict[str, object] = {
        "mode": loaded.mode,
        "modeling_start": str(start.date()),
        "first_complete_row": str(pd.Timestamp(first_complete).date()),
        "end": str(loaded.end.date()),
        "n_rows": len(features),
        "n_features": features.shape[1],
        "data_hash": loaded.data_hash,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "fitted_first_refit": {
            name: min(r.refit_date for r in fit_records if r.feature == name)
            for name in {r.feature for r in fit_records}
        },
        "fit_records": [asdict(r) for r in fit_records],
    }
    return FeatureStore(features, labels, meta, info)


def save_store(store: FeatureStore, cfg: Config, mode: str = "dev") -> Path:
    """Write the store to parquet plus ``info.json``."""
    out = store_dir(cfg, mode)
    out.mkdir(parents=True, exist_ok=True)
    store.features.to_parquet(out / "features.parquet")
    store.labels.to_parquet(out / "labels.parquet")
    store.metadata.to_parquet(out / "metadata.parquet")
    (out / "info.json").write_text(json.dumps(store.info, indent=2) + "\n", encoding="utf-8")
    return out


def load_store(cfg: Config, mode: str = "dev") -> FeatureStore:
    """Read a saved store."""
    d = store_dir(cfg, mode)
    if not (d / "features.parquet").exists():
        raise FileNotFoundError(f"no feature store in {d}; run `make features` first")
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    return FeatureStore(
        pd.read_parquet(d / "features.parquet"),
        pd.read_parquet(d / "labels.parquet"),
        pd.read_parquet(d / "metadata.parquet"),
        info,
    )
