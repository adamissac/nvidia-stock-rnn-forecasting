"""Model registry: config ``kind`` to constructor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nvquant.config.schema import Config, ModelConfig
from nvquant.models.base import Forecaster
from nvquant.models.baselines import ARForecaster, HistMeanForecaster, ZeroForecaster
from nvquant.models.deep.train import DeepForecaster
from nvquant.models.linear import ElasticNetForecaster, RidgeForecaster
from nvquant.models.trees import LGBMForecaster

Builder = Callable[[dict[str, Any], Config], Forecaster]


def _deep(arch: str) -> Builder:
    def build(params: dict[str, Any], cfg: Config) -> Forecaster:
        p = dict(params)
        head = p.pop("head", "gaussian")
        return DeepForecaster(arch=arch, head=head, device=cfg.device, base_seed=cfg.seed, **p)

    return build


MODEL_REGISTRY: dict[str, Builder] = {
    "zero": lambda p, c: ZeroForecaster(),
    "hist_mean": lambda p, c: HistMeanForecaster(),
    "ar": lambda p, c: ARForecaster(**p),
    "ridge": lambda p, c: RidgeForecaster(embargo=c.cv.embargo, **p),
    "elastic_net": lambda p, c: ElasticNetForecaster(embargo=c.cv.embargo, **p),
    "lgbm": lambda p, c: LGBMForecaster(seed=c.seed, n_jobs=1, **p),
    "rnn": _deep("rnn"),
    "tcn": _deep("tcn"),
    "patchtst": _deep("patchtst"),
}

BASELINES = ("zero", "hist_mean", "ar")


def build_model(mcfg: ModelConfig, cfg: Config, params: dict[str, Any] | None = None) -> Forecaster:
    """Instantiate a model from its config entry (``params`` overrides, e.g. tuned values)."""
    if mcfg.kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown model kind {mcfg.kind}; known: {sorted(MODEL_REGISTRY)}")
    merged = {**mcfg.params, **(params or {})}
    return MODEL_REGISTRY[mcfg.kind](merged, cfg)
