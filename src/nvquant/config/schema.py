"""Pydantic schema for nvquant configs.

Every number that can change a result lives here and in ``configs/*.yaml``.
Unknown keys are rejected so a typo in a YAML file fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    """Where artifacts go. Relative paths resolve against the project root."""

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    docs_dir: Path = Path("docs")
    readme: Path = Path("README.md")


class UniverseConfig(_Strict):
    """Tickers. ``target`` is the traded asset; the rest are exogenous inputs."""

    target: str = "NVDA"
    peers: list[str] = Field(
        default_factory=lambda: [
            "AMD",
            "AVGO",
            "TSM",
            "INTC",
            "MU",
            "QCOM",
            "TXN",
            "AMAT",
            "LRCX",
            "ASML",
        ]
    )
    etfs: list[str] = Field(default_factory=lambda: ["SMH", "SOXX", "QQQ", "SPY"])
    indices: list[str] = Field(default_factory=lambda: ["^SOX", "^VIX", "^VIX3M", "^TNX"])
    required: list[str] = Field(
        default_factory=lambda: ["NVDA", "SMH", "QQQ", "SPY", "^VIX", "^VIX3M", "DGS10", "DGS2"],
        description="Inputs whose first valid date sets the modeling start.",
    )

    def all_tickers(self) -> list[str]:
        """Every Yahoo ticker, target first, without duplicates."""
        seen: dict[str, None] = {}
        for t in [self.target, *self.peers, *self.etfs, *self.indices]:
            seen.setdefault(t, None)
        return list(seen)


class SyntheticConfig(_Strict):
    """Size of the synthetic market used by the fast profile and tests."""

    n_sessions: int = 1500
    start: date = date(2015, 1, 2)
    signal_strength: float = 0.0


class DataConfig(_Strict):
    """Data sources and download settings."""

    source: Literal["yahoo", "synthetic"] = "yahoo"
    download_start: date = date(1999, 1, 1)
    fred_series: list[str] = Field(default_factory=lambda: ["DGS10", "DGS2"])
    max_retries: int = 5
    backoff_seconds: float = 2.0
    stale_run_threshold: int = 5
    outlier_sigma: float = 10.0
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)


class LockboxConfig(_Strict):
    """Dates on or after ``start`` are the lockbox."""

    start: date = date(2025, 1, 1)


class FracdiffConfig(_Strict):
    """Fixed-width fractional differentiation settings."""

    d_grid: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    weight_threshold: float = 1e-4
    max_width: int = 252
    adf_pvalue: float = 0.05
    refit_every: int = 252
    min_train: int = 756


class FeaturesConfig(_Strict):
    """Feature store settings."""

    exogenous_lag: int = 1
    earnings_cap: int = 20
    return_horizons: list[int] = Field(default_factory=lambda: [1, 5, 21, 63])
    vol_windows: list[int] = Field(default_factory=lambda: [21, 63])
    beta_window: int = 63
    zscore_window: int = 63
    fracdiff: FracdiffConfig = Field(default_factory=FracdiffConfig)
    exclude: list[str] = Field(default_factory=list)


class TripleBarrierConfig(_Strict):
    """Triple-barrier labeling (Lopez de Prado 2018, ch. 3)."""

    upper_mult: float = 1.0
    lower_mult: float = 1.0
    vertical: int = 10
    vol_span: int = 50


class LabelsConfig(_Strict):
    """Label settings. ``target`` is what return models are trained on."""

    horizons: list[int] = Field(default_factory=lambda: [1, 5, 21])
    target: Literal["fwd_ret_1", "fwd_ret_vn_1"] = "fwd_ret_vn_1"
    vol_span: int = 63
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)


class CVConfig(_Strict):
    """Walk-forward and purged CV settings (in sessions)."""

    scheme: Literal["expanding", "rolling"] = "expanding"
    initial_train: int = 756
    rolling_window: int = 1260
    retrain_every: int = 21
    embargo: int = 5
    kfold_splits: int = 5
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2


class ModelConfig(_Strict):
    """One model entry. ``params`` go to the model constructor."""

    enabled: bool = True
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    retrain_every: int | None = None
    tune: bool = False


class TuningConfig(_Strict):
    """Nested Optuna tuning budget."""

    n_trials: int = 20
    inner_splits: int = 3
    retune_every: int = 252
    timeout_seconds: float = 120.0


class VolConfig(_Strict):
    """Volatility model settings."""

    models: list[str] = Field(
        default_factory=lambda: ["garch", "gjr_t", "egarch", "har", "lgbm_vol", "ewma"]
    )
    refit_every: int = 21
    min_train: int = 756
    sizing_model: str = "gjr_t"


class RegimeConfig(_Strict):
    """Gaussian HMM regime settings."""

    n_states: int = 2
    refit_every: int = 252
    min_train: int = 756
    n_iter: int = 200


class SizingConfig(_Strict):
    """One way to turn a forecast into a position."""

    name: str
    rule: Literal["sign", "scaled", "voltarget", "kelly"]
    long_short: bool = False
    threshold: float = 0.0
    regime_filter: bool = False
    kelly_fraction: float = 0.25


class StrategyConfig(_Strict):
    """Strategy grid and sizing limits."""

    target_vol: float = 0.35
    max_gross: float = 1.5
    scale_window: int = 252
    sizings: list[SizingConfig] = Field(
        default_factory=lambda: [
            SizingConfig(name="sign", rule="sign"),
            SizingConfig(name="scaled", rule="scaled"),
            SizingConfig(name="voltarget", rule="voltarget"),
            SizingConfig(name="kelly", rule="kelly"),
            SizingConfig(name="sign_ls", rule="sign", long_short=True),
            SizingConfig(name="voltarget_regime", rule="voltarget", regime_filter=True),
        ]
    )
    meta_primary: str = "trend_200"
    trend_window: int = 200
    random_null_paths: int = 1000


class CostsConfig(_Strict):
    """Transaction costs, in basis points unless noted."""

    half_spread_bps: float = 1.0
    commission_bps: float = 0.5
    slippage_vol_mult: float = 0.02
    impact_coef: float = 0.1
    aum: float = 1e7
    adv_window: int = 20
    borrow_bps_annual: float = 30.0
    sweep_bps: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    capacity_aum: list[float] = Field(default_factory=lambda: [1e6, 1e7, 1e8, 1e9, 3e9, 1e10, 3e10])


class V1Config(_Strict):
    """The v1 replica (docs/V1_POSTMORTEM.md)."""

    enabled: bool = True
    epochs: int = 100
    train_len: int = 252
    refit_every: int = 252


class EnsembleConfig(_Strict):
    """Which model forecasts feed the ensembles."""

    members: list[str] = Field(
        default_factory=lambda: [
            "ridge", "elastic_net", "lgbm", "lstm_gauss", "lstm_quant",
            "gru_gauss", "gru_quant", "tcn", "patchtst",
        ]
    )  # fmt: skip
    stack_refit_every: int = 21
    stack_min_history: int = 252


class BacktestConfig(_Strict):
    """Execution timing. ``next_open`` (the default and the only one reported) fills at
    the open of t+1; ``next_close`` fills at the close of t+1 (a sensitivity check).
    """

    execution: Literal["next_open", "next_close"] = "next_open"


class EvaluationConfig(_Strict):
    """Statistics settings."""

    n_bootstrap: int = 1000
    spa_reps: int = 1000
    pbo_blocks: int = 16
    mcs_size: float = 0.1
    newey_west_lags: int | None = None
    var_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    var_window: int = 250
    conformal_alpha: float = 0.1
    conformal_gamma: float = 0.005
    mc_paths: int = 1000
    mda_repeats: int = 5
    peer_study: bool = True
    peer_models: list[str] = Field(default_factory=lambda: ["ridge", "lgbm"])
    cpcv_models: list[str] = Field(default_factory=lambda: ["ridge", "lgbm"])
    importance: bool = True
    stress_windows: dict[str, tuple[date, date]] = Field(
        default_factory=lambda: {
            "gfc_2008": (date(2008, 9, 1), date(2009, 3, 9)),
            "q4_2018": (date(2018, 10, 1), date(2018, 12, 24)),
            "covid_2020": (date(2020, 2, 19), date(2020, 3, 23)),
            "drawdown_2022": (date(2022, 1, 3), date(2022, 10, 14)),
        }
    )
    lockbox_stress_windows: dict[str, tuple[date, date]] = Field(
        default_factory=lambda: {
            "deepseek_2025": (date(2025, 1, 24), date(2025, 2, 7)),
            "tariff_2025": (date(2025, 4, 2), date(2025, 4, 21)),
        }
    )


class Config(_Strict):
    """Top-level config."""

    profile: str = "full"
    seed: int = 42
    device: Literal["cpu", "auto"] = "cpu"
    n_jobs: int = 8
    paths: PathsConfig = Field(default_factory=PathsConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    lockbox: LockboxConfig = Field(default_factory=LockboxConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    vol: VolConfig = Field(default_factory=VolConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    v1: V1Config = Field(default_factory=V1Config)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _check(self) -> Config:
        if self.cv.embargo < 0:
            raise ValueError("cv.embargo must be >= 0")
        if self.features.exogenous_lag < 1:
            raise ValueError("features.exogenous_lag must be >= 1 (non-negotiable 1)")
        if max(self.labels.horizons) >= self.cv.initial_train:
            raise ValueError("label horizon must be shorter than the initial training window")
        return self

    def enabled_models(self) -> dict[str, ModelConfig]:
        """Models with ``enabled: true``, in config order."""
        return {k: v for k, v in self.models.items() if v.enabled}
