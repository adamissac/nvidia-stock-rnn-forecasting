"""Pipeline stages called by the CLI. Each stage reads the previous stage's artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nvquant.backtest.benchmarks import random_signals
from nvquant.backtest.costs import CostInputs
from nvquant.backtest.strategies import (
    STRATEGY_SEP,
    StrategySpec,
    benchmark_positions,
    inputs_for,
    meta_voltarget,
    model_positions,
    run_both,
)
from nvquant.backtest.vectorized import run_vectorized
from nvquant.config import Config, ModelConfig, SizingConfig, config_hash
from nvquant.cv.lockbox import LockboxError, check_can_open, mark_completed, record_open
from nvquant.cv.splits import CombinatorialPurgedCV, PurgedKFold
from nvquant.data.loader import LoadedData, download_all, load_market_data, resolve, write_synthetic
from nvquant.data.market import MarketData
from nvquant.data.quality import write_report
from nvquant.evaluation.attribution import align_factors_to_holding, attribution_suite
from nvquant.evaluation.forecast import (
    adaptive_conformal,
    interval_coverage,
    model_confidence_set,
    pit_gaussian,
    pit_quantiles,
    pit_uniformity,
    r2_oos,
    safe_spearman,
    summarize_forecast,
)
from nvquant.evaluation.importance import correlation_clusters, mda_purged, stability
from nvquant.evaluation.metrics import calendar_year_returns, performance, sharpe
from nvquant.evaluation.risk import (
    bootstrap_equity_paths,
    regime_conditional,
    stress_windows,
    var_suite,
)
from nvquant.evaluation.significance import (
    bootstrap_sharpe_ci,
    deflated_sharpe,
    min_track_record_length,
    pbo_cscv,
    psr_from_returns,
    spa_test,
)
from nvquant.experiments.harness import ForecastResult, first_test_date, sample_index, walk_forward
from nvquant.experiments.registry import Registry, Trial
from nvquant.experiments.repro import git_sha
from nvquant.features.docs import render_feature_docs
from nvquant.features.store import FeatureStore, build_store, load_store, save_store
from nvquant.labels.forward import holding_log_return
from nvquant.logging_utils import get_logger
from nvquant.models.ensemble import equal_weight, stacked
from nvquant.models.factory import build_model
from nvquant.models.trees import LGBMForecaster
from nvquant.models.v1_replica import V1Net, n_parameters, run_v1_replica, v1_metrics
from nvquant.models.volatility import all_vol_forecasts, score_vol
from nvquant.portfolio.meta_labeling import meta_label_positions, trend_side
from nvquant.portfolio.sizing import size

log = get_logger(__name__)


def reports_dir(cfg: Config) -> Path:
    """Resolved reports directory for the profile."""
    out = resolve(cfg.paths.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def stage_data(cfg: Config) -> Path:
    """Download (full profile) or generate (fast profile) raw data."""
    raw = write_synthetic(cfg) if cfg.data.source == "synthetic" else download_all(cfg)
    log.info("raw data in %s", raw)
    return raw


def stage_data_report(cfg: Config) -> Path:
    """Data-quality report over every session, lockbox included (data checks only)."""
    path = write_report(cfg, load_market_data(cfg, mode="lockbox"))
    log.info("data-quality report: %s", path)
    return path


def stage_features(cfg: Config) -> Path:
    """Build the development feature store and regenerate docs/FEATURES.md."""
    loaded = load_market_data(cfg, mode="dev")
    store = build_store(cfg, loaded)
    out = save_store(store, cfg)
    docs = resolve(cfg.paths.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "FEATURES.md").write_text(render_feature_docs(store), encoding="utf-8")
    info = {k: v for k, v in store.info.items() if k != "fit_records"}
    records = cast(list[dict[str, Any]], store.info["fit_records"])
    info["fracdiff_d"] = [r["summary"]["d"] for r in records if r["feature"] == "fracdiff"]
    _write_json(reports_dir(cfg) / "features_info.json", info)
    log.info(
        "feature store: %s (%d rows, %d features)",
        out,
        len(store.features),
        store.features.shape[1],
    )
    return out


# ----------------------------------------------------------------------------- train


def run_id() -> str:
    """Identifier shared by every registry row a stage writes."""
    from datetime import UTC, datetime

    from nvquant.experiments.repro import git_sha

    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{git_sha()}"


def registry(cfg: Config) -> Registry:
    """The profile's experiment registry."""
    return Registry(reports_dir(cfg) / "registry" / "trials.jsonl")


def _train_one(name: str, mcfg: ModelConfig, store: FeatureStore, cfg: Config) -> ForecastResult:
    """Worker: seed, restrict threads, run walk-forward for one model."""
    import torch

    from nvquant.experiments.repro import seed_everything

    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    return walk_forward(name, mcfg, store, cfg)


def stage_train(cfg: Config, only: list[str] | None = None) -> Path:
    """Walk-forward forecasts for every enabled model, ensembles, vol models, and the v1 replica."""
    from joblib import Parallel, delayed

    store = load_store(cfg)
    loaded = load_market_data(cfg, mode="dev")
    out = reports_dir(cfg)
    fdir = out / "forecasts"
    fdir.mkdir(exist_ok=True)
    rid = run_id()
    reg = registry(cfg)
    sha = git_sha()
    models = {k: v for k, v in cfg.enabled_models().items() if not only or k in only}
    order = sorted(models, key=lambda k: models[k].kind in ("rnn", "tcn", "patchtst"), reverse=True)
    results = Parallel(n_jobs=min(cfg.n_jobs, len(order)), backend="loky")(
        delayed(_train_one)(k, models[k], store, cfg) for k in order
    )
    base_cfg = {"target": cfg.labels.target, "cv": cfg.cv.model_dump(mode="json")}
    for res in results:
        res.frame.to_parquet(fdir / f"{res.name}.parquet")
        f = res.frame.dropna(subset=["y"])
        reg.log(
            Trial(
                kind="model", name=res.name, run_id=rid, git_sha=sha, data_hash=loaded.data_hash,
                config={**base_cfg, "model": models[res.name].model_dump(mode="json")},
                metrics={"ic": safe_spearman(f["pred"], f["y"]), "n": len(f)},
                fold_metrics=res.fold_metrics, wall_time_s=res.wall_time_s,
            )
        )  # fmt: skip
        for tr in res.tuning_trials:
            reg.log(
                Trial(
                    kind="tuning", name=f"{res.name}_tuning", run_id=rid, git_sha=sha,
                    data_hash=loaded.data_hash, config={"params": tr["params"], "refit_date": tr["refit_date"]},
                    metrics={"inner_mse": tr["inner_mse"], "fold_mse": tr["fold_mse"]},
                    wall_time_s=tr["wall_time_s"],
                )
            )  # fmt: skip
        log.info("trained %s in %.0fs", res.name, res.wall_time_s)
    if not only:
        _train_ensembles(cfg, store, fdir, reg, rid, sha, loaded.data_hash)
        _train_vol(cfg, store, loaded, out, reg, rid, sha)
        if cfg.v1.enabled:
            _train_v1(cfg, store, loaded, out, reg, rid, sha)
    return fdir


def load_forecasts(cfg: Config) -> dict[str, pd.DataFrame]:
    """Every saved forecast frame, keyed by model name."""
    fdir = reports_dir(cfg) / "forecasts"
    return {p.stem: pd.read_parquet(p) for p in sorted(fdir.glob("*.parquet"))}


def _train_ensembles(
    cfg: Config, store: FeatureStore, fdir: Path, reg: Registry, rid: str, sha: str, dh: str
) -> None:
    fc = load_forecasts(cfg)
    members = [m for m in cfg.ensemble.members if m in fc]
    if len(members) < 2:
        return
    P = pd.DataFrame({m: fc[m]["pred_vn"] for m in members})
    y_vn = store.labels[cfg.labels.target]
    t_end = store.labels["t_end_1"]
    stack, weights = stacked(
        P, y_vn, t_end, cfg.ensemble.stack_refit_every, cfg.ensemble.stack_min_history
    )
    for name, pred_vn in (("ens_equal", equal_weight(P)), ("ens_stack", stack)):
        frame = pd.DataFrame({"pred_vn": pred_vn})
        frame["sigma"] = store.labels["sigma"].reindex(frame.index)
        frame["pred"] = frame["pred_vn"] * frame["sigma"]
        frame["y_vn"] = y_vn.reindex(frame.index)
        frame["y"] = store.labels["fwd_ret_1"].reindex(frame.index)
        frame.to_parquet(fdir / f"{name}.parquet")
        f = frame.dropna(subset=["y"])
        reg.log(
            Trial(
                kind="model", name=name, run_id=rid, git_sha=sha, data_hash=dh,
                config={"members": members, "method": name},
                metrics={"ic": safe_spearman(f["pred"], f["y"]), "n": len(f)},
            )
        )  # fmt: skip
    weights.to_parquet(reports_dir(cfg) / "forecasts_stack_weights.parquet")


def _train_vol(
    cfg: Config,
    store: FeatureStore,
    loaded: LoadedData,
    out: Path,
    reg: Registry,
    rid: str,
    sha: str,
) -> None:
    vdir = out / "vol"
    vdir.mkdir(exist_ok=True)
    start = pd.Timestamp(store.info["modeling_start"])
    vf = all_vol_forecasts(loaded.market, store.features, start, cfg.vol, cfg.seed)
    failures = vf.attrs.get("convergence_failures", {})
    vf.to_parquet(vdir / "vol_forecasts.parquet")
    first = first_test_date(store, cfg)
    oos = vf.loc[first:]
    scores = {}
    for m in cfg.vol.models:
        scores[m] = asdict(score_vol(oos["rv_next"], oos[m]))
        scores[m]["convergence_failures"] = len(failures.get(m, []))
        reg.log(
            Trial(kind="vol", name=m, run_id=rid, git_sha=sha, data_hash=loaded.data_hash,
                  config={"vol": cfg.vol.model_dump(mode="json"), "model": m}, metrics=scores[m])
        )  # fmt: skip
    (vdir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")


def _train_v1(
    cfg: Config,
    store: FeatureStore,
    loaded: LoadedData,
    out: Path,
    reg: Registry,
    rid: str,
    sha: str,
) -> None:
    import time

    import torch

    torch.set_num_threads(min(cfg.n_jobs, 8))
    t0 = time.perf_counter()
    first = first_test_date(store, cfg)
    res = run_v1_replica(
        loaded.market.ohlcv()["open"], first, loaded.end, cfg.v1.train_len, cfg.v1.refit_every,
        cfg.v1.epochs, cfg.seed,
    )  # fmt: skip
    vdir = out / "v1"
    vdir.mkdir(exist_ok=True)
    res.frame.to_parquet(vdir / "v1_replica.parquet")
    metrics = v1_metrics(res.frame) | {"n_parameters": n_parameters(V1Net())}
    (vdir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "fits": res.fits}, indent=2) + "\n", encoding="utf-8"
    )
    reg.log(
        Trial(kind="v1_replica", name="v1_replica", run_id=rid, git_sha=sha, data_hash=loaded.data_hash,
              config=cfg.v1.model_dump(mode="json"), metrics=metrics, fold_metrics=res.fits,
              wall_time_s=time.perf_counter() - t0)
    )  # fmt: skip


# -------------------------------------------------------------------------- backtest


@dataclass
class BacktestBundle:
    """Everything the backtest stage produces for one market (dev, lockbox, or a peer)."""

    net: pd.DataFrame
    gross: pd.DataFrame
    positions: pd.DataFrame
    turnover: pd.DataFrame
    specs: dict[str, StrategySpec]
    engine_max_diff: float


def run_strategies(
    cfg: Config,
    market: MarketData,
    store: FeatureStore,
    forecasts: dict[str, pd.DataFrame],
    vol: pd.DataFrame,
    dates: pd.DatetimeIndex,
    include_meta: bool = True,
    only: list[str] | None = None,
) -> BacktestBundle:
    """Backtest every strategy on ``dates`` with both engines."""
    sigma = store.labels["sigma"]
    var_fc = vol[cfg.vol.sizing_model].reindex(dates)
    p_high = store.features["regime_p_high"]
    fc = {k: v.reindex(dates) for k, v in forecasts.items()}
    grid = model_positions(fc, var_fc, p_high, cfg)
    if include_meta:
        side = trend_side(market.ohlcv()["close"], cfg.strategy.trend_window)
        tb = store.labels[["tb_label", "tb_ret", "tb_t_end"]]
        meta_pos, _ = meta_label_positions(
            store.features, side, tb, store.labels["uniq_tb"], dates, cfg.cv.retrain_every, cfg.seed
        )
        t = cfg.universe.target
        grid["meta__raw"] = (StrategySpec("meta__raw", "meta", "meta", "raw", t), meta_pos)
        grid["meta__voltarget"] = (
            StrategySpec("meta__voltarget", "meta", "meta", "voltarget", t),
            meta_voltarget(meta_pos, var_fc, cfg),
        )
    grid.update(benchmark_positions(market, dates, var_fc, cfg))
    if only is not None:
        grid = {k: v for k, v in grid.items() if k in only}
    r_cache: dict[str, pd.Series] = {}
    in_cache: dict[str, CostInputs] = {}
    nets, grosses, poss, turns, specs = {}, {}, {}, {}, {}
    worst = 0.0
    for name, (spec, pos) in grid.items():
        tk = spec.ticker
        if tk not in r_cache:
            r_cache[tk] = holding_log_return(market, cfg.backtest.execution, tk)
            in_cache[tk] = inputs_for(market, sigma, cfg, tk)
        res, diff = run_both(pos.reindex(dates).fillna(0.0), r_cache[tk], in_cache[tk], cfg)
        worst = max(worst, diff)
        nets[name], grosses[name], poss[name], turns[name] = (
            res.net,
            res.gross,
            res.position,
            res.turnover,
        )
        specs[name] = spec
    return BacktestBundle(
        pd.DataFrame(nets),
        pd.DataFrame(grosses),
        pd.DataFrame(poss),
        pd.DataFrame(turns),
        specs,
        worst,
    )


def cost_sweep(
    cfg: Config, market: MarketData, store: FeatureStore, bundle: BacktestBundle
) -> pd.DataFrame:
    """Net annualized Sharpe of every strategy at flat per-side costs from 0 to 20 bps."""
    rows = {}
    sigma = store.labels["sigma"]
    for name, spec in bundle.specs.items():
        r = holding_log_return(market, cfg.backtest.execution, spec.ticker)
        inputs = inputs_for(market, sigma, cfg, spec.ticker)
        rows[name] = {
            f"{b:g}": sharpe(
                run_vectorized(bundle.positions[name], r, inputs, cfg.costs, flat_bps=b).net
            )
            for b in cfg.costs.sweep_bps
        }
    return pd.DataFrame(rows).T


def capacity_sweep(
    cfg: Config, market: MarketData, store: FeatureStore, bundle: BacktestBundle, names: list[str]
) -> pd.DataFrame:
    """Net Sharpe of selected strategies as assumed AUM grows (square-root impact)."""
    r = holding_log_return(market, cfg.backtest.execution)
    inputs = inputs_for(market, store.labels["sigma"], cfg, cfg.universe.target)
    rows = {}
    for n in names:
        rows[n] = {
            f"{a:g}": sharpe(run_vectorized(bundle.positions[n], r, inputs, cfg.costs, aum=a).net)
            for a in cfg.costs.capacity_aum
        }
    return pd.DataFrame(rows).T


def oos_dates(
    store: FeatureStore, forecasts: dict[str, pd.DataFrame], vol: pd.DataFrame, cfg: Config
) -> pd.DatetimeIndex:
    """Decision dates where every forecast and the sizing vol forecast exist."""
    idx = pd.DatetimeIndex(vol[cfg.vol.sizing_model].dropna().index)
    for f in forecasts.values():
        idx = idx.intersection(f["pred"].dropna().index)
    return idx[idx >= first_test_date(store, cfg)]


def stage_backtest(cfg: Config) -> Path:
    """Backtest every strategy on the development OOS dates and register each one as a trial."""
    store = load_store(cfg)
    loaded = load_market_data(cfg, mode="dev")
    out = reports_dir(cfg) / "backtest"
    out.mkdir(exist_ok=True)
    forecasts = load_forecasts(cfg)
    vol = pd.read_parquet(reports_dir(cfg) / "vol" / "vol_forecasts.parquet")
    dates = oos_dates(store, forecasts, vol, cfg)
    bundle = run_strategies(cfg, loaded.market, store, forecasts, vol, dates)
    if bundle.engine_max_diff > 1e-10:
        raise RuntimeError(f"engines disagree by {bundle.engine_max_diff:.2e}")
    for key, frame in (
        ("net", bundle.net),
        ("gross", bundle.gross),
        ("positions", bundle.positions),
        ("turnover", bundle.turnover),
    ):
        frame.to_parquet(out / f"{key}.parquet")
    specs = {k: asdict(v) for k, v in bundle.specs.items()}
    cost_sweep(cfg, loaded.market, store, bundle).to_parquet(out / "cost_sweep.parquet")
    model_names = [k for k, v in bundle.specs.items() if v.kind != "benchmark"]
    best = max(model_names, key=lambda k: sharpe(bundle.net[k]))
    capacity_sweep(
        cfg, loaded.market, store, bundle, [best, "bh_voltarget", "bh_target"]
    ).to_parquet(out / "capacity.parquet")
    (out / "strategies.json").write_text(
        json.dumps({"specs": specs, "engine_max_abs_diff": bundle.engine_max_diff,
                    "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()), "n_dates": len(dates)}, indent=2) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    reg = registry(cfg)
    rid = run_id()
    sha = git_sha()
    for name in model_names:
        reg.log(
            Trial(
                kind="strategy", name=name, run_id=rid, git_sha=sha, data_hash=loaded.data_hash,
                config={"spec": specs[name], "costs": cfg.costs.model_dump(mode="json"),
                        "strategy": cfg.strategy.model_dump(mode="json"), "execution": cfg.backtest.execution},
                metrics={"sharpe_net": sharpe(bundle.net[name]), "sharpe_gross": sharpe(bundle.gross[name]),
                         "sharpe_net_daily": sharpe(bundle.net[name], annualize=False), "n": int(bundle.net[name].notna().sum())},
            )
        )  # fmt: skip
    log.info(
        "backtested %d strategies on %d dates; engines agree to %.1e",
        len(bundle.specs),
        len(dates),
        bundle.engine_max_diff,
    )
    return out


# -------------------------------------------------------------------------- evaluate


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o: object) -> object:
    import numpy as np

    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return str(o)


def evaluate_forecasts(
    cfg: Config, store: FeatureStore, forecasts: dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> dict[str, object]:
    """Per-model forecast statistics on the common OOS dates, plus the MCS."""
    import numpy as np

    y = store.labels["fwd_ret_1"]
    horizons = {h: store.labels[f"fwd_ret_{h}"] for h in cfg.labels.horizons}
    hist = forecasts.get("hist_mean")
    out: dict[str, object] = {}
    losses = {}
    for name, f in forecasts.items():
        f = f.reindex(dates)
        s = summarize_forecast(
            f["pred"], y.reindex(dates), {h: r.reindex(dates) for h, r in horizons.items()}
        )
        if hist is not None:
            both = pd.concat(
                [f["pred"], hist["pred"].reindex(dates), y.reindex(dates)], axis=1
            ).dropna()
            s["r2_oos_vs_hist_mean"] = r2_oos(
                both.iloc[:, 2], both.iloc[:, 0], both.iloc[:, 1].to_numpy()
            )
        conf = adaptive_conformal(
            y.reindex(dates).to_numpy(),
            f["pred"].to_numpy(),
            cfg.evaluation.conformal_alpha,
            cfg.evaluation.conformal_gamma,
        )
        s["conformal"] = {
            "target": 1 - cfg.evaluation.conformal_alpha,
            "coverage": conf.coverage,
            "mean_width": conf.mean_width,
        }
        if "std" in f:
            ok = f[["pred", "std"]].notna().all(axis=1) & y.reindex(dates).notna()
            pit = pit_gaussian(y.reindex(dates)[ok], f["pred"][ok], f["std"][ok])
            z = 1.6448536269514722
            s["calibration"] = {
                "pit": pit_uniformity(pit),
                "coverage_90": interval_coverage(
                    y.reindex(dates)[ok],
                    f["pred"][ok] - z * f["std"][ok],
                    f["pred"][ok] + z * f["std"][ok],
                ),
            }
        if "q0.1" in f:
            ok = f[["q0.1", "q0.5", "q0.9"]].notna().all(axis=1) & y.reindex(dates).notna()
            qs = f.loc[ok, ["q0.1", "q0.5", "q0.9"]].to_numpy()
            pit = pit_quantiles(y.reindex(dates)[ok].to_numpy(), qs, (0.1, 0.5, 0.9))
            s["calibration"] = {
                "pit": pit_uniformity(pit),
                "coverage_80": interval_coverage(
                    y.reindex(dates)[ok], f["q0.1"][ok], f["q0.9"][ok]
                ),
            }
        out[name] = s
        losses[name] = (y.reindex(dates) - f["pred"]) ** 2
    # No fallback: the loss matrix comes from this pipeline, so an MCS failure means
    # something upstream is wrong, and an empty result would read as "every model survived".
    L = pd.DataFrame(losses).dropna()
    mcs = model_confidence_set(L * 1e4, cfg.evaluation.mcs_size, cfg.evaluation.spa_reps, cfg.seed)
    return {
        "models": out,
        "mcs_pvalues": mcs,
        "mcs_size": cfg.evaluation.mcs_size,
        "n_dates": len(dates),
    }


def evaluate_strategies(
    cfg: Config,
    bundle: BacktestBundle,
    reg_rows: list[dict[str, object]],
    r_hold: pd.Series,
    inputs: CostInputs,
) -> dict[str, object]:
    """Metrics, bootstrap CIs, PSR/DSR/MinTRL, PBO, SPA/RC, and the random-signal null."""
    import numpy as np

    net = bundle.net
    model_names = [k for k, v in bundle.specs.items() if v.kind != "benchmark"]
    bench_names = [k for k, v in bundle.specs.items() if v.kind == "benchmark"]
    trial_srs = np.array([sharpe(net[k], annualize=False) for k in model_names])
    n_trials = len(model_names)
    n_all_fits = len(reg_rows)
    rows: dict[str, dict[str, object]] = {}
    for name in [*model_names, *bench_names]:
        r = net[name].dropna()
        m: dict[str, object] = dict(performance(r, bundle.positions[name], bundle.turnover[name]))
        m["sharpe_gross"] = sharpe(bundle.gross[name])
        lo, hi, block = bootstrap_sharpe_ci(r, cfg.evaluation.n_bootstrap, cfg.seed)
        m.update({"sharpe_ci_lo": lo, "sharpe_ci_hi": hi, "block_length": block})
        m["psr_0"] = psr_from_returns(r.to_numpy())
        d = deflated_sharpe(r.to_numpy(), n_trials, trial_srs)
        m["dsr"] = d.dsr
        m["sr_star_daily"] = d.sr_star
        m["dsr_all_fits"] = deflated_sharpe(r.to_numpy(), max(n_all_fits, n_trials), trial_srs).dsr
        m["min_trl_days"] = min_track_record_length(r.to_numpy())
        m["kind"] = bundle.specs[name].kind
        rows[name] = m
    best = max(model_names, key=lambda k: float(rows[k]["sharpe"]))  # type: ignore[arg-type]
    pbo = pbo_cscv(net[model_names], cfg.evaluation.pbo_blocks)
    spa = {
        b: asdict(spa_test(net[b], net[model_names], cfg.evaluation.spa_reps, cfg.seed))
        for b in ("bh_target", "bh_voltarget")
        if b in net
    }
    pos = bundle.positions[best]
    long_share = float((pos > 0).mean())
    switch = float(((pos > 0).astype(int).diff().abs() > 0).mean())
    paths = random_signals(pos.index, switch, long_share, cfg.strategy.random_null_paths, cfg.seed)
    # random long/flat signals scaled like the best strategy's average gross exposure
    # Random long/flat timing with the best strategy's exposure and switching rate, sized with
    # the same vol-target leverage path and charged the same cost model. Beating this means the
    # *timing* adds something beyond vol targeting.
    lev = bundle.positions["bh_voltarget"].reindex(pos.index).fillna(0.0)
    null_srs = []
    for p in paths:
        pp = pd.Series(p, index=pos.index) * lev
        null_srs.append(sharpe(run_vectorized(pp, r_hold, inputs, cfg.costs).net))
    null = np.asarray(null_srs)
    return {
        "strategies": rows,
        "best": best,
        "n_trials": n_trials,
        "n_all_fits": n_all_fits,
        "trial_sharpe_variance_daily": float(np.var(trial_srs, ddof=1)),
        "pbo": {"pbo": pbo.pbo, "n_combinations": pbo.n_combinations, "degradation_slope": pbo.perf_degradation_slope,
                "prob_oos_loss": pbo.prob_oos_loss, "logits": pbo.logits.tolist()},
        "spa": spa,
        "random_null": {"n_paths": len(null), "sharpe_p50": float(np.nanmedian(null)), "sharpe_p95": float(np.nanquantile(null, 0.95)),
                        "best_percentile": float(np.nanmean(null < float(rows[best]["sharpe"]))), "long_share": long_share, "switch_rate": switch,
                        "sizing": "random long/flat direction times the vol-targeted buy-and-hold leverage path, full cost model"},
    }  # fmt: skip


def evaluate_risk(
    cfg: Config,
    market: MarketData,
    store: FeatureStore,
    bundle: BacktestBundle,
    vol: pd.DataFrame,
    best: str,
) -> dict[str, object]:
    """Attribution, VaR backtests, stress windows, regimes, calendar years, and Monte Carlo paths."""
    import numpy as np

    net = bundle.net
    dates = net.index
    sessions = market.sessions
    rf = align_factors_to_holding(market.factors[["rf"]], dates, sessions)["rf"]
    ff = align_factors_to_holding(market.factors, dates, sessions)
    etf = pd.DataFrame(
        {
            e: np.expm1(holding_log_return(market, cfg.backtest.execution, e)).reindex(dates)
            for e in ("QQQ", "SMH")
            if e in market.prices
        }
    )
    focus = [best, "bh_voltarget", "bh_target"]
    attribution = {n: attribution_suite(net[n], rf, etf, ff) for n in focus}
    var_fc = vol[cfg.vol.sizing_model].reindex(dates)
    risk = var_suite(
        net[best],
        bundle.positions[best],
        var_fc,
        cfg.evaluation.var_levels,
        cfg.evaluation.var_window,
    )
    stress_cols = [
        c
        for c in [
            best,
            "bh_target",
            "bh_voltarget",
            "bh_smh",
            "bh_qqq",
            f"trend_{cfg.strategy.trend_window}",
        ]
        if c in net
    ]
    stress = stress_windows(net[stress_cols], cfg.evaluation.stress_windows)
    regimes = {n: regime_conditional(net[n], store.features["regime_p_high"]) for n in focus}
    years = {n: calendar_year_returns(net[n]).to_dict() for n in focus}
    pnl = net[best]
    years_idx = pd.DatetimeIndex(pnl.index).year
    share_23_24 = (
        float(pnl[(years_idx >= 2023) & (years_idx <= 2024)].sum() / pnl.sum())
        if pnl.sum() != 0
        else float("nan")
    )
    block = bootstrap_sharpe_ci(pnl.dropna(), 10, cfg.seed)[2]
    mc = {
        n: bootstrap_equity_paths(net[n], cfg.evaluation.mc_paths, cfg.seed, block) for n in focus
    }
    return {
        "attribution": attribution, "var": risk, "stress": stress, "regimes": regimes,
        "calendar_years": {n: {str(k): v for k, v in y.items()} for n, y in years.items()},
        "pnl_share_2023_2024": share_23_24, "monte_carlo": mc,
    }  # fmt: skip


def peer_config(cfg: Config, peer: str) -> Config:
    """Same config with ``peer`` as the traded ticker and NVDA moved into the peer list."""
    u = cfg.universe
    peers = [p for p in [u.target, *u.peers] if p != peer]
    required = [peer if r == u.target else r for r in u.required]
    return cfg.model_copy(
        update={
            "universe": u.model_copy(update={"target": peer, "peers": peers, "required": required})
        }
    )


def _run_peer(
    cfg: Config, loaded: LoadedData, peer: str, models: list[str], sizing: str
) -> dict[str, object]:
    """The identical pipeline, unchanged, on one peer over the development period."""
    import torch

    from nvquant.experiments.repro import seed_everything

    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    pc = peer_config(cfg, peer)
    m = loaded.market.with_target(peer)
    first_valid = m.ohlcv()["close"].first_valid_index()
    start = max(loaded.modeling_start, pd.Timestamp(first_valid) + pd.Timedelta(days=1))
    pl = replace(loaded, market=m, modeling_start=start)
    store = build_store(pc, pl)
    fc = {name: walk_forward(name, pc.models[name], store, pc).frame for name in models}
    vol = all_vol_forecasts(
        m,
        store.features,
        start,
        pc.vol.model_copy(update={"models": [pc.vol.sizing_model]}),
        pc.seed,
    )
    dates = oos_dates(store, fc, vol, pc)
    names = [f"{mm}{STRATEGY_SEP}{sizing}" for mm in models] + ["bh_target", "bh_voltarget"]
    bundle = run_strategies(pc, m, store, fc, vol, dates, include_meta=False, only=names)
    return {
        "peer": peer,
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "n": len(dates),
        "sharpe": {n: sharpe(bundle.net[n]) for n in names},
        "ic": {
            mm: safe_spearman(
                fc[mm]["pred"].reindex(dates), store.labels["fwd_ret_1"].reindex(dates)
            )
            for mm in models
        },
    }


def evaluate_peers(cfg: Config, loaded: LoadedData, best: str) -> dict[str, object]:
    """Selection-bias check: run the finalist's sizing with the peer models on every peer."""
    from joblib import Parallel, delayed

    model, sizing = best.split(STRATEGY_SEP)
    models = list(
        dict.fromkeys([*cfg.evaluation.peer_models, *([model] if model in cfg.models else [])])
    )
    rows = Parallel(n_jobs=min(cfg.n_jobs, len(cfg.universe.peers)), backend="loky")(
        delayed(_run_peer)(cfg, loaded, p, models, sizing) for p in cfg.universe.peers
    )
    return {"sizing": sizing, "models": models, "peers": rows}


def evaluate_importance(cfg: Config, store: FeatureStore) -> dict[str, object]:
    """TreeSHAP, MDA (single and clustered) under purged k-fold, MDI, and stability across folds."""
    import numpy as np

    idx = sample_index(store, cfg.labels.target)
    X = store.features.loc[idx]
    y = store.labels[cfg.labels.target].loc[idx]
    t_end = store.labels["t_end_1"].loc[idx]
    params = {k: v for k, v in cfg.models["lgbm"].params.items()}
    singles = {c: [c] for c in X.columns}
    clusters = correlation_clusters(X)
    mda, mdi = mda_purged(X, y, t_end, pd.DatetimeIndex(store.features.index), singles, params,
                          cfg.cv.kfold_splits, cfg.cv.embargo, cfg.evaluation.mda_repeats, cfg.seed)  # fmt: skip
    cmda, _ = mda_purged(X, y, t_end, pd.DatetimeIndex(store.features.index), clusters, params,
                         cfg.cv.kfold_splits, cfg.cv.embargo, cfg.evaluation.mda_repeats, cfg.seed)  # fmt: skip
    shap_rows = []
    for f in PurgedKFold(cfg.cv.kfold_splits, cfg.cv.embargo).split(
        t_end, pd.DatetimeIndex(store.features.index)
    ):
        m = LGBMForecaster(seed=cfg.seed, **params).fit(X, y.iloc[f.train])
        shap_rows.append(m.shap_values(X, idx[f.test]).abs().mean())
    shap = pd.DataFrame(shap_rows)
    return {
        "mda": mda.mean().sort_values(ascending=False).to_dict(),
        "mda_se": (mda.std() / np.sqrt(len(mda))).to_dict(),
        "mdi": mdi.mean().sort_values(ascending=False).to_dict(),
        "shap": shap.mean().sort_values(ascending=False).to_dict(),
        "clusters": clusters,
        "cluster_mda": cmda.mean().sort_values(ascending=False).to_dict(),
        "stability": {"mda": stability(mda), "mdi": stability(mdi), "shap": stability(shap)},
    }


def evaluate_cpcv(
    cfg: Config, market: MarketData, store: FeatureStore, vol: pd.DataFrame
) -> dict[str, object]:
    """Combinatorial purged CV: the distribution of Sharpe ratios across backtest paths.

    Unlike walk-forward, early test groups here are predicted by models that
    were trained partly on later data. That's by design (it gives many paths
    through the same history), so these numbers are a sensitivity check, not
    the headline.
    """
    import numpy as np

    idx = sample_index(store, cfg.labels.target)
    idx = idx[idx >= first_test_date(store, cfg)]
    X = store.features
    y = store.labels[cfg.labels.target]
    t_end = store.labels["t_end_1"].loc[idx]
    cv = CombinatorialPurgedCV(cfg.cv.cpcv_groups, cfg.cv.cpcv_test_groups, cfg.cv.embargo)
    splits = cv.split(t_end, pd.DatetimeIndex(X.index))
    groups = cv.groups(len(idx))
    r_hold = holding_log_return(market, cfg.backtest.execution)
    inputs = inputs_for(market, store.labels["sigma"], cfg, cfg.universe.target)
    sc = SizingConfig(name="sign", rule="sign")
    out: dict[str, object] = {"n_splits": len(splits), "n_paths": cv.n_paths}
    for name in cfg.evaluation.cpcv_models:
        preds: dict[int, pd.Series] = {}
        for sp in splits:
            model = build_model(cfg.models[name], cfg)
            tr = idx[sp.fold.train]
            model.fit(X.loc[: tr.max()], y.loc[tr])
            te = idx[sp.fold.test]
            preds[sp.fold.fold_id] = model.predict(X.loc[: te.max()], te)
        path_srs = []
        for path in cv.paths():
            pieces = [preds[sid].loc[idx[groups[g]]] for g, sid in path]
            fc = pd.concat(pieces).sort_index() * store.labels["sigma"].reindex(idx)
            pos = size(
                fc, vol[cfg.vol.sizing_model], store.features["regime_p_high"], sc, cfg.strategy
            )
            path_srs.append(sharpe(run_vectorized(pos, r_hold, inputs, cfg.costs).net))
        a = np.asarray(path_srs)
        out[name] = {"sizing": "sign", "path_sharpes": a.tolist(), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                     "min": float(a.min()), "max": float(a.max())}  # fmt: skip
    return out


def load_bundle(cfg: Config) -> BacktestBundle:
    """Reload the saved backtest outputs."""
    d = reports_dir(cfg) / "backtest"
    meta = json.loads((d / "strategies.json").read_text(encoding="utf-8"))
    specs = {k: StrategySpec(**v) for k, v in meta["specs"].items()}
    frames = {
        k: pd.read_parquet(d / f"{k}.parquet") for k in ("net", "gross", "positions", "turnover")
    }
    return BacktestBundle(
        frames["net"],
        frames["gross"],
        frames["positions"],
        frames["turnover"],
        specs,
        meta["engine_max_abs_diff"],
    )


def stage_evaluate(cfg: Config) -> Path:
    """Every development-period statistic, written to ``reports/evaluation``."""
    store = load_store(cfg)
    loaded = load_market_data(cfg, mode="dev")
    out = reports_dir(cfg) / "evaluation"
    out.mkdir(exist_ok=True)
    forecasts = load_forecasts(cfg)
    vol = pd.read_parquet(reports_dir(cfg) / "vol" / "vol_forecasts.parquet")
    bundle = load_bundle(cfg)
    dates = pd.DatetimeIndex(bundle.net.index)
    _write_json(out / "forecasts.json", evaluate_forecasts(cfg, store, forecasts, dates))
    reg = registry(cfg)
    rows = [
        r for r in reg.rows() if r["kind"] in ("model", "strategy", "vol", "tuning", "v1_replica")
    ]
    r_hold = holding_log_return(loaded.market, cfg.backtest.execution)
    inputs = inputs_for(loaded.market, store.labels["sigma"], cfg, cfg.universe.target)
    strat = evaluate_strategies(cfg, bundle, rows, r_hold, inputs)
    _write_json(out / "strategies.json", strat)
    best = str(strat["best"])
    _write_json(out / "risk.json", evaluate_risk(cfg, loaded.market, store, bundle, vol, best))
    _write_json(out / "cpcv.json", evaluate_cpcv(cfg, loaded.market, store, vol))
    if cfg.evaluation.importance:
        _write_json(out / "importance.json", evaluate_importance(cfg, store))
    if cfg.evaluation.peer_study:
        peers = evaluate_peers(cfg, loaded, best)
        _write_json(out / "peers.json", peers)
        rid, sha = run_id(), git_sha()
        for p in peers["peers"]:  # type: ignore[attr-defined]
            reg.log(Trial(kind="peer", name=str(p["peer"]), run_id=rid, git_sha=sha, data_hash=loaded.data_hash,
                          config={"models": peers["models"], "sizing": peers["sizing"]}, metrics=p))  # fmt: skip
    log.info("evaluation written to %s (best dev strategy: %s)", out, best)
    return out


# ---------------------------------------------------------------------------- report


def stage_report(cfg: Config) -> Path:
    """results.json, figures, the tear sheet, app data, and the README results block."""
    from nvquant.reporting import figures as figs
    from nvquant.reporting.docs_gen import write_generated_docs
    from nvquant.reporting.readme import END, START
    from nvquant.reporting.results import build_results, save_results
    from nvquant.reporting.tearsheet import render_tearsheet

    out = reports_dir(cfg)
    res = build_results(cfg, out)
    save_results(res, out / "results.json")
    bundle = load_bundle(cfg)
    store = load_store(cfg)
    vol = pd.read_parquet(out / "vol" / "vol_forecasts.parquet")
    best = res["headline"]["best_strategy"]
    t = cfg.universe.target
    labels = {
        best: f"best: {best}",
        "bh_voltarget": "vol-targeted buy and hold",
        "bh_target": f"buy and hold {t}",
    }
    fdir = out / "figures"
    fg: dict[str, Path] = {
        "equity": figs.equity_curves(bundle.net, labels, fdir / "equity.png"),
        "drawdown": figs.drawdowns(bundle.net, labels, fdir / "drawdown.png"),
        "rolling": figs.rolling_sharpe_regimes(bundle.net, labels, store.features["regime_p_high"], fdir / "rolling_sharpe.png"),
        "vol": figs.vol_forecast_vs_realized(vol.loc[bundle.net.index[0]:], cfg.vol.sizing_model, fdir / "vol_forecast.png"),
    }  # fmt: skip
    sweep = pd.read_parquet(out / "backtest" / "cost_sweep.parquet")
    fg["costs"] = figs.cost_sensitivity(sweep, labels, fdir / "cost_sensitivity.png")
    if res.get("importance"):
        imp = json.loads((out / "evaluation" / "importance.json").read_text(encoding="utf-8"))
        fg["importance"] = figs.importance_bars(
            imp["mda"],
            imp["mda_se"],
            fdir / "importance_mda.png",
            "MDA: increase in out-of-fold MSE when permuted",
        )
        fg["cluster_importance"] = figs.importance_bars(
            imp["cluster_mda"],
            None,
            fdir / "importance_clusters.png",
            "Clustered MDA (features permuted together)",
        )
    peers_file = out / "evaluation" / "peers.json"
    if peers_file.exists():
        peers = json.loads(peers_file.read_text(encoding="utf-8"))
        key = f"{peers['models'][0]}{STRATEGY_SEP}{peers['sizing']}"
        fg["peers"] = figs.peer_dots(peers["peers"], key, fdir / "peers.png")
    (out / "tearsheet.html").write_text(render_tearsheet(res, fg), encoding="utf-8")
    app_dir = out / "app"
    app_dir.mkdir(exist_ok=True)
    store.features[["regime_p_high"]].loc[bundle.net.index[0] :].to_parquet(
        app_dir / "regime.parquet"
    )
    vol.loc[bundle.net.index[0] :].to_parquet(app_dir / "vol.parquet")
    readme = resolve(cfg.paths.readme)
    if not readme.exists():
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(f"# Results preview\n\n{START}\n{END}\n", encoding="utf-8")
    write_generated_docs(res, resolve(cfg.paths.docs_dir), readme)
    log.info("report written: %s", out / "results.json")
    return out / "results.json"


def reproduce(cfg: Config) -> Path:
    """Data -> data-report -> features -> train -> backtest -> evaluate -> report."""
    stage_data(cfg)
    stage_data_report(cfg)
    stage_features(cfg)
    stage_train(cfg)
    stage_backtest(cfg)
    stage_evaluate(cfg)
    return stage_report(cfg)


# --------------------------------------------------------------------------- lockbox


def preregistered(cfg: Config) -> dict[str, list[str]]:
    """The lockbox plan (``configs/preregistration.yaml``, fast: ``configs/preregistration_fast.yaml``)."""
    import yaml

    from nvquant.config import PROJECT_ROOT

    name = (
        "preregistration.yaml" if cfg.profile == "full" else f"preregistration_{cfg.profile}.yaml"
    )
    path = PROJECT_ROOT / "configs" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: write and commit the preregistration before the lockbox"
        )
    plan: dict[str, list[str]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return plan


def _require_committed(paths: list[str]) -> None:
    """Refuse to open the lockbox unless the preregistration files are committed and unchanged."""
    import subprocess

    from nvquant.config import PROJECT_ROOT

    for p in paths:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", p], cwd=PROJECT_ROOT, capture_output=True
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", p], cwd=PROJECT_ROOT, capture_output=True
        )
        if tracked.returncode != 0 or dirty.returncode != 0:
            raise LockboxError(
                f"{p} must be committed (and unchanged) before the lockbox is opened"
            )


def stage_lockbox(cfg: Config, force: bool = False, reason: str | None = None) -> Path:
    """Evaluate the preregistered strategies on 2025-01-01 onward, exactly once."""
    out = reports_dir(cfg)
    check_can_open(out, force, reason)
    plan = preregistered(cfg)
    if cfg.profile == "full":
        _require_committed(["docs/PREREGISTRATION.md", "configs/preregistration.yaml"])
    else:
        log.warning("profile %s: skipping the preregistration commit check", cfg.profile)
    sha = git_sha()
    if sha == "unknown":
        raise LockboxError("git SHA unavailable; the lockbox run must be tied to a commit")
    loaded = load_market_data(cfg, mode="lockbox")
    # Record the opening before any lockbox number exists, so a crash later in this
    # function still counts as the one evaluation.
    record_open(out, sha, loaded.data_hash, config_hash(cfg), reason)
    store = build_store(cfg, loaded)
    save_store(store, cfg, mode="lockbox")
    start = pd.Timestamp(cfg.lockbox.start)
    strategies: list[str] = plan["strategies"]
    needed = {s.split(STRATEGY_SEP)[0] for s in strategies if not s.startswith("meta")}
    forecasts = {}
    for name in sorted(needed):
        forecasts[name] = walk_forward(name, cfg.models[name], store, cfg, first_test=start).frame
    vol = all_vol_forecasts(loaded.market, store.features, pd.Timestamp(store.info["modeling_start"]),
                            cfg.vol.model_copy(update={"models": [cfg.vol.sizing_model]}), cfg.seed)  # fmt: skip
    dates = pd.DatetimeIndex(vol[cfg.vol.sizing_model].dropna().index)
    for f in forecasts.values():
        dates = dates.intersection(f["pred"].dropna().index)
    dates = dates[dates >= start]
    names = [*strategies, *plan.get("benchmarks", [])]
    bundle = run_strategies(cfg, loaded.market, store, forecasts, vol, dates,
                            include_meta=any(s.startswith("meta") for s in strategies), only=names)  # fmt: skip
    rows = {}
    for n in names:
        r = bundle.net[n].dropna()
        m: dict[str, object] = dict(performance(r, bundle.positions[n], bundle.turnover[n]))
        lo, hi, _ = bootstrap_sharpe_ci(r, cfg.evaluation.n_bootstrap, cfg.seed)
        m.update({"sharpe_ci_lo": lo, "sharpe_ci_hi": hi, "psr_0": psr_from_returns(r.to_numpy()),
                  "kind": bundle.specs[n].kind})  # fmt: skip
        rows[n] = m
    stress = stress_windows(bundle.net[names], cfg.evaluation.lockbox_stress_windows)
    ldir = out / "lockbox"
    ldir.mkdir(parents=True, exist_ok=True)
    for key, frame in (("net", bundle.net), ("positions", bundle.positions)):
        frame.to_parquet(ldir / f"{key}.parquet")
    result = {
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "n_days": len(dates),
        "strategies": rows, "stress": stress, "preregistered": plan, "git_sha": sha,
        "data_hash": loaded.data_hash, "engine_max_abs_diff": bundle.engine_max_diff,
    }  # fmt: skip
    _write_json(ldir / "results.json", result)
    mark_completed(out)
    registry(cfg).log(
        Trial(kind="lockbox", name="lockbox", run_id=run_id(), git_sha=sha, data_hash=loaded.data_hash,
              config={"plan": plan}, metrics={n: rows[n]["sharpe"] for n in names})
    )  # fmt: skip
    log.info("lockbox evaluated once: %s", ldir / "results.json")
    return ldir / "results.json"
