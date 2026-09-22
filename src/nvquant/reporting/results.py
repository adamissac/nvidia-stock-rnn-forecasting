"""Assemble ``reports/results.json``: the single source for every number in the docs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nvquant.config import Config, config_hash
from nvquant.experiments.repro import git_sha

FOCUS_SIZING = "voltarget"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _v1(reports: Path) -> dict[str, Any]:
    """v1 replica metrics recomputed from the saved forecasts (so new metrics don't need a retrain)."""
    from nvquant.models.v1_replica import V1Net, n_parameters, v1_metrics

    frame = pd.read_parquet(reports / "v1" / "v1_replica.parquet")
    return v1_metrics(frame) | {"n_parameters": n_parameters(V1Net())}


def load_results(path: Path) -> dict[str, Any]:
    """Read a results.json file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _strategy_row(name: str, m: dict[str, Any]) -> dict[str, Any]:
    keys = ["sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "sharpe_gross", "cagr", "ann_vol", "max_drawdown",
            "turnover_ann", "exposure", "dsr", "dsr_all_fits", "psr_0", "min_trl_days", "kind", "n"]  # fmt: skip
    return {"name": name, **{k: m.get(k) for k in keys}}


def table_strategies(strat: dict[str, Any]) -> list[str]:
    """Which strategies appear in the headline table: the best, one per model family, and benchmarks."""
    rows = strat["strategies"]
    best = strat["best"]
    picks = [best]
    for fam in (
        "ridge",
        "lgbm",
        "lstm_gauss",
        "gru_quant",
        "tcn",
        "patchtst",
        "ens_equal",
        "ens_stack",
        "meta",
    ):
        cands = [k for k in rows if k.split("__")[0] == fam and rows[k]["kind"] != "benchmark"]
        if cands:
            top = max(
                cands,
                key=lambda k: rows[k]["sharpe"] if rows[k]["sharpe"] == rows[k]["sharpe"] else -9,
            )
            if top not in picks:
                picks.append(top)
    picks += [
        b for b in ("bh_voltarget", "bh_target", "bh_smh", "bh_qqq", "trend_200") if b in rows
    ]
    return picks


def build_results(cfg: Config, reports: Path) -> dict[str, Any]:
    """Collect the evaluation artifacts into one JSON-ready dict."""
    ev = reports / "evaluation"
    strat = _read(ev / "strategies.json")
    fc = _read(ev / "forecasts.json")
    risk = _read(ev / "risk.json")
    peers = _read(ev / "peers.json")
    imp = _read(ev / "importance.json")
    cpcv = _read(ev / "cpcv.json")
    vol = _read(reports / "vol" / "scores.json")
    v1 = _read(reports / "v1" / "metrics.json")
    dq = _read(reports / "data_quality.json")
    bt = _read(reports / "backtest" / "strategies.json")
    lock = _read(reports / "lockbox" / "results.json")
    sentinel = _read(reports / "lockbox" / "SENTINEL.json")
    missing = [n for n, v in (("data_quality.json", dq), ("evaluation/strategies.json", strat),
                               ("evaluation/forecasts.json", fc), ("evaluation/risk.json", risk),
                               ("backtest/strategies.json", bt)) if v is None]  # fmt: skip
    if missing:
        raise FileNotFoundError(
            f"cannot build results.json; missing {missing} (run the earlier stages)"
        )
    sweep = pd.read_parquet(reports / "backtest" / "cost_sweep.parquet")
    capacity = pd.read_parquet(reports / "backtest" / "capacity.parquet")
    best = strat["best"]
    rows = strat["strategies"]
    b = rows[best]
    vt = rows["bh_voltarget"]
    bh = rows["bh_target"]
    peer_summary = None
    if peers:
        key = f"{peers['models'][0]}__{peers['sizing']}"
        per = []
        for p in peers["peers"]:
            s = p["sharpe"]
            best_model = max(
                (f"{m}__{peers['sizing']}" for m in peers["models"]),
                key=lambda k: s.get(k, float("-inf")),
            )
            per.append({"peer": p["peer"], "n": p["n"], "first_date": p["first_date"],
                        "strategy": {m: s.get(f"{m}__{peers['sizing']}") for m in peers["models"]},
                        "bh_target": s["bh_target"], "bh_voltarget": s["bh_voltarget"],
                        "beats_voltarget": {m: s.get(f"{m}__{peers['sizing']}", float("-inf")) > s["bh_voltarget"] for m in peers["models"]},
                        "best_model": best_model})  # fmt: skip
        model_counts = {m: sum(r["beats_voltarget"][m] for r in per) for m in peers["models"]}
        peer_summary = {"sizing": peers["sizing"], "models": peers["models"], "rows": per,
                        "n_peers": len(per), "n_beating_voltarget": model_counts, "reference_key": key}  # fmt: skip
    forecast_rows = {
        name: {k: m.get(k) for k in ("ic", "ic_t", "r2_oos", "r2_oos_vs_hist_mean", "hit_rate", "pt_pvalue", "n")}
        | {"dm_pvalue": m["dm_vs_zero"]["pvalue"], "dm_stat": m["dm_vs_zero"]["stat"], "ic_decay": m.get("ic_decay"),
           "conformal_coverage": m["conformal"]["coverage"], "calibration": m.get("calibration")}
        for name, m in fc["models"].items()
    }  # fmt: skip
    attribution = risk["attribution"]
    out: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "config_hash": config_hash(cfg),
            "profile": cfg.profile,
            "data_hash": dq["data_hash"],
            "modeling_start": dq["modeling_start"],
            "oos_start": bt["first_date"],
            "oos_end": bt["last_date"],
            "oos_days": bt["n_dates"],
            "n_strategy_trials": strat["n_trials"],
            "n_all_fits": strat["n_all_fits"],
            "engine_max_abs_diff": bt["engine_max_abs_diff"],
            "split_checks_passed": dq["split_checks_passed"],
            "target": cfg.universe.target,
            "n_features": (_read(reports / "features_info.json") or {}).get("n_features"),
        },
        "headline": {
            "best_strategy": best,
            "best_sharpe": b["sharpe"],
            "best_sharpe_ci": [b["sharpe_ci_lo"], b["sharpe_ci_hi"]],
            "best_dsr": b["dsr"],
            "best_dsr_all_fits": b["dsr_all_fits"],
            "best_psr_0": b["psr_0"],
            "best_min_trl_days": b["min_trl_days"],
            "voltarget_sharpe": vt["sharpe"],
            "buy_hold_sharpe": bh["sharpe"],
            "best_minus_voltarget_sharpe": b["sharpe"] - vt["sharpe"],
            "pbo": strat["pbo"]["pbo"],
            "spa_vs_bh_voltarget": strat["spa"].get("bh_voltarget", {}).get("spa_consistent"),
            "spa_vs_bh_target": strat["spa"].get("bh_target", {}).get("spa_consistent"),
            "rc_vs_bh_voltarget": strat["spa"].get("bh_voltarget", {}).get("reality_check"),
            "spa_vs_bh_voltarget_vol_matched": strat["spa"]
            .get("bh_voltarget", {})
            .get("vol_matched_spa_consistent"),
            "random_null_percentile": strat["random_null"]["best_percentile"],
            "alpha_ff5_ann": attribution[best]["FF5_MOM"].get("alpha_ann"),
            "alpha_ff5_t": attribution[best]["FF5_MOM"].get("alpha_t"),
            "alpha_ff5_voltarget_ann": attribution["bh_voltarget"]["FF5_MOM"].get("alpha_ann"),
            "alpha_ff5_voltarget_t": attribution["bh_voltarget"]["FF5_MOM"].get("alpha_t"),
            "alpha_smh_ann": attribution[best].get("SMH", {}).get("alpha_ann"),
            "alpha_smh_t": attribution[best].get("SMH", {}).get("alpha_t"),
            "pnl_share_2023_2024": risk["pnl_share_2023_2024"],
            "gain_share_2023_2024": risk["gain_share_2023_2024"],
        },
        "strategies": {k: _strategy_row(k, rows[k]) for k in table_strategies(strat)},
        "all_strategy_sharpes": {k: v["sharpe"] for k, v in rows.items()},
        "forecasts": forecast_rows,
        "mcs_pvalues": fc["mcs_pvalues"],
        "vol_models": vol,
        "v1_replica": _v1(reports) if v1 else None,
        "v1_fits": v1["fits"] if v1 else None,
        "cost_sweep": {
            k: sweep.loc[k].to_dict()
            for k in (best, "bh_voltarget", "bh_target")
            if k in sweep.index
        },
        "capacity": capacity.to_dict(orient="index"),
        "pbo": strat["pbo"],
        "spa": strat["spa"],
        "random_null": strat["random_null"],
        "attribution": attribution,
        "var": risk["var"],
        "stress": risk["stress"],
        "regimes": risk["regimes"],
        "calendar_years": risk["calendar_years"],
        "monte_carlo": risk["monte_carlo"],
        "peers": peer_summary,
        "importance": None
        if imp is None
        else {
            "mda_top": dict(list(imp["mda"].items())[:15]),
            "mda_se": imp["mda_se"],
            "shap_top": dict(list(imp["shap"].items())[:15]),
            "mdi_top": dict(list(imp["mdi"].items())[:15]),
            "cluster_mda": imp["cluster_mda"],
            "clusters": imp["clusters"],
            "stability": imp["stability"],
        },
        "cpcv": cpcv,
        "lockbox": lock,
        "lockbox_runs": sentinel["n_runs"] if sentinel else 0,
    }
    return out


def save_results(results: dict[str, Any], path: Path) -> Path:
    """Write results.json."""
    path.write_text(
        json.dumps(results, indent=2, default=str, allow_nan=True) + "\n", encoding="utf-8"
    )
    return path
