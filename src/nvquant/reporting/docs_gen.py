"""Generated documentation: docs/RESULTS.md in full, plus marked blocks in other docs.

Every number in these renderings comes from ``results.json``. Prose that
surrounds the blocks in hand-written docs describes methods, not results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvquant.reporting.readme import fmt, has_block, inject, label, render_results_block

BASELINES = ("zero", "hist_mean", "ar")
BENCHMARKS = ("bh_target", "bh_voltarget", "bh_smh", "bh_qqq", "trend_200")
DEEP = ("lstm_gauss", "lstm_quant", "gru_gauss", "gru_quant", "tcn", "patchtst")


def _model_of(name: str) -> str:
    return name.split("__")[0]


def ml_strategies(res: dict[str, Any]) -> dict[str, float]:
    """Net Sharpe of every strategy built on a non-baseline forecast (ML models, ensembles, meta)."""
    return {
        k: v
        for k, v in res["all_strategy_sharpes"].items()
        if k not in BENCHMARKS and _model_of(k) not in BASELINES and v == v
    }


def render_what_didnt_work(res: dict[str, Any]) -> str:
    """README bullet list of negative results, generated from results.json."""
    h = res["headline"]
    ml = ml_strategies(res)
    best_ml = max(ml, key=lambda k: ml[k])
    fc = res["forecasts"]
    ml_models = [m for m in fc if m not in BASELINES]
    sig = [m for m in ml_models if (fc[m]["ic_t"] or 0) > 2]
    deep_r2 = [fc[m]["r2_oos"] for m in DEEP if m in fc]
    sh = res["all_strategy_sharpes"]
    reg_pairs = [
        (k, k.replace("voltarget_regime", "voltarget"))
        for k in sh
        if k.endswith("__voltarget_regime")
    ]
    regime_worse = sum(1 for a, b in reg_pairs if b in sh and sh[a] < sh[b])
    mcs_out = [m for m, p in res["mcs_pvalues"].items() if p < 0.1]
    lines = [
        f"- **Machine learning didn't beat vol targeting.** The best strategy built on an ML forecast is "
        f"{label(best_ml, res['meta']['target'])} at a net Sharpe of {fmt(ml[best_ml])}, against "
        f"{fmt(h['voltarget_sharpe'])} for vol-targeted buy and hold. Across all {res['meta']['n_strategy_trials']} "
        f"configurations, Hansen's SPA p-value against that benchmark is {fmt(h['spa_vs_bh_voltarget'])}.",
        f"- **Forecasts are weak.** {len(sig)} of {len(ml_models)} ML and ensemble forecasts have a Newey-West IC "
        f"t-stat above 2 ({', '.join(f'`{m}`' for m in sig) or 'none'}), and none of them turns that into a better "
        "strategy than the benchmark after costs.",
        f"- **Deep models were the worst forecasters.** Their out-of-sample R2 against the zero forecast ranges from "
        f"{fmt(min(deep_r2), '.2f', True)} to {fmt(max(deep_r2), '.2f', True)}. The model confidence set at 10% drops "
        f"{', '.join(f'`{m}`' for m in mcs_out) or 'none'}.",
        f"- **The regime filter hurt.** Going flat in the filtered high-vol regime lowered the Sharpe for "
        f"{regime_worse} of {len(reg_pairs)} vol-targeted strategies.",
        f"- **Meta-labeling on the 200-day trend rule** reached a net Sharpe of {fmt(sh.get('meta__voltarget'))} "
        f"(vol-targeted), against {fmt(sh.get('trend_200'))} for the trend rule alone.",
    ]
    peers = res.get("peers")
    if peers:
        n = peers["n_peers"]
        counts = ", ".join(f"`{m}` on {c} of {n}" for m, c in peers["n_beating_voltarget"].items())
        lines.append(
            f"- **No edge on the peers either.** With the same pipeline and the {peers['sizing']} sizing, the strategy "
            f"beat that peer's vol-targeted buy and hold for {counts}."
        )
    v1 = res.get("v1_replica")
    if v1:
        lines.append(
            f"- **The v1 model doesn't survive walk-forward.** It was worse than persistence on "
            f"{fmt(v1['share_days_worse_than_persistence'], '.0f', True)} of days."
        )
    return "\n".join(lines)


def render_v1_block(res: dict[str, Any]) -> str:
    """V1_POSTMORTEM numbers."""
    v = res["v1_replica"]
    fits = res.get("v1_fits") or []
    rows = "\n".join(
        f"| {f['refit_date']} | {f['n_train']} | {fmt(f['train_min'])} to {fmt(f['train_max'])} | {fmt(f['final_train_mse_scaled'], '.4f')} |"
        for f in fits
    )
    return f"""| Metric | v1 replica | Persistence |
|---|---|---|
| Out-of-sample days | {v["n"]:.0f} | {v["n"]:.0f} |
| Median absolute error ($) | {fmt(v["median_abs_error_model"], ".3f")} | {fmt(v["median_abs_error_persistence"], ".3f")} |
| RMSE ($) | {fmt(v["rmse_model"], ".3g")} | {fmt(v["rmse_persistence"], ".3f")} |
| RMSE in v1's scaled units | {fmt(v["rmse_model_scaled"], ".3g")} | {fmt(v["rmse_persistence_scaled"], ".4f")} |

- Worse than persistence on {fmt(v["share_days_worse_than_persistence"], ".1f", True)} of days (Diebold-Mariano stat {fmt(v["dm_stat"])}, p = {fmt(v["dm_pvalue"], ".4f")}).
- On {fmt(v["share_test_above_train_max"], ".1f", True)} of days the latest open was above the training window's maximum, so the MinMax-scaled input was above 1.
- On {fmt(v["share_days_blown_up"], ".1f", True)} of days the forecast was more than 10 times the training maximum. The largest absolute forecast was {fmt(v["max_abs_prediction"], ".3g")} for a stock whose highest open in the period was {fmt(v["max_open"])}.
- Direction accuracy {fmt(v["direction_accuracy"], ".1f", True)}, while {fmt(v["share_days_up"], ".1f", True)} of days were up. The IC of the implied return is {fmt(v["implied_return_ic"], ".3f")}.
- Parameters: {v["n_parameters"]:,}.

| Refit | Training windows | Training range of the open ($) | Final training MSE (scaled) |
|---|---|---|---|
{rows}"""


def render_resume(res: dict[str, Any]) -> str:
    """Three resume bullets, using only numbers from results.json."""
    m, h = res["meta"], res["headline"]
    v1 = res.get("v1_replica") or {}
    return "\n".join(
        [
            f"- Built a leakage-tested research pipeline for daily NVDA returns in Python: {m.get('n_features', 'n/a')} causal "
            f"features checked by a perturb-the-future property test, purged walk-forward validation over {m['oos_days']:,} "
            f"out-of-sample sessions, and two independent backtest engines that agree to {m['engine_max_abs_diff']:.0e}.",
            f"- Evaluated {m['n_strategy_trials']} strategy configurations (linear, gradient boosting, LSTM/GRU/TCN/Transformer) "
            f"net of costs with trial-adjusted statistics (Deflated Sharpe, PBO {fmt(h['pbo'])}, Hansen SPA). Reported the "
            f"null result: no model beat vol-targeted buy and hold (SPA p = {fmt(h['spa_vs_bh_voltarget'])}).",
            f"- Audited my earlier LSTM price model: re-run with walk-forward it was worse than a persistence forecast on "
            f"{fmt(v1.get('share_days_worse_than_persistence'), '.0f', True)} of days, and I traced the failure to "
            "MinMax-scaled inputs leaving the training range.",
        ]
    )


def render_key_numbers(res: dict[str, Any]) -> str:
    """A compact table for INTERVIEW_NOTES."""
    m, h = res["meta"], res["headline"]
    return f"""| Quantity | Value |
|---|---|
| Out-of-sample period | {m["oos_start"]} to {m["oos_end"]} ({m["oos_days"]:,} sessions) |
| Strategy configurations (DSR trials) | {m["n_strategy_trials"]} |
| Every logged fit | {m["n_all_fits"]} |
| Best configuration | {label(h["best_strategy"], m["target"])} |
| Its net Sharpe [95% CI] | {fmt(h["best_sharpe"])} [{fmt(h["best_sharpe_ci"][0])}, {fmt(h["best_sharpe_ci"][1])}] |
| Vol-targeted buy and hold | {fmt(h["voltarget_sharpe"])} |
| Buy and hold | {fmt(h["buy_hold_sharpe"])} |
| DSR (strategy trials / all fits) | {fmt(h["best_dsr"])} / {fmt(h["best_dsr_all_fits"])} |
| PBO | {fmt(h["pbo"])} |
| SPA p vs vol-targeted buy and hold | {fmt(h["spa_vs_bh_voltarget"])} |
| FF5 + momentum alpha (t) | {fmt(h["alpha_ff5_ann"], ".1f", True)} ({fmt(h["alpha_ff5_t"])}) |
| Share of the compounded gain from 2023 and 2024 | {fmt(h["gain_share_2023_2024"], ".0f", True)} |
| Share of summed daily returns from 2023 and 2024 | {fmt(h["pnl_share_2023_2024"], ".0f", True)} |"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_results_md(res: dict[str, Any]) -> str:
    """The full generated docs/RESULTS.md."""
    m, h = res["meta"], res["headline"]
    t = m["target"]
    sh = res["all_strategy_sharpes"]
    parts = [
        "# Results",
        "",
        "Generated by `make report` from `reports/results.json`. Don't edit by hand; every number below is regenerated.",
        f"Git `{m['git_sha']}`, data `{m['data_hash']}`, config `{m['config_hash']}`, generated {m['generated_at']}.",
        "",
        "## Headline",
        "",
        render_results_block(res),
        "",
        "## What didn't work",
        "",
        render_what_didnt_work(res),
        "",
        "## Every strategy configuration (net Sharpe, development OOS)",
        "",
        _table(
            ["strategy", "net Sharpe"],
            [
                [label(k, t), fmt(v)]
                for k, v in sorted(sh.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 9)
            ],
        ),
        "",
        "## Forecast accuracy",
        "",
        _table(
            [
                "model",
                "IC",
                "IC t",
                "IC 5d",
                "IC 21d",
                "R2 vs zero",
                "R2 vs hist mean",
                "DM p",
                "hit rate",
                "PT p",
                "conformal 90% coverage",
            ],
            [
                [
                    f"`{k}`",
                    fmt(f["ic"], ".3f"),
                    fmt(f["ic_t"]),
                    fmt((f["ic_decay"] or {}).get("5"), ".3f"),
                    fmt((f["ic_decay"] or {}).get("21"), ".3f"),
                    fmt(f["r2_oos"], ".2f", True),
                    fmt(f["r2_oos_vs_hist_mean"], ".2f", True),
                    fmt(f["dm_pvalue"], ".3f"),
                    fmt(f["hit_rate"], ".1f", True),
                    fmt(f["pt_pvalue"], ".3f"),
                    fmt(f["conformal_coverage"], ".3f"),
                ]
                for k, f in res["forecasts"].items()
            ],
        ),
        "",
        "Calibration of the probabilistic heads:",
        "",
        _table(
            ["model", "PIT KS stat", "PIT KS p", "interval coverage (target)"],
            [
                [
                    f"`{k}`",
                    fmt(f["calibration"]["pit"]["ks_stat"], ".3f"),
                    fmt(f["calibration"]["pit"]["ks_pvalue"], ".3f"),
                    (
                        f"{fmt(f['calibration']['coverage_90'], '.1f', True)} (90%)"
                        if "coverage_90" in f["calibration"]
                        else f"{fmt(f['calibration']['coverage_80'], '.1f', True)} (80%)"
                    ),
                ]
                for k, f in res["forecasts"].items()
                if f.get("calibration")
            ],
        ),
        "",
        "Model confidence set p-values (models with p > 0.1 stay in the set): "
        + ", ".join(f"`{k}` {fmt(v, '.3f')}" for k, v in res["mcs_pvalues"].items())
        + ".",
        "",
        "## Volatility forecasts",
        "",
        _table(
            ["model", "QLIKE", "MZ alpha x1e4", "MZ beta", "MZ R2", "MZ Wald p (alpha=0, beta=1)"],
            [
                [
                    k,
                    fmt(v["qlike"], ".4f"),
                    fmt(v["mz_alpha"] * 1e4, ".3f"),
                    fmt(v["mz_beta"], ".3f"),
                    fmt(v["mz_r2"], ".3f"),
                    fmt(v["mz_wald_pvalue"], ".3f"),
                ]
                for k, v in (res["vol_models"] or {}).items()
            ],
        ),
        "",
        "## Significance",
        "",
        f"- PBO {fmt(res['pbo']['pbo'])} over {res['pbo']['n_combinations']:,} CSCV combinations; slope of OOS on IS Sharpe "
        f"{fmt(res['pbo']['degradation_slope'])}; share of IS winners that lose money OOS {fmt(res['pbo']['prob_oos_loss'], '.1f', True)}.",
    ]
    for b, s in res["spa"].items():
        parts.append(
            f"- Against {label(b, t)}: SPA consistent p {fmt(s['spa_consistent'], '.3f')}, lower {fmt(s['spa_lower'], '.3f')}, "
            f"upper {fmt(s['spa_upper'], '.3f')}; White's Reality Check p {fmt(s['reality_check'], '.3f')}."
        )
    rn = res["random_null"]
    parts += [
        f"- Random long/flat null ({rn['n_paths']} paths, long {fmt(rn['long_share'], '.1f', True)} of the time, switch rate "
        f"{fmt(rn['switch_rate'], '.4f')}): median Sharpe {fmt(rn['sharpe_p50'])}, 95th percentile {fmt(rn['sharpe_p95'])}; "
        f"the best configuration beats {fmt(rn['best_percentile'], '.1f', True)} of paths.",
        f"- Best configuration: PSR(0) {fmt(h['best_psr_0'], '.3f')}, DSR {fmt(h['best_dsr'], '.3f')}, minimum track record "
        f"{fmt(h['best_min_trl_days'], '.0f')} sessions.",
        "",
        "## Attribution (Newey-West t-stats)",
        "",
    ]
    rows = []
    for n, a in res["attribution"].items():
        for reg, v in a.items():
            if "alpha_ann" in v:
                betas = ", ".join(
                    f"{k} {fmt(b['beta'])} ({fmt(b['t'])})" for k, b in v["betas"].items()
                )
                rows.append(
                    [
                        label(n, t),
                        reg,
                        fmt(v["alpha_ann"], ".1f", True),
                        fmt(v["alpha_t"]),
                        betas,
                        fmt(v["r2"], ".2f"),
                    ]
                )
    parts += [
        _table(["strategy", "model", "alpha / yr", "alpha t", "betas (t)", "R2"], rows),
        "",
        "## Risk",
        "",
    ]
    vrows = []
    for lvl, v in res["var"].items():
        for meth, b in v["backtests"].items():
            vrows.append(
                [
                    lvl,
                    meth,
                    fmt(b["rate"], ".2f", True),
                    fmt(b["expected"], ".2f", True),
                    fmt(b["kupiec_pvalue"], ".3f"),
                    fmt(b["christoffersen_cc_pvalue"], ".3f"),
                ]
            )
    parts += [
        f"VaR backtests for {label(h['best_strategy'], t)}:",
        "",
        _table(
            ["level", "method", "exception rate", "expected", "Kupiec p", "Christoffersen CC p"],
            vrows,
        ),
        "",
        "Stress windows:",
        "",
    ]
    srows = []
    for name, w in res["stress"].items():
        if not w.get("covered"):
            srows.append([name, f"{w['start']} to {w['end']}", "no out-of-sample coverage", "", ""])
            continue
        for s, r in w["results"].items():
            srows.append(
                [
                    name,
                    f"{w['start']} to {w['end']}",
                    label(s, t),
                    fmt(r["total_return"], ".1f", True),
                    fmt(r["max_drawdown"], ".1f", True),
                ]
            )
    parts += [
        _table(["window", "dates", "strategy", "return", "max DD"], srows),
        "",
        "Regimes (filtered HMM state at the decision date):",
        "",
    ]
    parts.append(
        _table(
            ["strategy", "low-vol Sharpe", "high-vol Sharpe"],
            [
                [
                    label(n, t),
                    fmt(v.get("low_vol", {}).get("sharpe")),
                    fmt(v.get("high_vol", {}).get("sharpe")),
                ]
                for n, v in res["regimes"].items()
            ],
        )
    )
    years = res["calendar_years"]
    cols = list(years)
    all_years = sorted({y for v in years.values() for y in v})
    parts += [
        "",
        "Calendar-year returns:",
        "",
        _table(
            ["year", *[label(c, t) for c in cols]],
            [[y, *[fmt(years[c].get(y), ".1f", True) for c in cols]] for y in all_years],
        ),
    ]
    mc = res["monte_carlo"]
    parts += [
        "",
        "Block-bootstrap Monte Carlo of the equity path:",
        "",
        _table(
            ["strategy", "terminal 5%", "terminal 50%", "terminal 95%", "P(loss)", "max DD median"],
            [
                [
                    label(n, t),
                    fmt(v["terminal_p05"]),
                    fmt(v["terminal_p50"]),
                    fmt(v["terminal_p95"]),
                    fmt(v["prob_loss"], ".1f", True),
                    fmt(v["mdd_p50"], ".1f", True),
                ]
                for n, v in mc.items()
            ],
        ),
    ]
    parts += [
        "",
        "## Costs and capacity",
        "",
        _table(
            ["strategy", *[f"{b} bps" for b in next(iter(res["cost_sweep"].values()))]],
            [[label(k, t), *[fmt(x) for x in v.values()]] for k, v in res["cost_sweep"].items()],
        ),
    ]
    parts += [
        "",
        _table(
            ["strategy", *[f"AUM {float(a):,.0f}" for a in next(iter(res["capacity"].values()))]],
            [[label(k, t), *[fmt(x) for x in v.values()]] for k, v in res["capacity"].items()],
        ),
    ]
    peers = res.get("peers")
    if peers:
        parts += [
            "",
            "## Peer study (same pipeline on every peer, development period)",
            "",
            _table(
                [
                    "ticker",
                    "from",
                    *[f"`{mm}` + {peers['sizing']}" for mm in peers["models"]],
                    "buy and hold",
                    "vol-targeted buy and hold",
                ],
                [
                    [
                        r["peer"],
                        r["first_date"],
                        *[fmt(r["strategy"][mm]) for mm in peers["models"]],
                        fmt(r["bh_target"]),
                        fmt(r["bh_voltarget"]),
                    ]
                    for r in peers["rows"]
                ],
            ),
        ]
    imp = res.get("importance")
    if imp:
        parts += [
            "",
            "## Feature importance",
            "",
            f"Stability across purged folds (mean rank correlation): MDA {fmt(imp['stability']['mda'])}, "
            f"MDI {fmt(imp['stability']['mdi'])}, SHAP {fmt(imp['stability']['shap'])}.",
            "",
            _table(
                ["rank", "MDA", "TreeSHAP", "MDI (gain)"],
                [
                    [str(i + 1), f"`{a}`", f"`{b}`", f"`{c}`"]
                    for i, (a, b, c) in enumerate(
                        zip(imp["mda_top"], imp["shap_top"], imp["mdi_top"], strict=False)
                    )
                ],
            ),
            "",
            "Clustered MDA:",
            "",
            _table(
                ["cluster", "MDA", "members"],
                [
                    [k, fmt(v, ".2e"), ", ".join(f"`{c}`" for c in imp["clusters"][k])]
                    for k, v in imp["cluster_mda"].items()
                ],
            ),
        ]
    cp = res.get("cpcv")
    if cp:
        parts += [
            "",
            "## Combinatorial purged CV",
            "",
            f"{cp['n_splits']} splits, {cp['n_paths']} paths, sign sizing (a sensitivity check; see the methodology).",
            "",
            _table(
                ["model", "mean path Sharpe", "std", "min", "max"],
                [
                    [f"`{k}`", fmt(v["mean"]), fmt(v["std"]), fmt(v["min"]), fmt(v["max"])]
                    for k, v in cp.items()
                    if isinstance(v, dict)
                ],
            ),
        ]
    lock = res.get("lockbox")
    if lock:
        parts += [
            "",
            "## Lockbox",
            "",
            f"Evaluated once ({res['lockbox_runs']} run in the sentinel), {lock['start']} to {lock['end']}.",
            "",
        ]
        parts.append(
            _table(
                ["strategy", "net Sharpe [95% CI]", "total return", "max DD", "PSR(0)"],
                [
                    [
                        label(n, t),
                        f"{fmt(r['sharpe'])} [{fmt(r['sharpe_ci_lo'])}, {fmt(r['sharpe_ci_hi'])}]",
                        fmt(r["total_return"], ".1f", True),
                        fmt(r["max_drawdown"], ".1f", True),
                        fmt(r["psr_0"], ".3f"),
                    ]
                    for n, r in lock["strategies"].items()
                ],
            )
        )
        srows = []
        for name, w in lock["stress"].items():
            if not w.get("covered"):
                srows.append([name, f"{w['start']} to {w['end']}", "not covered", "", ""])
                continue
            for s, r in w["results"].items():
                srows.append(
                    [
                        name,
                        f"{w['start']} to {w['end']}",
                        label(s, t),
                        fmt(r["total_return"], ".1f", True),
                        fmt(r["max_drawdown"], ".1f", True),
                    ]
                )
        parts += [
            "",
            "2025 stress windows (lockbox run only):",
            "",
            _table(["window", "dates", "strategy", "return", "max DD"], srows),
        ]
    return "\n".join(parts) + "\n"


def generated_blocks(res: dict[str, Any]) -> dict[str, str]:
    """Every named block, keyed by marker name."""
    blocks = {"RESULTS": render_results_block(res), "WHAT_DIDNT_WORK": render_what_didnt_work(res), "KEY_NUMBERS": render_key_numbers(res),
              "RESUME": render_resume(res)}  # fmt: skip
    if res.get("v1_replica"):
        blocks["V1"] = render_v1_block(res)
    return blocks


def write_generated_docs(res: dict[str, Any], docs_dir: Path, readme: Path) -> list[Path]:
    """Write RESULTS.md and fill every marked block found in the README and docs."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "RESULTS.md").write_text(render_results_md(res), encoding="utf-8")
    touched = [docs_dir / "RESULTS.md"]
    blocks = generated_blocks(res)
    for path in [readme, *sorted(docs_dir.glob("*.md"))]:
        if not path.exists() or path.name == "RESULTS.md":
            continue
        text = path.read_text(encoding="utf-8")
        for name, block in blocks.items():
            if has_block(text, name):
                inject(path, block, name)
                touched.append(path)
    return touched
