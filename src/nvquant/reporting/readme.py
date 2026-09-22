"""Render the README results block from results.json and inject it between the markers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def markers(name: str) -> tuple[str, str]:
    """Start and end markers of a generated block."""
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


START, END = markers("RESULTS")

LABELS = {
    "bh_target": "Buy and hold",
    "bh_voltarget": "Vol-targeted buy and hold",
    "bh_smh": "Buy and hold SMH",
    "bh_qqq": "Buy and hold QQQ",
    "trend_200": "200-day trend rule",
}


def fmt(x: Any, spec: str = ".2f", pct: bool = False) -> str:
    """Format a number; missing or non-finite values become "n/a"."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    v = float(x) * (100 if pct else 1)
    return f"{v:{spec}}" + ("%" if pct else "")


def label(name: str, target: str) -> str:
    """Human label for a strategy key."""
    if name in LABELS:
        return (
            LABELS[name].replace("Buy and hold", f"Buy and hold {target}")
            if name == "bh_target"
            else LABELS[name]
        )
    model, _, sizing = name.partition("__")
    return f"`{model}` + {sizing}"


def render_results_block(res: dict[str, Any]) -> str:
    """Markdown for the README results section. Every number comes from ``res``."""
    m, h = res["meta"], res["headline"]
    t = m["target"]
    lines = [
        f"Out-of-sample decisions from {m['oos_start']} to {m['oos_end']} ({m['oos_days']} sessions), "
        f"after costs, next-open execution. {m['n_strategy_trials']} strategy configurations were backtested, "
        f"and all of them count as trials in the Deflated Sharpe Ratio.",
        "",
        "| Strategy | Net Sharpe [95% CI] | CAGR | Vol | Max DD | Turnover/yr | DSR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in res["strategies"].items():
        ci = f"{fmt(r['sharpe'])} [{fmt(r['sharpe_ci_lo'])}, {fmt(r['sharpe_ci_hi'])}]"
        dsr = fmt(r["dsr"]) if r["kind"] != "benchmark" else "benchmark"
        lines.append(
            f"| {label(name, t)} | {ci} | {fmt(r['cagr'], '.1f', True)} | {fmt(r['ann_vol'], '.1f', True)} | "
            f"{fmt(r['max_drawdown'], '.1f', True)} | {fmt(r['turnover_ann'], '.1f')} | {dsr} |"
        )
    lines += [
        "",
        f"- Best development strategy: {label(h['best_strategy'], t)}. Its Sharpe minus the vol-targeted buy and hold's is "
        f"{fmt(h['best_minus_voltarget_sharpe'])}. DSR {fmt(h['best_dsr'])} (with every logged fit counted as a trial: "
        f"{fmt(h['best_dsr_all_fits'])}).",
        f"- Probability of backtest overfitting (CSCV): {fmt(h['pbo'])}. Hansen SPA p-value against vol-targeted buy and hold: "
        f"{fmt(h['spa_vs_bh_voltarget'])}; against buy and hold: {fmt(h['spa_vs_bh_target'])}.",
        f"- Random long/flat signals with the same turnover beat it {fmt(1 - h['random_null_percentile'], '.1f', True)} of the time.",
        f"- Fama-French 5 + momentum alpha: {fmt(h['alpha_ff5_ann'], '.1f', True)} a year (t = {fmt(h['alpha_ff5_t'])}). "
        f"2023 and 2024 account for {fmt(h['gain_share_2023_2024'], '.0f', True)} of its compounded dollar gain "
        f"({fmt(h['pnl_share_2023_2024'], '.0f', True)} of its summed daily returns).",
        "",
        "| Forecast model | IC | IC t (NW) | R2 OOS vs zero | DM p vs zero | Hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, f in res["forecasts"].items():
        if name == "zero":
            continue
        lines.append(
            f"| `{name}` | {fmt(f['ic'], '.3f')} | {fmt(f['ic_t'])} | {fmt(f['r2_oos'], '.2f', True)} | "
            f"{fmt(f['dm_pvalue'], '.3f')} | {fmt(f['hit_rate'], '.1f', True)} |"
        )
    v1 = res.get("v1_replica")
    if v1:
        lines += [
            "",
            f"v1 replica (the original stacked ReLU LSTM on price levels, refit yearly): median absolute error "
            f"{fmt(v1['median_abs_error_model'])} vs {fmt(v1['median_abs_error_persistence'])} for persistence; worse than "
            f"persistence on {fmt(v1['share_days_worse_than_persistence'], '.0f', True)} of days; on "
            f"{fmt(v1['share_days_blown_up'], '.1f', True)} of days its forecast exceeded 10x the training range "
            f"(ReLU blow-up once prices left the MinMax range). Direction accuracy {fmt(v1['direction_accuracy'], '.1f', True)} "
            f"vs {fmt(v1['share_days_up'], '.1f', True)} up days. Details: [docs/V1_POSTMORTEM.md](docs/V1_POSTMORTEM.md).",
        ]
    lock = res.get("lockbox")
    if lock:
        lines += ["", f"**Lockbox** ({lock['start']} to {lock['end']}, {lock['n_days']} sessions, evaluated once):", "",
                  "| Strategy | Net Sharpe [95% CI] | Total return | Max DD |", "|---|---|---|---|"]  # fmt: skip
        for name, r in lock["strategies"].items():
            lines.append(
                f"| {label(name, t)} | {fmt(r['sharpe'])} [{fmt(r['sharpe_ci_lo'])}, {fmt(r['sharpe_ci_hi'])}] | "
                f"{fmt(r['total_return'], '.1f', True)} | {fmt(r['max_drawdown'], '.1f', True)} |"
            )
    else:
        lines += ["", "Lockbox (2025-01-01 onward): not evaluated yet."]
    lines += [
        "",
        f"<sub>Generated by `make report` from `reports/results.json` (git {m['git_sha']}, data {m['data_hash']}, config {m['config_hash']}).</sub>",
    ]
    return "\n".join(lines)


def _span(text: str, name: str) -> tuple[int, int]:
    start, end = markers(name)
    i, j = text.find(start), text.find(end)
    if i < 0 or j < i:
        raise ValueError(f"no {name} markers")
    return i + len(start), j


def has_block(text: str, name: str = "RESULTS") -> bool:
    """True if the text contains both markers of a block."""
    start, end = markers(name)
    return start in text and end in text


def extract_block(text: str, name: str = "RESULTS") -> str:
    """Text between a block's markers, without the surrounding newlines."""
    i, j = _span(text, name)
    return text[i:j].strip("\n")


def inject(path: Path, block: str, name: str = "RESULTS") -> None:
    """Replace the text between a block's markers (the markers must already exist)."""
    text = path.read_text(encoding="utf-8")
    i, j = _span(text, name)
    path.write_text(text[:i] + "\n" + block + "\n" + text[j:], encoding="utf-8")
