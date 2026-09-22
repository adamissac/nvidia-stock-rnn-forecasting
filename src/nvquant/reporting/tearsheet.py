"""Static HTML tear sheet built from results.json and the saved figures."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import markdown_it

from nvquant.reporting.readme import fmt, label, render_results_block

CSS = """
:root { --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --rule:#e4e3df; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --rule:#3a3a37; } img { filter: none; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; max-width:1040px; margin:0 auto; padding:24px 16px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin-top:32px; border-bottom:1px solid var(--rule); padding-bottom:4px; }
.meta { color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--rule); padding:4px 8px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
img { max-width:100%; border-radius:6px; margin:8px 0; background:#fcfcfb; }
code { font-size:12px; }
"""


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_tearsheet(res: dict[str, Any], figures: dict[str, Path]) -> str:
    """The full tear sheet as one self-contained HTML string."""
    m = res["meta"]
    md = markdown_it.MarkdownIt("commonmark").enable("table")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>nvquant tear sheet</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(m['target'])} daily strategies: out-of-sample tear sheet</h1>",
        f"<p class='meta'>Generated {m['generated_at']} from git {m['git_sha']}, data {m['data_hash']}, config {m['config_hash']}.</p>",
        "<h2>Summary</h2>",
        md.render(render_results_block(res)),
    ]
    for key, title in (("equity", "Equity"), ("drawdown", "Drawdowns"), ("rolling", "Rolling Sharpe and regimes"),
                       ("costs", "Cost sensitivity"), ("vol", "Volatility forecast"), ("peers", "Selection bias: peers"),
                       ("importance", "Feature importance (MDA, purged CV)"), ("cluster_importance", "Clustered importance")):  # fmt: skip
        if key in figures:
            parts += [f"<h2>{title}</h2>", _img(figures[key])]
    vol = res.get("vol_models") or {}
    if vol:
        parts += ["<h2>Volatility models</h2>", _table(
            ["model", "QLIKE", "MSE x1e8", "MZ alpha x1e4", "MZ beta", "MZ R2", "MZ Wald p"],
            [[k, fmt(v["qlike"], ".4f"), fmt(v["mse"] * 1e8, ".3f"), fmt(v["mz_alpha"] * 1e4, ".3f"), fmt(v["mz_beta"], ".3f"),
              fmt(v["mz_r2"], ".3f"), fmt(v["mz_wald_pvalue"], ".3f")] for k, v in vol.items()])]  # fmt: skip
    stress = res.get("stress") or {}
    if stress:
        rows = []
        for name, w in stress.items():
            if not w.get("covered"):
                rows.append([name, f"{w['start']} to {w['end']}", "no OOS coverage", "", ""])
                continue
            for strat, r in w["results"].items():
                rows.append(
                    [
                        name,
                        f"{w['start']} to {w['end']}",
                        label(strat, m["target"]),
                        fmt(r["total_return"], ".1f", True),
                        fmt(r["max_drawdown"], ".1f", True),
                    ]
                )
        parts += [
            "<h2>Stress windows (development period)</h2>",
            _table(["window", "dates", "strategy", "return", "max DD"], rows),
        ]
    peers = res.get("peers")
    if peers:
        parts += ["<h2>Peer study table</h2>", _table(
            ["ticker", "from", *[f"{mm} + {peers['sizing']}" for mm in peers["models"]], "buy and hold", "vol-targeted buy and hold"],
            [[r["peer"], r["first_date"], *[fmt(r["strategy"][mm]) for mm in peers["models"]], fmt(r["bh_target"]), fmt(r["bh_voltarget"])]
             for r in peers["rows"]])]  # fmt: skip
    parts.append("</body></html>")
    return "\n".join(parts)
