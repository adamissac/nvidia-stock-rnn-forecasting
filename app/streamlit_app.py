"""Streamlit viewer for the generated reports. It reads reports/ only and never trains.

Run with ``make app``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPORTS = Path(os.environ.get("NVQUANT_REPORTS", Path(__file__).resolve().parents[1] / "reports"))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


@st.cache_data
def load() -> dict[str, object]:
    """Read every artifact the app shows."""
    bt = REPORTS / "backtest"
    return {
        "results": json.loads((REPORTS / "results.json").read_text()),
        "net": pd.read_parquet(bt / "net.parquet"),
        "positions": pd.read_parquet(bt / "positions.parquet"),
        "sweep": pd.read_parquet(bt / "cost_sweep.parquet"),
        "capacity": pd.read_parquet(bt / "capacity.parquet"),
        "regime": pd.read_parquet(REPORTS / "app" / "regime.parquet"),
        "vol": pd.read_parquet(REPORTS / "app" / "vol.parquet"),
        "importance": json.loads((REPORTS / "evaluation" / "importance.json").read_text())
        if (REPORTS / "evaluation" / "importance.json").exists()
        else None,
    }


def long_frame(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Wide dates x series frame to long form for altair."""
    out = df.reset_index(names="date").melt("date", var_name="series", value_name=value)
    return out.dropna()


def line_chart(df: pd.DataFrame, value: str, title: str, log: bool = False) -> alt.Chart:
    """A line chart with a hover rule and tooltip."""
    data = long_frame(df, value)
    scale = alt.Scale(type="log") if log else alt.Scale(zero=False)
    color = alt.Color("series:N", scale=alt.Scale(range=COLORS), legend=alt.Legend(orient="top"))
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None), y=alt.Y(f"{value}:Q", scale=scale), color=color
    )
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=40, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    rule = (
        alt.Chart(data)
        .mark_rule(color="#8a8984")
        .encode(x="date:T", tooltip=["date:T", "series:N", alt.Tooltip(f"{value}:Q", format=".3f")])
        .add_params(hover)
        .transform_filter(hover)
    )
    return (lines + points + rule).properties(title=title, height=320)


def main() -> None:
    """Render the app."""
    st.set_page_config(page_title="nvquant results", layout="wide")
    if not (REPORTS / "results.json").exists():
        st.error(f"No results in {REPORTS}. Run `make reproduce` first.")
        return
    d = load()
    res = d["results"]  # type: ignore[index]
    meta, head = res["meta"], res["headline"]
    net: pd.DataFrame = d["net"]  # type: ignore[assignment]
    st.title(f"{meta['target']}: out-of-sample strategies")
    st.caption(
        f"Decisions {meta['oos_start']} to {meta['oos_end']}, net of costs. git {meta['git_sha']}, data {meta['data_hash']}."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best dev strategy Sharpe", f"{head['best_sharpe']:.2f}")
    c2.metric("Vol-targeted buy and hold", f"{head['voltarget_sharpe']:.2f}")
    c3.metric("Deflated Sharpe (best)", f"{head['best_dsr']:.2f}")
    c4.metric("PBO", f"{head['pbo']:.2f}")

    defaults = [head["best_strategy"], "bh_voltarget", "bh_target"]
    picks = st.multiselect(
        "Strategies (up to 3)",
        list(net.columns),
        default=[c for c in defaults if c in net],
        max_selections=3,
    )
    if not picks:
        st.info("Pick at least one strategy.")
        return
    sub = net[picks].fillna(0.0)
    eq = (1 + sub).cumprod()
    tab_eq, tab_dd, tab_rs, tab_vol, tab_imp, tab_cost, tab_tbl = st.tabs(
        ["Equity", "Drawdowns", "Rolling Sharpe", "Volatility", "Importance", "Costs", "Table"]
    )
    with tab_eq:
        st.altair_chart(
            line_chart(eq, "equity", "Growth of $1 (log scale)", log=True), width="stretch"
        )
    with tab_dd:
        st.altair_chart(
            line_chart(eq / eq.cummax() - 1, "drawdown", "Drawdown from peak"), width="stretch"
        )
    with tab_rs:
        rs = sub.rolling(252).mean() / sub.rolling(252).std() * np.sqrt(252)
        regime: pd.DataFrame = d["regime"]  # type: ignore[assignment]
        hi = regime["regime_p_high"].reindex(rs.index).gt(0.5)
        bands = pd.DataFrame({"start": rs.index[hi.to_numpy()], "end": rs.index[hi.to_numpy()]})
        shade = alt.Chart(bands).mark_rule(color="#d9d8d2", opacity=0.35).encode(x="start:T")
        st.altair_chart(
            shade
            + line_chart(rs, "sharpe", "Rolling one-year Sharpe (grey: filtered high-vol regime)"),
            width="stretch",
        )
    with tab_vol:
        vol: pd.DataFrame = d["vol"]  # type: ignore[assignment]
        model = st.selectbox("Vol model", [c for c in vol.columns if c != "rv_next"])
        v = pd.DataFrame(
            {
                "realized (21d avg)": np.sqrt(vol["rv_next"].rolling(21).mean() * 252),
                f"{model} forecast": np.sqrt(vol[model] * 252),
            }
        )
        st.altair_chart(
            line_chart(v, "vol", "Annualized vol: forecast vs realized"), width="stretch"
        )
    with tab_imp:
        imp = d["importance"]
        if imp:
            kind = st.radio("Measure", ["mda", "shap", "mdi", "cluster_mda"], horizontal=True)
            s = pd.Series(imp[kind]).sort_values(ascending=False).head(20)  # type: ignore[index]
            chart = (
                alt.Chart(s.rename_axis("feature").rename("importance").reset_index())
                .mark_bar(color=COLORS[0])
                .encode(
                    x="importance:Q",
                    y=alt.Y("feature:N", sort="-x"),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".4g")],
                )
            )
            st.altair_chart(chart.properties(height=480), width="stretch")
    with tab_cost:
        sweep: pd.DataFrame = d["sweep"]  # type: ignore[assignment]
        rows = sweep.loc[[p for p in picks if p in sweep.index]].T
        rows.index = rows.index.astype(float)
        data = rows.reset_index(names="bps").melt("bps", var_name="series", value_name="sharpe")
        chart = (
            alt.Chart(data)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("bps:Q", title="cost per side (bps)"),
                y="sharpe:Q",
                color=alt.Color("series:N", scale=alt.Scale(range=COLORS)),
                tooltip=["series", "bps", alt.Tooltip("sharpe:Q", format=".2f")],
            )
        )
        st.altair_chart(chart.properties(height=320), width="stretch")
        st.dataframe(d["capacity"])
    with tab_tbl:
        st.dataframe(pd.DataFrame(res["strategies"]).T)


if __name__ == "__main__":
    main()
