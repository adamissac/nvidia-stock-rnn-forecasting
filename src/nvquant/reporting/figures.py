"""Static figures for the tear sheet and README (matplotlib, light surface).

Colors follow a validated categorical order: at most three series per chart
(blue, orange, aqua), which stays distinguishable under color-vision
deficiency. Every multi-series chart has a legend, lines are 2px, and grids
are recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SHADE = "#d9d8d2"


def _style(ax: Axes, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(width: float = 9.0, height: float = 3.6) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _label_end(ax: Axes, s: pd.Series, text: str, color: str) -> None:
    s = s.dropna()
    if len(s):
        ax.annotate(text, (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TEXT, fontsize=8, va="center")  # fmt: skip
        ax.plot([s.index[-1]], [s.iloc[-1]], "o", color=color, markersize=4)


def equity_curves(
    returns: pd.DataFrame, labels: dict[str, str], path: Path, log_scale: bool = True
) -> Path:
    """Growth of $1 for up to three series."""
    fig, ax = _fig()
    for i, (col, label) in enumerate(list(labels.items())[:3]):
        eq = (1 + returns[col].fillna(0.0)).cumprod()
        ax.plot(eq.index, eq, color=SERIES[i], linewidth=2, label=label)
        _label_end(ax, eq, f"{eq.iloc[-1]:.1f}x", SERIES[i])
    if log_scale:
        ax.set_yscale("log")
    _style(
        ax,
        "Growth of $1, out of sample, net of costs",
        "equity (log scale)" if log_scale else "equity",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def drawdowns(returns: pd.DataFrame, labels: dict[str, str], path: Path) -> Path:
    """Drawdown from peak for up to three series."""
    fig, ax = _fig(height=3.0)
    for i, (col, label) in enumerate(list(labels.items())[:3]):
        eq = (1 + returns[col].fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd * 100, color=SERIES[i], linewidth=1.5, label=label)
    _style(ax, "Drawdown from peak", "percent")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, path)


def rolling_sharpe_regimes(
    returns: pd.DataFrame, labels: dict[str, str], p_high: pd.Series, path: Path, window: int = 252
) -> Path:
    """Trailing one-year Sharpe with high-vol regime periods shaded."""
    fig, ax = _fig()
    hi = (p_high.reindex(returns.index) > 0.5).to_numpy()
    idx = returns.index
    start = None
    for k, flag in enumerate(hi):
        if flag and start is None:
            start = idx[k]
        if (not flag or k == len(hi) - 1) and start is not None:
            ax.axvspan(start, idx[k], color=SHADE, alpha=0.5, linewidth=0)
            start = None
    for i, (col, label) in enumerate(list(labels.items())[:3]):
        r = returns[col]
        rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
        ax.plot(rs.index, rs, color=SERIES[i], linewidth=1.5, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Rolling one-year Sharpe (shaded: filtered HMM high-vol regime)", "Sharpe")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def vol_forecast_vs_realized(vol: pd.DataFrame, model: str, path: Path) -> Path:
    """Annualized forecast vol against a 21-day average of realized vol."""
    fig, ax = _fig()
    realized = np.sqrt(vol["rv_next"].rolling(21).mean() * 252)
    fc = np.sqrt(vol[model] * 252)
    ax.plot(
        realized.index,
        realized * 100,
        color=SERIES[1],
        linewidth=1.2,
        label="realized (21-day avg of proxy)",
    )
    ax.plot(fc.index, fc * 100, color=SERIES[0], linewidth=1.2, label=f"{model} forecast")
    _style(ax, "Volatility forecast vs realized", "annualized vol, percent")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def importance_bars(
    values: dict[str, float], errors: dict[str, float] | None, path: Path, title: str, top: int = 15
) -> Path:
    """Horizontal bars for the top features, largest at the top."""
    s = pd.Series(values).sort_values(ascending=False).head(top)[::-1]
    fig, ax = _fig(width=7.5, height=0.28 * len(s) + 1.0)
    err = None if errors is None else [errors.get(k, 0.0) for k in s.index]
    ax.barh(
        s.index,
        s.to_numpy(),
        xerr=err,
        color=SERIES[0],
        height=0.6,
        error_kw={"ecolor": MUTED, "elinewidth": 0.8},
    )
    ax.axvline(0, color=MUTED, linewidth=0.8)
    _style(ax, title)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def cost_sensitivity(sweep: pd.DataFrame, labels: dict[str, str], path: Path) -> Path:
    """Net Sharpe as the per-side cost grows from 0 to 20 bps."""
    fig, ax = _fig(width=7.0, height=3.4)
    x = [float(c) for c in sweep.columns]
    for i, (row, label) in enumerate(list(labels.items())[:3]):
        y = sweep.loc[row].to_numpy(dtype=float)
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("cost per side, bps", color=MUTED, fontsize=9)
    _style(ax, "Cost sensitivity", "net Sharpe")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def peer_dots(peers: list[dict[str, object]], strategy: str, path: Path) -> Path:
    """Strategy Sharpe versus buy-and-hold and vol-targeted buy-and-hold on every ticker."""
    names = [str(p["peer"]) for p in peers]
    fig, ax = _fig(width=8.0, height=3.4)
    x = np.arange(len(names))
    series = [
        (strategy, "strategy"),
        ("bh_voltarget", "vol-targeted buy and hold"),
        ("bh_target", "buy and hold"),
    ]
    for i, (key, label) in enumerate(series):
        y = [float(p["sharpe"].get(key, np.nan)) for p in peers]  # type: ignore[attr-defined]
        ax.plot(x + (i - 1) * 0.18, y, "o", color=SERIES[i], markersize=6, label=label)
    ax.set_xticks(x, names)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Same pipeline on every ticker (development period, net Sharpe)", "net Sharpe")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)
