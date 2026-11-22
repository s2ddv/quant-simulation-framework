"""Plotly figures for the report dashboard, built from ``dashboard_data`` output.

Each function takes the dictionary returned by
``simulation.dashboard_data.build_dashboard_data`` and returns a Plotly figure.
Nothing here recomputes statistics, so the charts show exactly the numbers
stored in (or verified against) the saved report. Plotly is an optional
dependency: ``pip install -e ".[dashboard]"``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Dark terminal palette, shared with the Matplotlib/HTML figures in spirit.
BACKGROUND = "#0b0d10"
PANEL = "#12161b"
GRID = "#232a32"
TEXT = "#d8dde3"
MUTED = "#7d8894"
AMBER = "#f5a524"
CYAN = "#4cc3d9"
RED = "#ef5a5a"

SENSITIVITY_LABELS = {"var": "VaR (preço)", "mean": "E[S(T)]", "q0.05": "Quantil 5%",
                      "q0.5": "Mediana", "q0.95": "Quantil 95%"}


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Dashboard charts need plotly: pip install -e \".[dashboard]\"") from exc
    return go


def _layout(fig, *, height: int = 360, **axes: Any):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=height,
        font=dict(family="IBM Plex Mono, ui-monospace, monospace", size=12, color=MUTED),
        margin=dict(l=60, r=20, t=20, b=45), showlegend=False,
        hoverlabel=dict(bgcolor=BACKGROUND, bordercolor=AMBER, font=dict(color=TEXT)),
        **axes)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def future_dates(last_date: str, steps: int, horizon: float, step_index) -> list[str]:
    """Approximate calendar dates of simulation steps as weekdays after ``last_date``.

    Uses 252 steps per year of horizon; holidays are ignored, so dates are
    display aids only. The model's time axis remains ``time_years``.
    """
    sessions = max(1, round(horizon * 252))
    calendar = pd.bdate_range(pd.Timestamp(last_date), periods=sessions + 1)
    return [calendar[round(int(k) * sessions / steps)].date().isoformat() for k in step_index]


def fan_figure(data: dict, *, show_sample_paths: bool = True):
    """Observed history followed by simulated percentile bands and benchmarks."""
    go = _go()
    summary, history, fan = data["summary"], data["history"], data["fan"]
    dates = future_dates(summary["last_date"], summary["steps"], summary["horizon"], fan["step"])
    p = fan["percentiles"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["dates"], y=history["close"], mode="lines", name="history",
                             line=dict(color=TEXT, width=1.3),
                             hovertemplate="%{x}<br>%{y:.2f}<extra>fechamento</extra>"))
    for low, high, alpha, name in (("p5", "p95", 0.13, "P5-P95"), ("p25", "p75", 0.24, "P25-P75")):
        fig.add_trace(go.Scatter(x=dates, y=p[low], mode="lines", line=dict(width=0),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=p[high], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba(245,165,36,{alpha})",
                                 name=name, hoverinfo="skip"))
    if show_sample_paths:
        for values in fan["sample_paths"]:
            fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", hoverinfo="skip",
                                     name="sample path",
                                     line=dict(width=0.6, color="rgba(76,195,217,0.28)")))
    fig.add_trace(go.Scatter(x=dates, y=p["p50"], mode="lines", name="Median",
                             line=dict(color=AMBER, width=1.8),
                             hovertemplate="%{x}<br>mediana %{y:.2f}<extra>simulado</extra>"))
    analytical = fan["analytical"]
    for key, label, style in (("mean", "E[S(t)]", dict(color=AMBER, width=1, dash="dot")),
                              ("q0.05", "q5%", dict(color=MUTED, width=1, dash="dash")),
                              ("q0.95", "q95%", dict(color=MUTED, width=1, dash="dash"))):
        fig.add_trace(go.Scatter(x=dates, y=analytical[key], mode="lines", name=label, line=style,
                                 hovertemplate=label + " %{y:.2f}<extra>analítico</extra>"))
    fig.add_vline(x=dates[0], line=dict(color=AMBER, width=1, dash="dot"))
    return _layout(fig, height=440, hovermode="closest", yaxis_title="Preço")


def terminal_figure(data: dict):
    """Terminal-price histogram, lognormal overlay and VaR/CVaR thresholds."""
    go = _go()
    summary, terminal = data["summary"], data["terminal"]
    edges = terminal["bin_edges"]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    var_price, cvar_price = summary["var_threshold_price"], summary["cvar_threshold_price"]
    fig = go.Figure(go.Bar(
        x=mids, y=terminal["counts"], width=edges[1] - edges[0], name="simulated",
        marker_color=["rgba(239,90,90,0.75)" if m < var_price else "rgba(76,195,217,0.55)"
                      for m in mids],
        hovertemplate="%{x:.2f}<br>%{y} trajetórias<extra></extra>"))
    if terminal["lognormal"] is not None:
        curve = terminal["lognormal"]
        fig.add_trace(go.Scatter(x=curve["price"], y=curve["expected_count"], mode="lines",
                                 name="lognormal", line=dict(color=AMBER, width=1.6),
                                 hovertemplate="lognormal %{y:.1f}<extra></extra>"))
    for x, color, dash, label, y in ((summary["last_price"], TEXT, "dot", "S0", 1.0),
                                     (var_price, RED, "solid", f"VaR {summary['var']:.2f}", 0.9),
                                     (cvar_price, RED, "dash", f"CVaR {summary['cvar']:.2f}", 0.8)):
        fig.add_vline(x=x, line=dict(color=color, width=1, dash=dash))
        fig.add_annotation(x=x, y=y, yref="paper", text=label, showarrow=False,
                           xanchor="right" if x < summary["last_price"] else "left",
                           yanchor="top", font=dict(color=color))
    return _layout(fig, bargap=0, xaxis_title="S(T) · preço", yaxis_title="trajetórias")


def convergence_figure(data: dict):
    """|mean error| per nested prefix against the analytical z*SE envelope (log-log)."""
    go = _go()
    conv = data["convergence"]
    level = round(100 * conv["envelope_level"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conv["n_sims"], y=conv["mean_error_envelope"], mode="lines",
                             name=f"{level}% envelope", line=dict(color=MUTED, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=conv["n_sims"], y=conv["mean_absolute_error"], mode="lines+markers",
                             name="|mean error|", line=dict(color=AMBER, width=1.4),
                             hovertemplate="N=%{x}<br>|erro| %{y:.4f}<extra></extra>"))
    return _layout(fig, height=300, xaxis=dict(type="log", dtick=1, title="trajetórias (N)"),
                   yaxis=dict(type="log", dtick=1, title="preço"))


def sensitivity_figure(data: dict, metric: str = "var"):
    """Analytical metric over the (mu, sigma) scenario grid with the fitted point."""
    go = _go()
    grid = data["sensitivity"]
    if grid is None:
        raise ValueError("This report has no calibration sensitivity grid.")
    if metric not in grid["metrics"]:
        raise ValueError(f"Unknown sensitivity metric: {metric}")
    label = SENSITIVITY_LABELS.get(metric, metric)
    fig = go.Figure(go.Heatmap(
        x=grid["mu"], y=grid["sigma"], z=grid["metrics"][metric],
        colorscale=[[0, "#0e2a33"], [0.5, CYAN], [0.8, AMBER], [1, RED]],
        reversescale=metric.startswith("q"),
        colorbar=dict(thickness=10, outlinewidth=0, title=dict(text=label, side="right")),
        hovertemplate="mu %{x:.4f}<br>sigma %{y:.4f}<br>" + label + " %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[grid["baseline"]["mu"]], y=[grid["baseline"]["sigma"]],
                             mode="markers", name="fitted",
                             marker=dict(symbol="x-thin", size=12, line=dict(width=2, color=TEXT))))
    return _layout(fig, xaxis_title="mu (year^-1)", yaxis_title="sigma (year^-1/2)")


def backtest_figure(data: dict):
    """Realized historical losses against the forecast VaR; violations in red."""
    go = _go()
    bt = data["backtest"]
    if bt is None:
        raise ValueError("This report has no historical backtest.")
    fig = go.Figure(go.Bar(
        x=bt["target_date"], y=bt["realized_loss"], name="realized loss",
        marker_color=[RED if v else "rgba(125,136,148,0.45)" for v in bt["violation"]],
        hovertemplate="%{x}<br>perda %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bt["target_date"], y=bt["var"], mode="lines", name="VaR",
                             line=dict(color=AMBER, width=1.2),
                             hovertemplate="VaR %{y:.2f}<extra></extra>"))
    return _layout(fig, bargap=0, yaxis_title="perda (S0 − S1)")
