"""Optional interactive (Plotly) percentile fan chart saved as standalone HTML."""

from pathlib import Path

import numpy as np

# Same dark palette as the Matplotlib figures in plotting.py.
_BACKGROUND = "#101820"
_PANEL = "#17232e"
_TEXT = "#f2f5f7"
_MUTED = "#b8c7d1"
_GRID = "#2c3b49"
_AMBER = "#ffbd59"
_BLUE = "122,184,255"


def plot_fan_html(
    paths: np.ndarray,
    title: str | None = None,
    save_path: str | Path | None = None,
    n_sample_paths: int = 40,
):
    """Build an interactive dark fan chart; optionally write standalone HTML.

    Bands are pointwise 5-95% and 25-75% percentiles over ALL simulations, with
    the median and a few faint sample paths for texture. Plotly is an optional
    dependency (``pip install -e ".[interactive]"``). The HTML embeds plotly.js,
    so it opens offline (about 5 MB).

    Args:
        paths: Prices shaped (steps + 1, n_sims), sharing one initial price.
        title: Optional title.
        save_path: Optional .html destination; parent directories are created.
        n_sample_paths: Number of faint example paths (0 disables them).

    Returns:
        The Plotly figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive charts need plotly: pip install -e \".[interactive]\""
        ) from exc
    from simulation.plotting import _validate_paths

    paths = _validate_paths(paths)
    if n_sample_paths < 0:
        raise ValueError("n_sample_paths must be non-negative.")
    s0 = float(paths[0, 0])
    steps = np.arange(paths.shape[0])
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)

    fig = go.Figure()
    sampled = min(n_sample_paths, paths.shape[1])
    for index in np.linspace(0, paths.shape[1] - 1, sampled, dtype=int):
        fig.add_trace(go.Scatter(
            x=steps, y=paths[:, index], mode="lines", hoverinfo="skip",
            line=dict(width=0.6, color="rgba(184,199,209,0.13)"),
            showlegend=False))

    def band(low, high, alpha, name):
        fig.add_trace(go.Scatter(x=steps, y=high, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=steps, y=low, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba({_BLUE},{alpha})",
                                 name=name, hoverinfo="skip"))

    band(p5, p95, 0.16, "90% of simulations (P5-P95)")
    band(p25, p75, 0.36, "50% of simulations (P25-P75)")
    for name, values in (("P95", p95), ("P5", p5)):
        fig.add_trace(go.Scatter(
            x=steps, y=values, mode="lines", name=name, showlegend=False,
            line=dict(width=1, color=f"rgba({_BLUE},0.55)", dash="dot"),
            hovertemplate=name + ": %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=steps, y=p50, mode="lines", name="Median",
        line=dict(width=3, color=_TEXT),
        hovertemplate="Median: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[0, steps[-1]], y=[s0, s0], mode="lines", name=f"Initial price: {s0:.2f}",
        line=dict(color=_AMBER, width=1.5, dash="dash"), hoverinfo="skip"))

    heading = title if title is not None else "GBM simulated price fan chart"
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_BACKGROUND, plot_bgcolor=_PANEL,
        font=dict(family="Inter, system-ui, sans-serif", size=14, color=_TEXT),
        hoverlabel=dict(bgcolor=_BACKGROUND, bordercolor=_GRID, font=dict(color=_TEXT)),
        title=dict(text=f"<b>{heading}</b><br><sup>{paths.shape[1]:,} simulations; "
                        "bands = percentiles of the simulations at each step "
                        "· not a forecast</sup>", x=0.02),
        hovermode="x unified",
        xaxis=dict(title="Simulation step", showgrid=False, zeroline=False),
        yaxis=dict(title="Price (asset currency)", gridcolor=_GRID, zeroline=False),
        legend=dict(orientation="h", y=-0.18, font=dict(color=_MUTED)),
        margin=dict(l=60, r=30, t=90, b=70), height=650)

    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        html = fig.to_html(include_plotlyjs=True).replace(
            "<head>",
            f"<head><style>html,body{{margin:0;background:{_BACKGROUND}}}</style>", 1)
        destination.write_text(html, encoding="utf-8")
    return fig
