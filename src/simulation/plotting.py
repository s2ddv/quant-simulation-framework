"""Price paths, terminal distributions and convergence figures."""

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np


# Local palette: each figure keeps its own style without changing rcParams.
_BACKGROUND = "#101820"
_PANEL = "#17232e"
_TEXT = "#f2f5f7"
_MUTED = "#b8c7d1"
_GRID = "#41515f"
_AMBER = "#ffbd59"
_TEAL = "#4dd6c2"
_RED = "#ff7285"
_BLUE = "#7ab8ff"
_PURPLE = "#d29bff"


def _style_axes(figure: Figure, axes) -> None:
    """Apply the dark financial-terminal style to this figure only."""
    figure.patch.set_facecolor(_BACKGROUND)
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_MUTED, labelsize=9, length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(color=_GRID, alpha=0.42, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(_MUTED)
        ax.yaxis.label.set_color(_MUTED)
        ax.title.set_color(_TEXT)


def _style_legend(ax) -> None:
    legend = ax.legend(loc="best", fontsize=9, frameon=True,
                       facecolor=_BACKGROUND, edgecolor=_GRID, framealpha=0.96)
    for label in legend.get_texts():
        label.set_color(_TEXT)


def _validate_paths(paths: np.ndarray) -> np.ndarray:
    """Return a finite price matrix with a shared initial price."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] == 0:
        raise ValueError(
            "paths must be 2D with at least 2 rows and 1 simulation."
        )
    if not np.isfinite(paths).all():
        raise ValueError("paths must contain only finite prices.")
    if not np.all(paths[0] == paths[0, 0]):
        raise ValueError("All simulations must share the same initial price.")
    return paths


def _save_figure(figure: Figure, save_path: str | Path | None) -> None:
    """Save the current figure, creating parent directories when needed."""
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, facecolor=figure.get_facecolor())


def plot_paths(
    paths: np.ndarray,
    n_display: int = 300,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot evenly spaced sample paths without opening a window.

    Args:
        paths: Prices shaped (steps + 1, n_sims), including initial prices.
        n_display: Positive maximum number of displayed paths (default: 300).
        title: Optional chart title.
        save_path: Optional output file; parent directories are created.

    Returns:
        The figure, left open for the caller to display or close.

    Raises:
        ValueError: If paths or n_display are invalid.
    """
    paths = _validate_paths(paths)
    if (isinstance(n_display, (bool, np.bool_))
            or not isinstance(n_display, (int, np.integer))
            or n_display <= 0):
        raise ValueError("n_display must be a positive integer.")

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    figure, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    _style_axes(figure, ax)
    shown = min(n_display, paths.shape[1])
    indices = np.linspace(0, paths.shape[1] - 1, num=shown, dtype=int)
    selected = paths[:, indices]
    final_prices = selected[-1]
    color_map = colormaps["turbo"]
    distinct_finals = final_prices.min() < final_prices.max()
    normalization = Normalize(vmin=final_prices.min(), vmax=final_prices.max())
    for index, terminal in enumerate(final_prices):
        color = color_map(normalization(terminal)) if distinct_finals else _TEAL
        ax.plot(selected[:, index], color=color, alpha=0.54, linewidth=0.7,
                zorder=2)
    median_line, = ax.plot(np.median(selected, axis=1), color=_TEXT,
                           linewidth=2.2, zorder=4,
                           label="Median of displayed paths (each step)")
    initial_line = ax.axhline(paths[0, 0], color=_AMBER, linestyle="--",
                              linewidth=1.1, alpha=0.85, zorder=3,
                              label=f"Initial price: {paths[0, 0]:.2f}")
    sample_handle = Line2D([], [], color=_BLUE, linewidth=1.3,
                           label=f"Simulated paths: {shown} of {paths.shape[1]}")
    ax.legend(handles=(sample_handle, median_line, initial_line), loc="best",
              fontsize=9, facecolor=_BACKGROUND, edgecolor=_GRID,
              framealpha=0.96, labelcolor=_TEXT)
    if distinct_finals:
        colorbar = figure.colorbar(ScalarMappable(norm=normalization,
                                                  cmap=color_map), ax=ax,
                                   pad=0.02, fraction=0.04)
        colorbar.ax.set_facecolor(_PANEL)
        colorbar.ax.tick_params(colors=_MUTED, labelsize=9, length=0)
        colorbar.outline.set_visible(False)
        colorbar.set_label("Terminal price (asset currency)", color=_MUTED)
    ax.set_title(title if title is not None else "GBM simulated price paths")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Price (asset currency)")
    _save_figure(figure, save_path)
    return figure


def plot_final_distribution(
    paths: np.ndarray,
    var: float | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot terminal prices and reference lines without opening a window.

    Args:
        paths: Prices shaped (steps + 1, n_sims), including initial prices.
        var: Optional loss VaR in price units; its price line is s0 - var.
        title: Optional chart title.
        save_path: Optional output file; parent directories are created.

    Returns:
        The figure, left open for the caller to display or close.

    Raises:
        ValueError: If paths are invalid or var is nonfinite.
    """
    paths = _validate_paths(paths)
    if var is not None and not np.isfinite(var):
        raise ValueError("var must be finite when provided.")

    import matplotlib.pyplot as plt

    s0 = paths[0, 0]
    final_prices = paths[-1]
    figure, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    _style_axes(figure, ax)
    if var is None:
        ax.hist(final_prices, bins=50, color=_TEAL, alpha=0.84,
                edgecolor=_PANEL, linewidth=0.4, label="Terminal prices")
    else:
        var_price = s0 - var
        # Shared bins and stacking preserve the exact count when a bin crosses VaR.
        ax.hist((final_prices[final_prices < var_price],
                 final_prices[final_prices >= var_price]), bins=50,
                stacked=True, color=(_RED, _TEAL), alpha=0.84,
                edgecolor=_PANEL, linewidth=0.4,
                label=("Below price at VaR", "At/above price at VaR"))
        ax.axvline(var_price, color=_RED, linestyle=":", linewidth=2.0,
                   label=f"Price at VaR (s0 - VaR): {var_price:.2f}")
    ax.axvline(s0, color=_AMBER, linestyle="--", linewidth=1.7,
               label=f"Initial price: {s0:.2f}")
    mean = final_prices.mean()
    ax.axvline(mean, color=_BLUE, linewidth=1.8, label=f"Mean terminal price: {mean:.2f}")
    ax.set_title(title if title is not None else "GBM final price distribution")
    ax.set_xlabel("Final price (asset currency)")
    ax.set_ylabel("Number of simulations")
    _style_legend(ax)
    _save_figure(figure, save_path)
    return figure


def plot_convergence(rows, path):
    # Figure + Agg avoids requiring a desktop backend for --no-plot runs.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    _style_axes(figure, axes)
    counts = [row["n_sims"] for row in rows]
    for ax, metric, color in zip(axes[:2], ("mean", "variance"), (_TEAL, _AMBER)):
        ax.plot(counts, [row[metric]["absolute_error"] for row in rows],
                "o-", color=color, markersize=4.5, linewidth=1.9,
                label="Observed absolute error")
        ax.set_title(f"Terminal {metric}")
    axes[0].plot(counts, [row["mean_standard_error"] for row in rows],
                 "--", color=_RED, linewidth=1.6,
                 label="Theoretical mean SE (1/sqrt(N))")
    quantile_colors = (_BLUE, _PURPLE, _TEAL, _AMBER, _RED)
    for index, p in enumerate(rows[0]["quantiles"]):
        axes[2].plot(counts, [row["quantiles"][p]["absolute_error"] for row in rows],
                     "o-", color=quantile_colors[index % len(quantile_colors)],
                     markersize=4.5, linewidth=1.9, label=f"Price quantile p={p}")
    axes[2].set_title("Terminal quantiles")
    for index, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlabel("Simulations (nested N)")
        ax.set_ylabel("Absolute error (asset currency²)" if index == 1
                      else "Absolute error (asset currency)")
        _style_legend(ax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    return figure
