"""Financial labels and plotted counts for the dark chart style."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simulation.plotting import plot_final_distribution, plot_paths


def test_terminal_histogram_preserves_counts_and_loss_var_conversion(tmp_path):
    paths = np.array([[100.0] * 5, [80.0, 90.0, 95.0, 110.0, 120.0]])
    figure = plot_final_distribution(paths, var=10.0,
                                     save_path=tmp_path / "distribution.png")
    try:
        axis = figure.axes[0]
        bars = axis.containers
        # Every terminal observation contributes once, including one at the VaR line.
        assert sum(patch.get_height() for container in bars
                   for patch in container.patches) == 5
        var_lines = [line for line in axis.lines
                     if line.get_label().startswith("Price at VaR")]
        assert len(var_lines) == 1
        assert np.all(np.asarray(var_lines[0].get_xdata()) == 90.0)
        assert axis.get_xlabel() == "Final price (asset currency)"
        assert (tmp_path / "distribution.png").is_file()
    finally:
        plt.close(figure)


def test_path_figure_uses_steps_and_preserves_caller_style(tmp_path):
    original_facecolor = matplotlib.rcParams["figure.facecolor"]
    paths = np.array([[100.0, 100.0], [90.0, 110.0], [80.0, 120.0]])
    figure = plot_paths(paths, save_path=tmp_path / "paths.png")
    try:
        assert figure.axes[0].get_xlabel() == "Simulation step"
        assert len(figure.axes[0].lines) == 4  # two paths, median, initial price
        median = next(line for line in figure.axes[0].lines
                      if line.get_label().startswith("Median of displayed paths"))
        np.testing.assert_allclose(median.get_ydata(), [100.0, 100.0, 100.0])
        assert figure.axes[1].get_ylabel() == "Terminal price (asset currency)"
        assert matplotlib.rcParams["figure.facecolor"] == original_facecolor
        assert (tmp_path / "paths.png").is_file()
    finally:
        plt.close(figure)


def test_path_fan_defaults_to_300_and_samples_across_all_simulations(tmp_path):
    # Each simulated path has a unique terminal price, allowing selection checks.
    paths = np.vstack((np.full(1000, 100.0), np.arange(1000, dtype=float)))
    figure = plot_paths(paths, save_path=tmp_path / "fan.png")
    try:
        axis = figure.axes[0]
        simulated = axis.lines[:300]
        assert len(axis.lines) == 302  # 300 samples, median, initial marker
        assert simulated[0].get_ydata()[-1] == 0.0
        assert simulated[-1].get_ydata()[-1] == 999.0
        assert all(line.get_linewidth() < 1 for line in simulated)
        assert any(text.get_text() == "Simulated paths: 300 of 1000"
                   for text in axis.get_legend().get_texts())
        assert len(figure.axes) == 2  # terminal-price colorbar
        assert (tmp_path / "fan.png").is_file()
    finally:
        plt.close(figure)
