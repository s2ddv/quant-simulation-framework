import numpy as np
import pytest

pytest.importorskip("plotly")

from simulation.interactive import plot_fan_html


def test_fan_html_uses_all_simulations_and_writes_standalone_file(tmp_path):
    paths = np.array([[100.0] * 4, [80.0, 90.0, 110.0, 120.0], [70.0, 95.0, 105.0, 130.0]])
    destination = tmp_path / "out" / "paths.html"
    figure = plot_fan_html(paths, n_sample_paths=2, save_path=destination)
    median = next(trace for trace in figure.data if trace.name == "Median")
    np.testing.assert_allclose(median.y, np.median(paths, axis=1))
    assert destination.is_file() and "plotly" in destination.read_text(encoding="utf-8")[:200000].lower()


def test_fan_html_rejects_invalid_paths():
    with pytest.raises(ValueError):
        plot_fan_html(np.array([[100.0, 101.0], [100.0, np.nan]]))
