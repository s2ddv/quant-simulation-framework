"""Plotly dashboard figures show the stored dashboard series unchanged."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from simulation.dashboard_data import build_dashboard_data
from simulation.dashboard_figures import (backtest_figure, convergence_figure, fan_figure,
                                          future_dates, sensitivity_figure, terminal_figure)
from simulation.report import SimulationConfig, historical_snapshot, run_simulation


@pytest.fixture(scope="module")
def data():
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    prices = pd.Series(100 * np.exp(log_prices),
                       index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B"))
    snapshot = historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                                   retrieved_at="2026-03-01T12:00:00+00:00",
                                   source="deterministic_test_fixture")
    report, paths = run_simulation(
        snapshot, SimulationConfig(horizon=0.5, steps=20, n_sims=500, seed=3),
        {"calibration_observations": 10, "horizon_days": 1, "alpha": 0.95},
        {"mu_delta": 0.02, "sigma_relative_delta": 0.2, "scenario_levels": 3})
    return build_dashboard_data(report, paths, n_sample_paths=4)


def test_fan_figure_plots_history_bands_and_analytical_series(data):
    figure = fan_figure(data)
    by_name = {trace.name: trace for trace in figure.data}
    np.testing.assert_array_equal(by_name["history"].y, data["history"]["close"])
    np.testing.assert_array_equal(by_name["Median"].y, data["fan"]["percentiles"]["p50"])
    np.testing.assert_array_equal(by_name["E[S(t)]"].y, data["fan"]["analytical"]["mean"])
    assert sum(trace.name == "sample path" for trace in figure.data) == 4
    hidden = fan_figure(data, show_sample_paths=False)
    assert not any(trace.name == "sample path" for trace in hidden.data)
    assert by_name["Median"].x[0] == data["summary"]["last_date"]


def test_future_dates_are_weekdays_and_cover_the_horizon():
    dates = future_dates("2026-01-02", 4, 1 / 63, range(5))
    assert dates == ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    assert all(pd.Timestamp(d).weekday() < 5 for d in future_dates("2026-01-02", 252, 1.0, range(253)))


def test_terminal_sensitivity_backtest_and_convergence_figures(data):
    terminal = terminal_figure(data)
    assert sum(terminal.data[0].y) == data["terminal"]["n"]
    assert {shape.x0 for shape in terminal.layout.shapes} == {
        data["summary"]["last_price"], data["summary"]["var_threshold_price"],
        data["summary"]["cvar_threshold_price"]}
    heat = sensitivity_figure(data, "mean").data[0]
    np.testing.assert_array_equal(heat.z, data["sensitivity"]["metrics"]["mean"])
    with pytest.raises(ValueError):
        sensitivity_figure(data, "nope")
    bars = backtest_figure(data).data[0]
    assert len(bars.y) == data["backtest"]["metrics"]["forecast_count"]
    assert convergence_figure(data).layout.xaxis.type == "log"


def test_missing_optional_sections_raise_clear_errors(data):
    bare = dict(data, sensitivity=None, backtest=None)
    with pytest.raises(ValueError, match="sensitivity"):
        sensitivity_figure(bare)
    with pytest.raises(ValueError, match="backtest"):
        backtest_figure(bare)
