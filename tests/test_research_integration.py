"""Offline integration of rolling forecasts and parameter scenarios in reports."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import main as app
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                               run_simulation, write_report)


@pytest.mark.parametrize("levels", [None, 3, 5, 7, 9, 100])
def test_research_report_round_trip_and_tamper_detection(tmp_path, levels):
    returns = 0.001 + 0.003 * np.sin(np.arange(51))
    prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]),
                       index=pd.date_range("2025-01-01", periods=52, freq="B"))
    snapshot = historical_snapshot(prices, ticker="SYN", period="fixture",
                                   source="synthetic", retrieved_at="2025-05-01T00:00:00Z")
    sensitivity_config = {"mu_delta": 0.03, "sigma_relative_delta": 0.25}
    if levels is not None:
        sensitivity_config["scenario_levels"] = levels
    report, _ = run_simulation(
        snapshot, SimulationConfig(steps=2, n_sims=10),
        {"calibration_observations": 20, "horizon_days": 5, "alpha": 0.9,
         "calibration_mode": "rolling", "recalibration_frequency": 2},
        sensitivity_config)
    path = tmp_path / "research.json"
    write_report(report, path)
    assert read_report(path) == report
    assert report["out_of_sample"]["evaluation"]["metrics"]["forecast_count"] == 6
    assert len(report["calibration_sensitivity"]["scenarios"]) == (levels or 3) ** 2
    altered = deepcopy(report)
    altered["calibration_sensitivity"]["ranges"]["var"]["max"] += 1
    write_report(altered, path)
    with pytest.raises(ValueError, match="calibration_sensitivity"):
        read_report(path)

    altered = deepcopy(report)
    altered["calibration_sensitivity"]["scenarios"][1]["quantiles"]["0.05"] += 1
    write_report(altered, path)
    with pytest.raises(ValueError, match="calibration_sensitivity"):
        read_report(path)

    altered = deepcopy(report)
    altered["calibration_sensitivity"]["configuration"]["scenario_levels"] = (
        7 if levels == 5 else 5)
    write_report(altered, path)
    with pytest.raises(ValueError, match="calibration_sensitivity"):
        read_report(path)

    if levels in (5, 7, 9, 100):
        altered = deepcopy(report)
        del altered["calibration_sensitivity"]["configuration"]["scenario_levels"]
        write_report(altered, path)
        with pytest.raises(ValueError, match="calibration_sensitivity"):
            read_report(path)


def test_cli_accepts_single_parameter_sensitivity_and_rejects_invalid_spread():
    parser = app.build_parser()
    args = parser.parse_args(["--calibration-sensitivity", "--mu-delta", "0",
                              "--sigma-relative-delta", "0"])
    assert args.mu_delta == args.sigma_relative_delta == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--sigma-relative-delta", "1.1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--mu-delta", "-0.01"])
