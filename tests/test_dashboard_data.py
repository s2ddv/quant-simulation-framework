"""Chart-ready dashboard series derived from saved GBM reports (offline)."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm

from simulation.dashboard_data import (build_dashboard_data, fan_chart,
                                       lognormal_density, regenerate_paths,
                                       sensitivity_grid, terminal_distribution)
from simulation.report import SimulationConfig, historical_snapshot, run_simulation


@pytest.fixture(scope="module")
def run():
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    prices = pd.Series(100 * np.exp(log_prices),
                       index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B"))
    snapshot = historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                                   retrieved_at="2026-03-01T12:00:00+00:00",
                                   source="deterministic_test_fixture")
    config = SimulationConfig(horizon=0.5, steps=20, n_sims=2000, seed=7, confidence=0.95)
    report, paths = run_simulation(
        snapshot, config,
        {"calibration_observations": 10, "horizon_days": 1, "alpha": 0.95,
         "calibration_mode": "rolling", "recalibration_frequency": 1},
        {"mu_delta": 0.02, "sigma_relative_delta": 0.2, "scenario_levels": 3})
    return report, paths


def test_build_is_json_safe_and_regeneration_matches_given_paths(run):
    report, paths = run
    given = build_dashboard_data(report, paths)
    regenerated = build_dashboard_data(report)
    assert given == regenerated
    json.dumps(given, allow_nan=False)
    assert given["summary"]["var_threshold_price"] == report["last_price"] - report["results"]["risk"]["var"]
    assert len(given["history"]["close"]) == report["historical_data"]["count"]


def test_fan_bands_are_sample_percentiles_and_analytical_mean_is_exact(run):
    report, paths = run
    mu, sigma = report["parameters"]["mu"], report["parameters"]["sigma"]
    fan = fan_chart(paths, mu=mu, sigma=sigma, horizon=0.5, n_sample_paths=5)
    expected = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    for key, row in zip(("p5", "p25", "p50", "p75", "p95"), expected):
        np.testing.assert_allclose(fan["percentiles"][key], row, rtol=0, atol=0)
    times = np.arange(21) * 0.5 / 20
    np.testing.assert_allclose(fan["analytical"]["mean"],
                               report["last_price"] * np.exp(mu * times), rtol=1e-12)
    assert fan["analytical"]["q0.05"][0] == fan["analytical"]["q0.95"][0] == report["last_price"]
    assert len(fan["sample_paths"]) == 5 and fan["sample_path_indices"][-1] == paths.shape[1] - 1


def test_fan_thinning_keeps_first_and_last_rows(run):
    report, paths = run
    fan = fan_chart(paths, mu=report["parameters"]["mu"],
                    sigma=report["parameters"]["sigma"], horizon=0.5, max_points=6)
    assert fan["step"][0] == 0 and fan["step"][-1] == 20 and len(fan["step"]) == 6
    assert fan["time_years"][-1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fan_chart(paths, mu=0.1, sigma=0.2, horizon=0.5, max_points=1)


def test_lognormal_density_matches_scipy_and_histogram_counts_add_up(run):
    report, paths = run
    s0, mu, sigma = report["last_price"], report["parameters"]["mu"], report["parameters"]["sigma"]
    x = np.array([-1.0, 0.0, 80.0, 100.0, 130.0])
    reference = lognorm.pdf(x, s=sigma * np.sqrt(0.5), scale=s0 * np.exp((mu - sigma**2 / 2) * 0.5))
    np.testing.assert_allclose(lognormal_density(x, s0=s0, mu=mu, sigma=sigma, horizon=0.5),
                               reference, rtol=1e-12, atol=0)
    dist = terminal_distribution(paths[-1], s0=s0, mu=mu, sigma=sigma, horizon=0.5, bins=30)
    assert sum(dist["counts"]) == dist["n"] == 2000
    assert len(dist["bin_edges"]) == 31
    degenerate = terminal_distribution(np.full(3, 100.0), s0=100.0, mu=0.0, sigma=0.0, horizon=0.5)
    assert degenerate["lognormal"] is None


def test_sensitivity_grid_and_backtest_reshape_saved_sections(run):
    report, paths = run
    data = build_dashboard_data(report, paths)
    grid = data["sensitivity"]
    assert len(grid["mu"]) == len(grid["sigma"]) == 3
    assert grid["metrics"]["var"][1][1] == pytest.approx(
        report["calibration_sensitivity"]["baseline"]["var"])
    assert grid["baseline"]["mu"] == grid["mu"][1]
    backtest = data["backtest"]
    metrics = report["out_of_sample"]["evaluation"]["metrics"]
    assert len(backtest["var"]) == metrics["forecast_count"]
    assert sum(backtest["violation"]) == metrics["violation_count"]


def test_optional_sections_absent_and_invalid_inputs_rejected(run):
    report, paths = run
    bare = {k: v for k, v in report.items() if k not in ("calibration_sensitivity", "out_of_sample")}
    data = build_dashboard_data(bare, paths)
    assert data["sensitivity"] is None and data["backtest"] is None
    with pytest.raises(ValueError, match="shape"):
        build_dashboard_data(report, paths[:, :10])
    shifted = paths.copy(); shifted[0] = shifted[0, 0] + 1
    with pytest.raises(ValueError, match="last price"):
        build_dashboard_data(report, shifted)
    duplicated = deepcopy(report)
    duplicated["calibration_sensitivity"]["scenarios"][1] = duplicated["calibration_sensitivity"]["scenarios"][0]
    with pytest.raises(ValueError):
        sensitivity_grid(duplicated)


def test_regeneration_detects_a_report_it_cannot_reproduce(run):
    report, _ = run
    tampered = deepcopy(report)
    tampered["results"]["terminal"]["mean"]["estimate"] += 1e-9
    with pytest.raises(ValueError, match="reproduce"):
        regenerate_paths(tampered)


def test_sensitivity_grid_collapses_identical_repeats_and_rejects_conflicts(run):
    report, _ = run
    s0, mu, sigma = report["last_price"], report["parameters"]["mu"], report["parameters"]["sigma"]
    from validation.calibration_sensitivity import calibration_sensitivity
    repeated = dict(report, calibration_sensitivity=calibration_sensitivity(
        s0, mu, sigma, 0.5, alpha=0.95, mu_delta=0.0, sigma_relative_delta=0.2, scenario_levels=3))
    grid = sensitivity_grid(repeated)
    assert grid["mu"] == [mu] and len(grid["sigma"]) == 3
    assert grid["metrics"]["var"][1][0] == pytest.approx(repeated["calibration_sensitivity"]["baseline"]["var"])
    conflicting = deepcopy(repeated)
    conflicting["calibration_sensitivity"]["scenarios"][0]["var"] += 1.0
    with pytest.raises(ValueError, match="disagree"):
        sensitivity_grid(conflicting)


def test_single_path_report_builds_with_undefined_interval():
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    snapshot = historical_snapshot(
        pd.Series(100 * np.exp(log_prices),
                  index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B")),
        ticker="ONE", period="fixture", retrieved_at="2026-03-01T12:00:00+00:00")
    report, paths = run_simulation(snapshot, SimulationConfig(horizon=0.5, steps=5, n_sims=1, seed=1))
    data = build_dashboard_data(report)
    json.dumps(data, allow_nan=False)
    assert data["summary"]["terminal_mean_confidence_interval"]["lower"] is None
