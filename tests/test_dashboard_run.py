"""New dashboard simulations: memory budget and snapshot recording (offline)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from simulation.dashboard_run import (check_path_budget, path_memory_bytes,
                                      run_new_simulation)
from simulation.report import SimulationConfig, prices_from_snapshot, run_simulation

NOW = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)


def fake_prices(ticker, period):
    log_prices = np.cumsum(np.r_[0, np.tile([0.012, -0.008, 0.004, -0.002], 10)])
    return pd.Series(100 * np.exp(log_prices),
                     index=pd.date_range("2026-01-05", periods=log_prices.size, freq="B"))


def test_run_records_snapshot_and_matches_the_cli_pipeline():
    calls = []

    def download(ticker, period):
        calls.append((ticker, period))
        return fake_prices(ticker, period)

    config = SimulationConfig(horizon=0.5, steps=10, n_sims=200, seed=1)
    report, paths = run_new_simulation(" petr4 ", "3y", config, download=download, now=NOW)
    assert calls == [("PETR4.SA", "3y")]
    snapshot = report["historical_data"]
    assert snapshot["ticker"] == "PETR4.SA" and snapshot["requested_period"] == "3y"
    assert snapshot["retrieved_at"] == NOW.isoformat()
    pd.testing.assert_series_equal(prices_from_snapshot(snapshot), fake_prices(None, None),
                                   check_freq=False, check_names=False)
    expected, expected_paths = run_simulation(snapshot, config)
    assert report == expected
    np.testing.assert_array_equal(paths, expected_paths)


def test_budget_is_checked_before_any_download():
    config = SimulationConfig(steps=99, n_sims=1000)
    assert path_memory_bytes(99, 1000) == 800_000 == check_path_budget(config, 800_000)

    def download(ticker, period):
        raise AssertionError("must not download")

    with pytest.raises(ValueError, match="limit"):
        run_new_simulation("AAPL", "3y", config, max_path_bytes=799_999, download=download)
    with pytest.raises(ValueError):
        run_new_simulation("two tickers", "3y", config, download=download)
    with pytest.raises(ValueError, match="period"):
        run_new_simulation("AAPL", " ", config, download=download)


def test_optional_sections_are_passed_through():
    config = SimulationConfig(horizon=0.5, steps=5, n_sims=50, seed=2)
    report, _ = run_new_simulation(
        "AAPL", "1y", config, download=fake_prices, now=NOW,
        out_of_sample={"calibration_observations": 10, "horizon_days": 1, "alpha": 0.95},
        sensitivity={"mu_delta": 0.01, "sigma_relative_delta": 0.1, "scenario_levels": 3})
    assert report["out_of_sample"]["protocol"]["calibration_observations"] == 10
    assert len(report["calibration_sensitivity"]["scenarios"]) == 9
