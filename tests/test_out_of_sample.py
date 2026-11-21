"""Offline checks of chronological GBM forecasts against dated synthetic closes."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, norm

import main as app
from calibration.fundamentals import company_snapshot
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                               run_simulation, write_report)
from validation.out_of_sample import evaluate_out_of_sample
import validation.out_of_sample as oos


def dated_prices(n=70):
    daily_logs = np.resize([0.01, -0.006, 0.004, -0.002], n - 1)
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(daily_logs)]),
                     index=pd.date_range("2025-01-02", periods=n, freq="B"))


def test_frozen_calibration_and_nonoverlapping_forecasts():
    prices = dated_prices()
    result = evaluate_out_of_sample(prices, calibration_observations=40,
                                    horizon_days=7, alpha=0.9)
    changed = prices.copy()
    changed.iloc[40:] *= 3
    altered = evaluate_out_of_sample(changed, calibration_observations=40,
                                     horizon_days=7, alpha=0.9)
    assert result["calibration"] == altered["calibration"]
    rows = result["evaluation"]["rows"]
    assert len(rows) == 4
    assert result["evaluation"]["unscored_tail_observations"] == 2
    assert rows[0]["origin_date"] == prices.index[39].isoformat()
    assert rows[0]["target_date"] == prices.index[46].isoformat()
    assert rows[-1]["target_date"] == prices.index[67].isoformat()
    for current, following in zip(rows, rows[1:]):
        assert current["origin_date"] < current["target_date"]
        assert current["target_date"] == following["origin_date"]
    assert rows[-1]["origin_date"] < rows[-1]["target_date"]
    assert rows[0]["predicted_mean"] == pytest.approx(
        altered["evaluation"]["rows"][0]["predicted_mean"])
    assert rows[0]["var"] == pytest.approx(
        altered["evaluation"]["rows"][0]["var"])


def test_manual_losses_var_and_point_errors():
    # The training closes are constant, so sigma=mu=0. Four disjoint
    # one-session forecasts permit exact hand calculations without simulation.
    prices = pd.Series([100.0] * 31 + [90.0, 110.0, 100.0, 80.0],
                       index=pd.date_range("2025-01-02", periods=35, freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=31,
                                    horizon_days=1, alpha=0.9)
    rows = result["evaluation"]["rows"]
    assert result["calibration"]["sigma"] == 0
    assert result["calibration"]["mu"] == 0
    expected_origins = [100.0, 90.0, 110.0, 100.0]
    expected_targets = [90.0, 110.0, 100.0, 80.0]
    expected_errors = [-10.0, 20.0, -10.0, -20.0]
    for row, origin, target, error in zip(rows, expected_origins,
                                           expected_targets, expected_errors):
        assert row["origin_price"] == origin
        assert row["observed_price"] == target
        assert row["predicted_mean"] == origin
        assert row["predicted_median"] == origin
        assert row["lower_quantile"] == origin
        assert row["var"] == 0
        assert row["realized_loss"] == origin - target
        assert row["error"] == error
        assert row["var_violation"] is (error < 0)
    evaluation = result["evaluation"]["metrics"]
    assert evaluation["forecast_count"] == 4
    assert evaluation["violation_count"] == 3
    assert evaluation["observed_violation_rate"] == pytest.approx(0.75)
    assert evaluation["nominal_violation_rate"] == pytest.approx(0.1)
    assert evaluation["mean_bias"] == pytest.approx(-5.0)
    assert evaluation["mae"] == pytest.approx(15.0)
    assert evaluation["rmse"] == pytest.approx(np.sqrt(250.0))
    assert evaluation["mae_percent"] == pytest.approx(100 * np.mean(
        [10 / 100, 20 / 90, 10 / 110, 20 / 100]))
    json.dumps(result, allow_nan=False)


def test_nonzero_volatility_quantile_is_independent_analytic_reference():
    prices = dated_prices(50)
    result = evaluate_out_of_sample(prices, calibration_observations=40,
                                    horizon_days=5, alpha=0.95)
    calibration = result["calibration"]
    row = result["evaluation"]["rows"][0]
    training_logs = np.diff(np.log(prices.iloc[:40].to_numpy()))
    annual_sigma = np.std(training_logs, ddof=1) * np.sqrt(252)
    annual_mu = 252 * np.mean(training_logs) + annual_sigma**2 / 2
    t = 5 / 252
    s0 = float(prices.iloc[39])
    expected_mean = s0 * np.exp(annual_mu * t)
    expected_median = s0 * np.exp((annual_mu - annual_sigma**2 / 2) * t)
    expected_lower = s0 * np.exp((annual_mu - annual_sigma**2 / 2) * t
                                 + annual_sigma * np.sqrt(t) * norm.ppf(0.05))
    assert calibration["sigma"] == pytest.approx(annual_sigma)
    assert calibration["mu"] == pytest.approx(annual_mu)
    assert row["predicted_mean"] == pytest.approx(expected_mean)
    assert row["predicted_median"] == pytest.approx(expected_median)
    assert row["lower_quantile"] == pytest.approx(expected_lower)
    assert row["var"] == pytest.approx(s0 - expected_lower)
    assert row["var_violation"] is (row["realized_loss"] > row["var"])


@pytest.mark.parametrize("change", [
    lambda p: p.set_axis(pd.RangeIndex(len(p))),
    lambda p: p.set_axis(p.index[::-1]),
    lambda p: p.set_axis(pd.DatetimeIndex([p.index[0]] * len(p))),
    lambda p: p.mask(np.arange(len(p)) == 2, np.nan),
    lambda p: p.mask(np.arange(len(p)) == 2, 0),
    lambda p: p.mask(np.arange(len(p)) == 2, np.inf),
])
def test_invalid_history_rejected(change):
    with pytest.raises(ValueError):
        evaluate_out_of_sample(change(dated_prices()), calibration_observations=40,
                               horizon_days=7, alpha=0.9)


@pytest.mark.parametrize("calibration,horizon,alpha,n", [
    (2, 1, 0.9, 70), (31, 1, 0.9, 31), (40, 31, 0.9, 70),
    (40, 0, 0.9, 70), (40, 1.5, 0.9, 70), (True, 1, 0.9, 70),
    (40, 1, 0, 70), (40, 1, 1, 70), (40, 1, np.nan, 70),
])
def test_invalid_parameters_or_insufficient_history(calibration, horizon, alpha, n):
    with pytest.raises(ValueError):
        evaluate_out_of_sample(dated_prices(n), calibration_observations=calibration,
                               horizon_days=horizon, alpha=alpha)


def test_no_violations_and_zero_volatility_are_json_finite():
    prices = pd.Series([100.0] * 35,
                       index=pd.date_range("2025-01-02", periods=35, freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=31,
                                    horizon_days=1, alpha=0.95)
    assert result["evaluation"]["metrics"]["violation_count"] == 0
    assert result["evaluation"]["metrics"]["observed_violation_rate"] == 0
    assert all(row["var_violation"] is False for row in result["evaluation"]["rows"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("violations,forecasts", [(0, 1), (1, 1), (0, 4), (3, 4),
                                                   (4, 4), (5, 100), (50, 100)])
def test_violation_rate_interval_matches_independent_wilson_reference(violations, forecasts):
    # Flat training makes VaR zero. Each following decrease is a strict
    # violation and each unchanged close is a nonviolation, giving exact k/n.
    closes = [100.0] * 31
    for i in range(forecasts):
        closes.append(closes[-1] - 0.1 if i < violations else closes[-1])
    prices = pd.Series(closes, index=pd.date_range("2025-01-02", periods=len(closes), freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=31, alpha=0.8)
    evaluation = result["evaluation"]
    interval = evaluation["violation_rate_interval"]
    reference = binomtest(violations, forecasts).proportion_ci(confidence_level=0.95, method="wilson")
    assert evaluation["metrics"]["forecast_count"] == forecasts
    assert evaluation["metrics"]["violation_count"] == violations
    assert interval["method"] == "wilson_score"
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] == pytest.approx(reference.low, abs=1e-14)
    assert interval["upper"] == pytest.approx(reference.high, abs=1e-14)
    assert 0 <= interval["lower"] <= violations / forecasts <= interval["upper"] <= 1
    assert interval["upper"] > interval["lower"]  # even k=0, k=n and n=1
    assert "Independent Bernoulli" in interval["assumptions"]
    assert "Small samples" in interval["limitations"]
    assert "GBM parameters" in interval["limitations"]
    assert evaluation["metrics"]["nominal_violation_rate"] == pytest.approx(0.2)
    json.dumps(result, allow_nan=False)


def test_tiny_valid_alpha_keeps_lower_tail_quantile_finite():
    result = evaluate_out_of_sample(dated_prices(50),
                                    calibration_observations=40,
                                    horizon_days=5, alpha=1e-20)
    row = result["evaluation"]["rows"][0]
    assert np.isfinite(row["lower_quantile"])
    assert row["lower_quantile"] > 0
    assert row["lower_quantile"] > row["predicted_median"]
    json.dumps(result, allow_nan=False)


def test_saved_diagnostic_detects_changed_snapshot_or_metrics(tmp_path):
    prices = dated_prices()
    snapshot = historical_snapshot(prices, ticker="SYN", period="fixture",
                                   retrieved_at="2025-05-01T00:00:00Z", source="synthetic")
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10),
                               {"calibration_observations": 40,
                                "horizon_days": 7, "alpha": 0.9})
    path = tmp_path / "valid.json"
    write_report(report, path)
    assert read_report(path) == report
    changed = deepcopy(report)
    changed["out_of_sample"]["evaluation"]["metrics"]["violation_count"] += 1
    write_report(changed, tmp_path / "changed-metrics.json")
    with pytest.raises(ValueError):
        read_report(tmp_path / "changed-metrics.json")
    changed = deepcopy(report)
    changed["historical_data"]["observations"][0]["close"] += 1
    write_report(changed, tmp_path / "changed-snapshot.json")
    with pytest.raises(ValueError):
        read_report(tmp_path / "changed-snapshot.json")


def test_cli_opt_in_replay_is_offline_and_legacy_stays_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prices = dated_prices()
    monkeypatch.setattr(app, "download_prices", lambda *a, **k: prices)
    monkeypatch.setattr(app, "download_company", lambda ticker: company_snapshot(
        ticker, {"longName": "Synthetic", "currency": "USD"},
        retrieved_at="2025-05-01T00:00:00Z"))
    first = tmp_path / "first.json"
    app.main(["--ticker", "SYN", "--period", "fixture", "--out-of-sample",
              "--calibration-observations", "40", "--evaluation-horizon-days", "7",
              "--evaluation-confidence", "0.9", "--steps", "2", "--n-sims", "10",
              "--report", str(first), "--no-plot"])
    first_report = json.loads((tmp_path / "first" / "report.json").read_text())
    assert first_report["out_of_sample"]["evaluation"]["metrics"]["forecast_count"] == 4
    assert first_report["out_of_sample"]["historical_snapshot"]["sha256"] == (
        first_report["historical_data"]["sha256"])
    monkeypatch.setattr(app, "download_prices", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(app, "download_company", lambda *a, **k: pytest.fail("network"))
    second = tmp_path / "second.json"
    app.main(["--replay", str(tmp_path / "first" / "report.json"),
              "--report", str(second), "--no-plot"])
    second_report = json.loads((tmp_path / "second" / "report.json").read_text())
    assert second_report["out_of_sample"] == first_report["out_of_sample"]
    assert second_report == first_report
    snapshot = historical_snapshot(prices, ticker="SYN", period="fixture",
                                   retrieved_at="2025-05-01T00:00:00Z", source="synthetic")
    baseline, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10))
    assert "out_of_sample" not in baseline
    json.dumps(second_report, allow_nan=False)


def changing_regime_prices():
    logs = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.04,
                     0.05, -0.03, 0.04, 0.01, -0.02, 0.03, -0.01])
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(logs)]),
                     index=pd.date_range("2025-01-02", periods=14, freq="B"))


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_refit_windows_chronology_and_independent_formula(mode, monkeypatch):
    prices = changing_regime_prices()
    fitted_windows = []
    original_estimator = oos.estimate_gbm_params

    def record_fit(training):
        fitted_windows.append(training.copy())
        return original_estimator(training)

    monkeypatch.setattr(oos, "estimate_gbm_params", record_fit)
    result = evaluate_out_of_sample(
        prices, calibration_observations=4, horizon_days=2, alpha=0.9,
        calibration_mode=mode, recalibration_frequency=2)
    rows = result["evaluation"]["rows"]
    assert len(rows) == 5
    assert len(fitted_windows) == 3  # reuse fit at forecast indices 1 and 3
    assert result["evaluation"]["unscored_tail_observations"] == 0
    assert result["protocol"]["recalibration_unit"] == "forecast_origins"
    assert result["calibration"] == rows[0]["calibration"]
    for i, row in enumerate(rows):
        origin = 3 + 2 * i
        refit_origin = 3 + 4 * (i // 2)
        first = refit_origin - 3 if mode == "rolling" else 0
        calibration = row["calibration"]
        assert row["origin_date"] == prices.index[origin].isoformat()
        assert row["target_date"] == prices.index[origin + 2].isoformat()
        assert calibration["first_date"] == prices.index[first].isoformat()
        assert calibration["last_date"] == prices.index[refit_origin].isoformat()
        assert calibration["last_date"] <= row["origin_date"] < row["target_date"]
        assert row["calibration_reused"] is (i % 2 != 0)
        sample = prices.iloc[first:refit_origin + 1].to_numpy()
        # Direct scalar formulas independently reconstruct the fitted sample
        # variance, rather than calling the production calibration functions.
        returns = np.log(sample[1:] / sample[:-1])
        mean_log = sum(returns) / len(returns)
        variance = sum((value - mean_log)**2 for value in returns) / (len(returns) - 1)
        mu = 252 * (mean_log + variance / 2)
        sigma = np.sqrt(252 * variance)
        start = prices.iloc[origin]
        t = 2 / 252
        lower = start * np.exp((mu - sigma**2 / 2) * t
                               + sigma * np.sqrt(t) * norm.ppf(0.1))
        assert calibration["price_count"] == len(sample)
        assert calibration["return_count"] == len(sample) - 1
        assert calibration["mu"] == pytest.approx(mu)
        assert calibration["sigma"] == pytest.approx(sigma)
        assert row["predicted_mean"] == pytest.approx(start * np.exp(mu * t))
        assert row["predicted_median"] == pytest.approx(start * np.exp(2 * mean_log))
        assert row["lower_quantile"] == pytest.approx(lower)
        assert row["var"] == pytest.approx(start - lower)
        assert row["realized_loss"] == start - prices.iloc[origin + 2]
        assert row["var_violation"] is (row["realized_loss"] > row["var"])
        if i:
            assert rows[i - 1]["target_date"] == row["origin_date"]
    errors = [row["observed_price"] - row["predicted_mean"] for row in rows]
    metrics = result["evaluation"]["metrics"]
    assert metrics["forecast_count"] == 5
    assert metrics["mean_bias"] == pytest.approx(sum(errors) / 5)
    assert metrics["mae"] == pytest.approx(sum(abs(error) for error in errors) / 5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(sum(error**2 for error in errors) / 5))
    assert metrics["violation_count"] == sum(row["var_violation"] for row in rows)
    assert metrics["observed_violation_rate"] == metrics["violation_count"] / 5
    assert any("share observations" in warning for warning in result["evaluation"]["warnings"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode,frequency", [
    ("fixed", 1), ("rolling", 1), ("rolling", 2),
    ("expanding", 1), ("expanding", 2),
])
def test_future_prices_cannot_change_any_available_forecast(mode, frequency):
    prices = changing_regime_prices()
    options = dict(calibration_observations=4, horizon_days=2, alpha=0.9,
                   calibration_mode=mode, recalibration_frequency=frequency)
    baseline = evaluate_out_of_sample(prices, **options)
    prediction_fields = ["origin_date", "origin_price", "target_date",
                         "predicted_mean", "predicted_median", "lower_quantile", "var"]
    if mode != "fixed":
        prediction_fields += ["calibration", "calibration_reused"]
    for index, _row in enumerate(baseline["evaluation"]["rows"]):
        origin = 3 + 2 * index
        changed = prices.copy()
        changed.iloc[origin + 1:] *= np.linspace(0.7, 1.5, len(prices) - origin - 1)
        altered = evaluate_out_of_sample(changed, **options)
        for earlier in range(index + 1):
            expected = baseline["evaluation"]["rows"][earlier]
            actual = altered["evaluation"]["rows"][earlier]
            assert {key: expected[key] for key in prediction_fields} == {
                key: actual[key] for key in prediction_fields}
        assert baseline["calibration"] == altered["calibration"]


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_incomplete_horizon_and_long_refit_frequency(mode):
    result = evaluate_out_of_sample(
        changing_regime_prices().iloc[:-1], calibration_observations=4,
        horizon_days=2, calibration_mode=mode, recalibration_frequency=100)
    rows = result["evaluation"]["rows"]
    assert len(rows) == 4
    assert result["evaluation"]["unscored_tail_observations"] == 1
    assert all(row["calibration"] == result["calibration"] for row in rows)
    assert [row["calibration_reused"] for row in rows] == [False, True, True, True]


@pytest.mark.parametrize("mode,frequency", [
    ("invalid", 1), (None, 1), ([], 1), ("rolling", 0),
    ("expanding", -1), ("rolling", 1.5), ("rolling", True),
    ("expanding", None), ("fixed", 2),
])
def test_invalid_calibration_protocol(mode, frequency):
    with pytest.raises(ValueError):
        evaluate_out_of_sample(changing_regime_prices(), calibration_observations=4,
                               calibration_mode=mode, recalibration_frequency=frequency)


def test_fixed_default_retains_legacy_shape():
    prices = changing_regime_prices()
    default = evaluate_out_of_sample(prices, calibration_observations=4)
    explicit = evaluate_out_of_sample(prices, calibration_observations=4,
                                      calibration_mode="fixed", recalibration_frequency=1)
    assert default == explicit
    assert default["method"] == "fixed_initial_calibration_nonoverlapping_historical_forecasts"
    assert "calibration_mode" not in default["protocol"]
    assert all("calibration" not in row for row in default["evaluation"]["rows"])


def test_zero_volatility_warning_accounts_for_all_refits():
    prices = pd.Series([100., 100., 100., 101., 99., 102.],
                       index=pd.date_range("2025-01-02", periods=6, freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=3,
                                    calibration_mode="rolling")
    rows = result["evaluation"]["rows"]
    assert rows[0]["calibration"]["sigma"] == 0
    assert rows[-1]["calibration"]["sigma"] > 0
    assert any("Zero volatility" in warning for warning in result["evaluation"]["warnings"])
