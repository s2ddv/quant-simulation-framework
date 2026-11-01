"""Deterministic, offline checks of the experimental variance forecasts."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from forecasting.core import ForecastConfig, evaluate, fit_nonnegative


def prices_from_returns(returns):
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]),
                     index=pd.date_range("2024-01-02", periods=len(returns) + 1,
                                         freq="B"))


def sample_prices():
    return prices_from_returns(np.random.default_rng(12).normal(.0002, .012, 75))


def configuration(**changes):
    options = dict(train_size=8, validation_size=5, refit_frequency=2,
                   historical_windows=(5, 22), ewma_factors=(.5, .94), alpha=.9)
    options.update(changes)
    return ForecastConfig(**options)


def test_alignment_same_dates_and_known_targets_only():
    prices = sample_prices()
    result = evaluate(prices, configuration())
    returns = np.diff(np.log(prices.to_numpy()))
    for period in ("validation", "test"):
        for row in result[period]["forecasts"]:
            origin = prices.index.get_loc(pd.Timestamp(row["origin_date"]))
            assert row["target_date"] == prices.index[origin + 1].isoformat()
            assert row["origin_price"] == prices.iloc[origin]
            assert row["target_price"] == prices.iloc[origin + 1]
            assert row["features"] == pytest.approx(
                [np.mean(returns[origin - lag:origin] ** 2) for lag in (1, 5, 22)])
            assert row["realized_squared_return"] == pytest.approx(returns[origin] ** 2)
            assert set(row["models"]) == {"historical", "ewma", "ml"}
        counts = {item["forecast_count"] for item in result[period]["metrics"].values()}
        assert counts == {len(result[period]["forecasts"])}
    for fit in result["ml_fits"]:
        assert fit["last_target_date"] <= fit["origin_date"]
        assert np.all(np.asarray(fit["coefficients"]) >= 0)
        origin = prices.index.get_loc(pd.Timestamp(fit["origin_date"]))
        training_origins = range(22, origin)
        x = [[np.mean(returns[t - lag:t] ** 2) for lag in (1, 5, 22)]
             for t in training_origins]
        y = returns[22:origin] ** 2
        assert fit["training_count"] == len(y)
        assert fit["coefficients"] == pytest.approx(fit_nonnegative(x, y), abs=1e-9)


def test_moving_mean_and_causal_ewma_manual_recursion():
    prices = sample_prices()
    result = evaluate(prices, configuration(historical_windows=(5,), ewma_factors=(.5,)))
    squared = np.diff(np.log(prices.to_numpy())) ** 2
    ewma = squared[0]
    states = [ewma]
    for value in squared[1:]:
        ewma = .5 * ewma + .5 * value
        states.append(ewma)
    for period in ("validation", "test"):
        for row in result[period]["forecasts"]:
            origin = prices.index.get_loc(pd.Timestamp(row["origin_date"]))
            assert row["models"]["historical"]["daily_variance"] == pytest.approx(
                np.mean(squared[origin - 5:origin]))
            assert row["models"]["ewma"]["daily_variance"] == pytest.approx(states[origin - 1])


def test_nonnegative_linear_fit_recovers_controlled_predictions():
    rng = np.random.default_rng(84)
    x = rng.uniform(0, .002, size=(80, 3))
    expected = np.array([.00001, .2, .4, .1])
    y = expected[0] + x @ expected[1:]
    fitted = fit_nonnegative(x, y)
    assert np.all(np.asarray(fitted) >= 0)
    assert fitted == pytest.approx(expected, rel=1e-6, abs=1e-10)
    assert fitted[0] + x @ fitted[1:] == pytest.approx(y, rel=1e-7, abs=1e-11)


def test_future_prices_cannot_change_issued_forecast_or_validation_selection():
    prices = sample_prices()
    result = evaluate(prices, configuration())
    first = result["test"]["forecasts"][0]
    origin = prices.index.get_loc(pd.Timestamp(first["origin_date"]))
    changed = prices.copy()
    changed.iloc[origin + 1:] *= np.exp(np.linspace(.1, .5, len(prices) - origin - 1))
    other = evaluate(changed, configuration())
    assert other["selected_parameters"] == result["selected_parameters"]
    assert other["validation"] == result["validation"]
    before = first["models"]
    after = other["test"]["forecasts"][0]["models"]
    for model in before:
        for field in ("daily_variance", "annualized_volatility", "price_quantile", "var"):
            assert after[model][field] == before[model][field]


def test_risk_units_pinball_and_aggregate_metrics():
    result = evaluate(sample_prices(), configuration())
    tau = .1
    for period in ("validation", "test"):
        rows = result[period]["forecasts"]
        for model in ("historical", "ewma", "ml"):
            errors, losses, violations = [], [], []
            for row in rows:
                prediction = row["models"][model]
                variance = prediction["daily_variance"]
                quantile = row["origin_price"] * np.exp(np.sqrt(variance) * norm.ppf(tau))
                assert prediction["annualized_volatility"] == pytest.approx(np.sqrt(252 * variance))
                assert prediction["price_quantile"] == pytest.approx(quantile)
                assert prediction["var"] == pytest.approx(row["origin_price"] - quantile)
                residual = row["target_price"] - quantile
                pinball = residual * (tau - (residual < 0))
                assert prediction["quantile_loss"] == pytest.approx(pinball)
                assert prediction["violation"] == (row["target_price"] < quantile)
                errors.append(variance - row["realized_squared_return"])
                losses.append(pinball)
                violations.append(prediction["violation"])
            metrics = result[period]["metrics"][model]
            assert metrics["mse"] == pytest.approx(np.mean(np.square(errors)))
            assert metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
            assert metrics["quantile_loss"] == pytest.approx(np.mean(losses))
            assert metrics["violation_count"] == sum(violations)
            assert metrics["violation_rate"] == pytest.approx(np.mean(violations))
            assert metrics["nominal_violation_rate"] == pytest.approx(tau)
            # Independent Wilson formula, with the standard normal 97.5% quantile.
            n, p = len(rows), np.mean(violations)
            z = norm.ppf(.975)
            center = (p + z*z / (2*n)) / (1 + z*z / n)
            half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
            assert metrics["wilson_95"]["lower"] == pytest.approx(max(0, center-half))
            assert metrics["wilson_95"]["upper"] == pytest.approx(min(1, center+half))
    json.dumps(result, allow_nan=False)


def test_constant_series_is_degenerate_with_strict_no_violations():
    result = evaluate(prices_from_returns(np.zeros(50)), configuration())
    for period in ("validation", "test"):
        assert result[period]["warnings"]
        for row in result[period]["forecasts"]:
            for prediction in row["models"].values():
                assert prediction["daily_variance"] == 0
                assert prediction["annualized_volatility"] == 0
                assert prediction["price_quantile"] == row["origin_price"]
                assert prediction["var"] == 0
                assert prediction["violation"] is False


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_prices_rejected(value):
    prices = sample_prices()
    prices.iloc[20] = value
    with pytest.raises(ValueError):
        evaluate(prices, configuration())


def test_insufficient_history_and_invalid_configuration():
    with pytest.raises(ValueError):
        evaluate(sample_prices().iloc[:30], configuration())
    for changes in ({"train_size": 0}, {"validation_size": 0},
                    {"refit_frequency": 0}, {"historical_windows": ()},
                    {"ewma_factors": (1.,)}, {"alpha": 1.}):
        with pytest.raises(ValueError):
            configuration(**changes)


def snapshot():
    from simulation.report import historical_snapshot
    return historical_snapshot(sample_prices(), ticker="SYNTHETIC", period="offline",
                               retrieved_at="2026-01-01T00:00:00+00:00", source="synthetic")


def test_report_roundtrip_reproduction_tamper_and_overwrite(tmp_path):
    from forecasting.report import run_experiment, read_experiment, publish_experiment
    report = run_experiment(snapshot(), configuration())
    destination = publish_experiment(report, tmp_path / "experiment.json")
    saved_path = destination / "report.json"
    assert (destination / "volatility.png").read_bytes().startswith(b"\x89PNG")
    assert read_experiment(saved_path) == report
    replay = run_experiment(report["historical_data"],
                            ForecastConfig(**report["experiment"]["configuration"]))
    assert replay == report
    with pytest.raises(FileExistsError):
        publish_experiment(report, tmp_path / "experiment.json")
    (tmp_path / "occupied.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_experiment(report, tmp_path / "occupied.json")
    assert (tmp_path / "occupied.json").read_text(encoding="utf-8") == "keep"
    report["experiment"]["test"]["forecasts"][0]["models"]["ml"]["daily_variance"] += .1
    saved_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        read_experiment(saved_path)


def test_snapshot_checksum_is_checked_before_forecasting():
    from forecasting.report import run_experiment
    data = snapshot()
    data["observations"][4]["close"] *= 1.01
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(data, configuration())


def test_failed_publication_leaves_no_complete_bundle(tmp_path, monkeypatch):
    import forecasting.report as reports
    report = reports.run_experiment(snapshot(), configuration())

    def failed_plot(*args):
        raise OSError("synthetic figure failure")

    monkeypatch.setattr(reports, "plot_experiment", failed_plot)
    with pytest.raises(OSError, match="synthetic figure failure"):
        reports.publish_experiment(report, tmp_path / "failed.json")
    assert list(tmp_path.iterdir()) == []


def test_cli_snapshot_and_replay_are_offline(tmp_path, monkeypatch, capsys):
    from forecasting.__main__ import main
    import calibration.data as provider

    def forbidden_download(*args, **kwargs):
        pytest.fail("offline experiment attempted provider download")

    monkeypatch.setattr(provider, "download_prices", forbidden_download)
    snapshot_path, config_path = tmp_path / "snapshot.json", tmp_path / "config.json"
    snapshot_path.write_text(json.dumps(snapshot()), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(configuration())), encoding="utf-8")
    main(["--snapshot", str(snapshot_path), "--config", str(config_path),
          "--report", str(tmp_path / "first.json")])
    main(["--replay", str(tmp_path / "first" / "report.json"),
          "--report", str(tmp_path / "replay.json")])
    first = json.loads((tmp_path / "first" / "report.json").read_text(encoding="utf-8"))
    replay = json.loads((tmp_path / "replay" / "report.json").read_text(encoding="utf-8"))
    assert first == replay
    assert "Final test" in capsys.readouterr().out


def test_timezone_snapshot_crossing_dst_keeps_session_dates_and_replays(tmp_path):
    from simulation.report import historical_snapshot
    from forecasting.report import run_experiment, publish_experiment, read_experiment
    prices = sample_prices()
    prices.index = pd.date_range("2024-09-02", periods=len(prices), freq="B",
                                 tz="America/New_York")
    data = historical_snapshot(prices, ticker="SYNTHETIC", period="offline",
                               retrieved_at="2026-01-01T00:00:00+00:00", source="synthetic")
    report = run_experiment(data, configuration())
    assert report["historical_data"] == data
    first = report["experiment"]["test"]["forecasts"][0]
    # 22-return lookback + 8 training rows + 5 validation rows.
    assert pd.Timestamp(first["origin_date"]).date() == prices.index[35].date()
    assert pd.Timestamp(first["target_date"]).date() == prices.index[36].date()
    destination = publish_experiment(report, tmp_path / "dst.json")
    assert read_experiment(destination / "report.json") == report


def test_single_forecast_chart_has_visible_markers(tmp_path):
    import matplotlib.pyplot as plt
    from forecasting.report import run_experiment, plot_experiment
    from simulation.report import historical_snapshot
    # Exactly 22 lookback + 8 train + 5 validation + 1 test + 1 initial close.
    prices = sample_prices().iloc[:37]
    data = historical_snapshot(prices, ticker="SYNTHETIC", period="offline",
                               retrieved_at="2026-01-01T00:00:00+00:00", source="synthetic")
    report = run_experiment(data, configuration())
    assert len(report["experiment"]["test"]["forecasts"]) == 1
    figure = plot_experiment(report, tmp_path / "single.png")
    try:
        assert len(figure.axes[0].lines) == 4
        assert all(line.get_marker() not in (None, "None", "", " ")
                   for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_savefig_failure_closes_new_figure_and_removes_staging(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from forecasting.report import run_experiment, publish_experiment
    report = run_experiment(snapshot(), configuration())
    existing = plt.figure()
    before = plt.get_fignums()

    def failed_save(*args, **kwargs):
        raise OSError("synthetic save failure")

    monkeypatch.setattr(Figure, "savefig", failed_save)
    try:
        with pytest.raises(OSError, match="synthetic save failure"):
            publish_experiment(report, tmp_path / "failed-save.json")
        assert plt.get_fignums() == before
        assert list(tmp_path.iterdir()) == []
    finally:
        plt.close(existing)
