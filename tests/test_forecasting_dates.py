"""Calendar splits use target sessions and retain the existing causal engine."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecasting.core import ForecastConfig, evaluate


def prices():
    returns = np.random.default_rng(123).normal(0, .01, 149)
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]),
                     index=pd.bdate_range("2018-11-01", periods=150))


def config(**changes):
    kwargs = dict(validation_start="2019-01-01", test_start="2019-03-02",
                  historical_windows=(5, 22), ewma_factors=(.9, .94),
                  refit_frequency=7)
    kwargs.update(changes)
    return ForecastConfig(**kwargs)


def test_date_boundaries_and_count_mode_produce_identical_forecasts():
    series = prices()
    result = evaluate(series, config())
    periods = result["periods"]
    assert periods["train"]["last_target_date"] == "2018-12-31T00:00:00"
    assert periods["validation"]["first_target_date"] == "2019-01-01T00:00:00"
    assert periods["validation"]["last_target_date"] == "2019-03-01T00:00:00"
    # A non-session boundary uses the next observed target, never imputation.
    assert periods["test"]["first_target_date"] == "2019-03-04T00:00:00"
    assert periods["test"]["first_origin_date"] == "2019-03-01T00:00:00"
    counted = evaluate(series, replace(config(), validation_start=None, test_start=None,
                                       train_size=periods["train"]["count"],
                                       validation_size=periods["validation"]["count"]))
    for field in ("periods", "selected_parameters", "validation", "test", "ml_fits"):
        assert counted[field] == result[field]
    assert "validation_start" not in counted["configuration"]
    assert "test_start" not in counted["configuration"]


def test_date_mode_ignores_count_sizes_and_uses_local_session_dates():
    series = prices()
    series.index = series.index.tz_localize("America/New_York")
    result = evaluate(series, config(train_size=10000, validation_size=10000))
    assert result["periods"]["test"]["first_target_date"] == "2019-03-04T00:00:00-05:00"
    assert all(row["target_date"][:10] >= "2019-03-02"
               for row in result["test"]["forecasts"])


def test_date_split_selection_and_first_forecast_do_not_use_future_targets():
    series = prices()
    result = evaluate(series, config())
    first = result["test"]["forecasts"][0]
    origin = series.index.get_loc(first["origin_date"])
    changed = series.copy()
    changed.iloc[origin + 1:] *= np.exp(np.linspace(.1, .8, len(series)-origin-1))
    other = evaluate(changed, config())
    assert result["selected_parameters"] == other["selected_parameters"]
    assert result["validation"] == other["validation"]
    for model in ("historical", "ewma", "ml"):
        assert (first["models"][model]["daily_variance"] ==
                other["test"]["forecasts"][0]["models"][model]["daily_variance"])
    assert all(fit["last_target_date"] <= fit["origin_date"] for fit in result["ml_fits"])


def test_continuous_test_refits_do_not_restart_at_calendar_boundaries():
    result = evaluate(prices(), config())
    rows = result["test"]["forecasts"]
    origins = [row["origin_date"] for row in rows]
    fits = [fit for fit in result["ml_fits"] if fit["origin_date"] >= origins[0]]
    assert [fit["origin_date"] for fit in fits] == origins[::7]
    assert len({stamp[:7] for stamp in origins}) > 1
    for row in rows:
        assert set(row["models"]) == {"historical", "ewma", "ml"}
        i = prices().index.get_loc(row["origin_date"])
        assert row["target_date"] == prices().index[i+1].isoformat()


@pytest.mark.parametrize("changes", [
    {"validation_start": None}, {"test_start": None},
    {"validation_start": "2019-1-01"}, {"test_start": "2019-02-30"},
    {"validation_start": "2019-01-01T00:00:00"}, {"test_start": 20190302},
    {"test_start": "2019-01-01"}, {"test_start": "2018-12-31"},
])
def test_invalid_date_configurations(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize("validation_start,test_start", [
    ("2018-11-02", "2019-03-02"),  # Insufficient training labels after lookback.
    ("2019-03-02", "2019-03-03"),  # No observed validation target on weekend.
    ("2019-01-01", "2020-01-01"),  # No final test targets.
])
def test_insufficient_periods_rejected(validation_start, test_start):
    with pytest.raises(ValueError, match="Date splits require"):
        evaluate(prices(), config(validation_start=validation_start, test_start=test_start))


def test_date_report_replays_from_snapshot_offline(tmp_path):
    from forecasting.report import read_experiment, run_experiment
    from simulation.report import historical_snapshot

    snapshot = historical_snapshot(prices(), ticker="SYNTHETIC", period="offline",
                                   retrieved_at="2026-01-01T00:00:00+00:00", source="synthetic")
    report = run_experiment(snapshot, config())
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert read_experiment(path) == report
