"""Chronological checks for an unlabelled forecast origin and frozen selection."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from forecasting.core import ForecastConfig, evaluate
from forecasting.origin import forecast_at_origin, make_selection


def history(n=90):
    r = np.random.default_rng(37).normal(0, .012, n - 1)
    return pd.Series(100 * np.exp(np.r_[0., r.cumsum()]),
                     index=pd.bdate_range("2017-01-02", periods=n))


def test_two_baselines_at_last_known_price():
    prices = pd.Series(np.exp([0., .1, .3, .2]), index=pd.bdate_range("2024-02-01", periods=4))
    r2 = [.01, .04, .01]
    for model, options, expected in [
            ("historical", {"window": 2}, sum(r2[-2:]) / 2),
            ("ewma", {"factor": .5}, (.5 * r2[0] + .5 * r2[1]) * .5 + .5 * r2[2])]:
        forecast = forecast_at_origin(prices, model, **options)
        assert forecast["origin_price"] == prices.iloc[-1]
        assert forecast["daily_variance"] == pytest.approx(expected)
        assert forecast["annualized_volatility"] == pytest.approx(np.sqrt(252 * expected))


@pytest.mark.parametrize("calendar", [False, True])
def test_selection_and_origin_reproduce_sequential_experiment(calendar):
    prices = history()
    splits = ({"validation_start": prices.index[47].date().isoformat(),
               "test_start": prices.index[59].date().isoformat()} if calendar else
              {"train_size": 13, "validation_size": 12})
    config = ForecastConfig(historical_windows=(5, 33), ewma_factors=(0, .77),
                            refit_frequency=3, **splits)
    result = evaluate(prices, config)
    artifact = make_selection(prices, config, source="offline input")
    assert artifact["selected_parameters"] == result["selected_parameters"]
    assert artifact["candidate_scores"] == result["validation"]["candidate_scores"]
    test_rows = result["test"]["forecasts"]
    future_changed = prices.copy()
    cutoff = pd.Timestamp(test_rows[0]["origin_date"])
    future_changed.loc[future_changed.index > cutoff] *= 1.4
    assert make_selection(future_changed, config, source="offline input") == artifact

    # At two refits and the first intervening origin the independent API and
    # the scoring loop must use identical known inputs and the same fit date.
    for i in (0, 1, 3):
        row = test_rows[i]
        prefix = prices.loc[:row["origin_date"]]
        for model in ("ml", "historical", "ewma"):
            options = {"fit_through": test_rows[i - i % 3]["origin_date"]} if model == "ml" else {}
            forecast = forecast_at_origin(prefix, model, selection=artifact, **options)
            assert forecast["daily_variance"] == pytest.approx(row["models"][model]["daily_variance"],
                                                                rel=1e-12, abs=1e-17)
            if model == "ml":
                assert forecast["training"]["last_target_date"] == options["fit_through"]
                assert forecast["parameters"]["features"] == pytest.approx(row["features"])
                assert forecast["training"]["first_target_date"] == prices.index[34].isoformat()


def test_selection_rejects_future_foreign_or_modified_data():
    prices = history()
    config = ForecastConfig(train_size=8, validation_size=6, historical_windows=(5, 22))
    selection = make_selection(prices, config, source="saved validation run")
    with pytest.raises(ValueError, match="after the forecast origin"):
        forecast_at_origin(prices.iloc[:30], "historical", selection=selection)
    other = prices.copy()
    other.iloc[0] *= 1.001
    with pytest.raises(ValueError, match="does not reproduce"):
        forecast_at_origin(other, "ewma", selection=selection)
    altered = deepcopy(selection)
    altered["selected_parameters"]["historical_window"] = 40
    with pytest.raises(ValueError, match="checksum"):
        forecast_at_origin(prices, "historical", selection=altered)
    with pytest.raises(ValueError, match="override"):
        forecast_at_origin(prices, "historical", window=10, selection=selection)


def test_ml_minimum_flat_returns_and_invalid_cutoffs():
    prices = history(27)
    prices[:] = 17.
    forecast = forecast_at_origin(prices, "ml")
    assert forecast["training"]["training_count"] == 4
    assert forecast["parameters"]["coefficients"] == [0.] * 4
    assert forecast["daily_variance"] == 0.
    with pytest.raises(ValueError, match="four observed targets"):
        forecast_at_origin(prices.iloc[:-1], "ml")
    with pytest.raises(ValueError, match="observed close"):
        forecast_at_origin(prices, "ml", fit_through="2099-01-01")


def test_origin_rejects_invalid_dates_and_input():
    prices = history(30)
    with pytest.raises(ValueError, match="available returns"):
        forecast_at_origin(prices, "historical", window=30)
    with pytest.raises(ValueError, match="EWMA factor"):
        forecast_at_origin(prices, "ewma", factor=float("nan"))
    with pytest.raises(ValueError, match="incompatible"):
        forecast_at_origin(prices, "ml", window=6)
    duplicate_session = prices.index.to_list()
    duplicate_session[1] = duplicate_session[0] + pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="one adjusted close"):
        forecast_at_origin(prices.set_axis(duplicate_session), "ewma")
    prices.iloc[-1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        forecast_at_origin(prices, "historical", window=3)
