"""One-session forecasts conditional on prices available at the origin close."""

from dataclasses import asdict
from numbers import Real

import numpy as np
import pandas as pd

from calibration.estimation import TRADING_DAYS
from calibration.prices import validate_prices
from forecasting.core import (FEATURE_WINDOWS, ForecastConfig, _ewma, _log_returns,
                              _predict_nnls, _rolling, _select, fit_nonnegative)
from simulation.report import _digest


def _prices(prices, minimum):
    prices = validate_prices(prices, require_dates=True, min_observations=minimum)
    if prices.index.normalize().has_duplicates:
        raise ValueError("Daily prices require one adjusted close per session date.")
    return prices


def make_selection(prices, config, *, source):
    """Freeze baseline hyperparameters using only the specified validation targets.

    Supply the full training and validation history, with the same first close
    as the study. For calendar splits, supply at least one close on/after
    test_start to establish the end of validation. Only dates are inspected
    to delimit the prefix. Final test prices and labels do not enter the artifact.
    """
    if not isinstance(config, ForecastConfig) or not isinstance(source, str) or not source.strip():
        raise ValueError("A ForecastConfig and nonempty selection source are required.")
    lookback = max(*FEATURE_WINDOWS, *config.historical_windows)
    prices = _prices(prices, lookback + 6)
    if config.test_start is None:
        train_count = config.train_size
        stop = lookback + train_count + config.validation_size + 1
        if stop > len(prices):
            raise ValueError("Insufficient closes for complete validation.")
    else:
        dates = np.array([d.date().isoformat() for d in prices.index])
        start, stop = np.searchsorted(dates, [config.validation_start, config.test_start])
        train_count = int(start) - lookback - 1
        if stop == len(prices) or train_count < 4 or stop <= start:
            raise ValueError("Calendar selection requires train, validation and a test session date.")
    prefix = prices.iloc[:stop]
    return _selection_from_prefix(prefix, config, source, lookback, train_count)


def _selection_from_prefix(prefix, config, source, lookback, train_count):
    squared = _log_returns(prefix.to_numpy()) ** 2
    targets = squared[lookback + train_count:]
    validation = slice(train_count, train_count + len(targets))
    historical = {w: _rolling(squared, w, lookback) for w in config.historical_windows}
    ewma = {f: _ewma(squared, f, lookback) for f in config.ewma_factors}
    window, window_scores = _select(historical, targets, validation, "window")
    factor, factor_scores = _select(ewma, targets, validation, "factor")
    configuration = asdict(config)
    for field in ("historical_windows", "ewma_factors"):
        configuration[field] = list(configuration[field])
    artifact = {
        "schema": "validation_selection_v1", "source": source, "configuration": configuration,
        "method": "Minimum validation MSE per baseline, grid order breaks ties.",
        "first_close": prefix.index[0].isoformat(),
        "first_validation_target": prefix.index[lookback + train_count + 1].isoformat(),
        "last_validation_target": prefix.index[-1].isoformat(),
        "validation_count": len(targets),
        "prefix_sha256": _digest([{"date": d.isoformat(), "close": float(v)}
                                  for d, v in prefix.items()]),
        "selected_parameters": {"historical_window": window, "ewma_factor": factor},
        "candidate_scores": {"historical": window_scores, "ewma": factor_scores},
    }
    return {**artifact, "sha256": _digest(artifact)}


def _check_selection(prices, selection):
    if not isinstance(selection, dict) or selection.get("schema") != "validation_selection_v1":
        raise ValueError("Unsupported selection artifact.")
    payload = {k: v for k, v in selection.items() if k != "sha256"}
    if _digest(payload) != selection.get("sha256"):
        raise ValueError("Selection artifact checksum mismatch.")
    try:
        config = ForecastConfig(**selection["configuration"])
        cutoff = pd.Timestamp(selection["last_validation_target"])
        if cutoff > prices.index[-1]:
            raise ValueError("Selection uses validation targets after the forecast origin.")
        if cutoff not in prices.index:
            raise ValueError("Selection cutoff must be an observed origin-history session.")
        # Recompute from the known prices to bind the selection to this series.
        # For a calendar selection the last pre-test close can itself be the
        # first test forecast origin. Its next close is not needed to verify it.
        prefix = prices.loc[:cutoff]
        lookback = max(*FEATURE_WINDOWS, *config.historical_windows)
        if config.test_start is None:
            train_count = config.train_size
            if len(prefix) != lookback + train_count + config.validation_size + 1:
                raise ValueError("Selection does not match configured validation period.")
        else:
            dates = [d.date().isoformat() for d in prices.index]
            start, stop = np.searchsorted(dates, [config.validation_start, config.test_start])
            train_count = int(start) - lookback - 1
            if train_count < 4 or start >= stop or stop != len(prefix):
                raise ValueError("Selection does not match calendar validation period.")
        expected = _selection_from_prefix(prefix, config, selection["source"], lookback, train_count)
    except (KeyError, TypeError, OverflowError, IndexError) as exc:
        raise ValueError("Invalid selection artifact.") from exc
    if expected != selection:
        raise ValueError("Selection does not reproduce from the origin price history.")
    return config


def forecast_at_origin(prices, model, *, window=None, factor=None, lookback=None,
                       selection=None, fit_through=None):
    """Forecast the variance of the next zero-mean daily log return from known closes.

    Report annualized volatility in year^-1/2. Historical and EWMA defaults
    are 22 and .94. NNLS uses 1/5/22-return features and expands through the
    last close, or through an explicit earlier observed refit close. Its first
    training feature uses the close at index lookback (default 22).
    """
    if model not in ("historical", "ewma", "ml"):
        raise ValueError("Model must be historical, ewma or ml.")
    if ((model != "historical" and window is not None)
            or (model != "ewma" and factor is not None)
            or (model != "ml" and (lookback is not None or fit_through is not None))):
        raise ValueError("Forecast option is incompatible with the selected model.")
    prices = _prices(prices, 2)
    if selection is not None:
        if any(p is not None for p in (window, factor, lookback)):
            raise ValueError("Cannot override frozen selection parameters.")
        config = _check_selection(prices, selection)
        window = selection["selected_parameters"]["historical_window"]
        factor = selection["selected_parameters"]["ewma_factor"]
        lookback = max(*FEATURE_WINDOWS, *config.historical_windows)
    squared = _log_returns(prices.to_numpy()) ** 2
    if model == "historical":
        window = 22 if window is None else window
        if type(window) is not int or not 1 <= window <= len(squared):
            raise ValueError("Historical window must be positive and fit the available returns.")
        variance = _rolling(squared, window, window)[-1]
        parameters = {"window": window}
        training = {"policy": "Trailing window through origin.", "return_count": window,
                    "first_close": prices.index[-window - 1].isoformat()}
    elif model == "ewma":
        factor = .94 if factor is None else factor
        if isinstance(factor, bool) or not isinstance(factor, Real) or not 0 <= factor < 1:
            raise ValueError("EWMA factor must lie in [0, 1).")
        variance = _ewma(squared, factor, 1)[-1]
        parameters = {"factor": float(factor)}
        training = {"policy": "Initialize at first squared return. Update through origin.",
                    "return_count": len(squared), "first_close": prices.index[0].isoformat()}
    else:
        lookback = max(FEATURE_WINDOWS) if lookback is None else lookback
        if type(lookback) is not int or not max(FEATURE_WINDOWS) <= lookback <= len(squared) - 4:
            raise ValueError("NNLS requires lookback >= 22 and at least four observed targets.")
        stop = len(squared)
        if fit_through is not None:
            try:
                stop = prices.index.get_loc(pd.Timestamp(fit_through))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("fit_through must identify an observed close at or before origin.") from exc
        features = np.column_stack([_rolling(squared, w, lookback) for w in FEATURE_WINDOWS])
        n_labels = stop - lookback
        if n_labels < 4:
            raise ValueError("NNLS requires four observed targets at the refit close.")
        coefficients = fit_nonnegative(features[:n_labels], squared[lookback:stop])
        variance = _predict_nnls(features[-1], coefficients)
        parameters = {"lookback": lookback, "feature_windows": list(FEATURE_WINDOWS),
                      "coefficients": coefficients.tolist(), "features": features[-1].tolist()}
        training = {"policy": ("Expanding NNLS refit through origin." if fit_through is None else
                               "Expanding NNLS fit through specified close, then hold coefficients."),
                    "training_count": n_labels, "first_origin_date": prices.index[lookback].isoformat(),
                    "first_target_date": prices.index[lookback + 1].isoformat(),
                    "last_target_date": prices.index[stop].isoformat()}
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Daily variance must be finite and nonnegative.")
    return {"model": model, "origin_date": prices.index[-1].isoformat(),
            "origin_price": float(prices.iloc[-1]), "daily_variance": float(variance),
            "annualized_volatility": float(np.sqrt(variance) * np.sqrt(TRADING_DAYS)),
            "parameters": parameters, "training": training,
            "selection": selection if selection is not None else {"method": "Fixed ex ante."}}
