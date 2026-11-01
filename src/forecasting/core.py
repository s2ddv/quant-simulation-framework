"""Compare one-session conditional second moments on shared forecast origins."""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
from scipy.optimize import nnls
from scipy.special import ndtri

from calibration.estimation import TRADING_DAYS
from calibration.prices import validate_prices
from validation.out_of_sample import _violation_rate_interval


FEATURE_WINDOWS = (1, 5, 22)


@dataclass(frozen=True)
class ForecastConfig:
    """Sizes count labelled examples after the largest common lookback."""

    train_size: int = 126
    validation_size: int = 63
    refit_frequency: int = 22
    historical_windows: tuple[int, ...] = (5, 22, 63)
    ewma_factors: tuple[float, ...] = (0.90, 0.94, 0.97)
    alpha: float = 0.95

    def __post_init__(self):
        for name, minimum in (("train_size", 4), ("validation_size", 1),
                              ("refit_frequency", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not _real(self.alpha) or not 0 < float(self.alpha) < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        object.__setattr__(self, "alpha", float(self.alpha))
        for name in ("historical_windows", "ewma_factors"):
            grid = getattr(self, name)
            if not isinstance(grid, (tuple, list)) or not grid:
                raise ValueError(f"{name} must be a nonempty ordered grid.")
            if name == "historical_windows":
                valid = all(type(w) is int and w >= 1 for w in grid)
            else:
                valid = all(_real(f) and 0 <= float(f) < 1 for f in grid)
                if valid:
                    grid = tuple(float(f) for f in grid)
            if not valid or len(set(grid)) != len(grid):
                raise ValueError(f"Invalid or duplicate entries in {name}.")
            object.__setattr__(self, name, tuple(grid))


def _real(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def fit_nonnegative(features, targets):
    """Fit y = intercept + X @ slopes by NNLS. All four parameters are >= 0."""
    x, y = np.asarray(features), np.asarray(targets)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("Training examples must be real.")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),) or len(x) < 4
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x < 0) or np.any(y < 0)):
        raise ValueError("Require at least four finite, nonnegative 3-feature examples.")
    if np.all(y == y[0]):
        return np.array([y[0], 0.0, 0.0, 0.0])
    design = np.column_stack((np.ones(len(x)), x))
    # Positive rescaling preserves the objective and the nonnegative constraint.
    # Scaling is local to the fit. A zero column needs no scaling.
    scale = design.max(axis=0)
    scale[scale == 0] = 1.0
    y_scale = y.max()
    try:
        scaled, _ = nnls(design / scale, y / y_scale)
        coefficients = scaled * y_scale / scale
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError("Nonnegative least-squares fit failed.") from exc
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite regression coefficients.")
    return coefficients


def _rolling(squared, window, lookback):
    """Row i ends at close i+lookback. Its target return is squared[i+lookback]."""
    means = np.convolve(squared, np.full(window, 1 / window), mode="valid")
    return means[lookback - window:-1]


def _ewma(squared, factor, lookback):
    moments = np.empty_like(squared)
    moments[0] = squared[0]
    for i in range(1, len(moments)):
        moments[i] = factor * moments[i - 1] + (1 - factor) * squared[i]
    return moments[lookback - 1:-1]


def _select(candidates, targets, validation, parameter):
    scores = [{parameter: key, "mse": float(np.mean((values[validation] - targets) ** 2))}
              for key, values in candidates.items()]
    return min(scores, key=lambda row: row["mse"])[parameter], scores


def _diagnostics(variance, origin, target, alpha, z):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("A forecast variance must be finite and nonnegative.")
    log_shift = np.sqrt(variance) * z
    # Evaluate in log space if the intermediate multiplier would overflow
    # or underflow even though the quantile itself may be representable.
    if abs(log_shift) < 700:
        quantile = origin * np.exp(log_shift)
    else:
        quantile = np.exp(np.log(origin) + log_shift)
    if not np.isfinite(quantile) or quantile <= 0:
        raise ValueError("Forecast price quantile exceeds positive float64 range.")
    residual = target - quantile
    return {"daily_variance": float(variance),
            "annualized_volatility": float(np.sqrt(variance) * np.sqrt(TRADING_DAYS)),
            "price_quantile": float(quantile), "var": float(origin - quantile),
            "violation": bool(target < quantile),
            "quantile_loss": float(-alpha * residual if residual < 0 else (1-alpha) * residual)}


def _metrics(rows, model, alpha):
    predictions = [row["models"][model] for row in rows]
    errors = np.array([p["daily_variance"] - row["realized_squared_return"]
                       for row, p in zip(rows, predictions)])
    violations = sum(p["violation"] for p in predictions)
    n = len(rows)
    # A sum of large currency losses can overflow even for a finite mean.
    losses = np.array([p["quantile_loss"] for p in predictions])
    scale = losses.max()
    return {"mse": float(np.mean(errors**2)), "mae": float(np.mean(np.abs(errors))),
            "quantile_loss": float(scale * np.mean(losses / scale)) if scale else 0.0,
            "forecast_count": n, "violation_count": violations,
            "violation_rate": violations / n, "nominal_violation_rate": 1-alpha,
            "wilson_95": _violation_rate_interval(violations, n)}


def evaluate(prices, config=ForecastConfig()):
    """Evaluate fixed grids and expanding NNLS fits on dated adjusted closes."""
    if not isinstance(config, ForecastConfig):
        raise ValueError("config must be a ForecastConfig.")
    lookback = max(*FEATURE_WINDOWS, *config.historical_windows)
    minimum = lookback + config.train_size + config.validation_size + 2
    prices = validate_prices(prices, min_observations=minimum, require_dates=True)
    if prices.index.normalize().has_duplicates:
        raise ValueError("Daily prices require one adjusted close per session date.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _evaluate(prices, config, lookback)
    except FloatingPointError as exc:
        raise ValueError("Forecast computation exceeds finite float64 range.") from exc


def _evaluate(prices, config, lookback):
    closes = prices.to_numpy()
    # log1p retains small changes at high price levels. Log differences
    # handle extreme ratios without overflowing S_next / S_origin.
    previous, following = closes[:-1], closes[1:]
    returns = np.diff(np.log(closes))
    near = np.minimum(previous, following) >= 0.5 * np.maximum(previous, following)
    returns[near] = np.log1p((following[near] - previous[near]) / previous[near])
    squared = returns**2
    rolling = {w: _rolling(squared, w, lookback)
               for w in set((*FEATURE_WINDOWS, *config.historical_windows))}
    historical = {w: rolling[w] for w in config.historical_windows}
    ewma = {f: _ewma(squared, f, lookback) for f in config.ewma_factors}
    features = np.column_stack([rolling[w] for w in FEATURE_WINDOWS])
    targets = squared[lookback:]
    validation = slice(config.train_size, config.train_size + config.validation_size)
    val_targets = targets[validation]
    window, window_scores = _select(historical, val_targets, validation, "window")
    factor, factor_scores = _select(ewma, val_targets, validation, "factor")
    dates = [date.isoformat() for date in prices.index]
    ml_fits = []
    # Phi^-1(1-alpha) = -Phi^-1(alpha), stable for tiny positive alpha.
    z = -ndtri(config.alpha)

    def period(start, stop):
        return {"count": stop - start, "first_origin_date": dates[lookback + start],
                "last_origin_date": dates[lookback + stop - 1],
                "first_target_date": dates[lookback + start + 1],
                "last_target_date": dates[lookback + stop]}

    def score_period(start, stop, *, is_validation):
        rows = []
        for i in range(start, stop):
            origin = lookback + i
            if (i - start) % config.refit_frequency == 0:
                # Row i-1 ends at close origin-1 and its target is observed
                # at close origin. Row i's target will arrive tomorrow.
                beta = fit_nonnegative(features[:i], targets[:i])
                ml_fits.append({"origin_date": dates[origin],
                                "last_target_date": dates[origin],
                                "training_count": i, "coefficients": beta.tolist()})
            forecasts = {"historical": historical[window][i], "ewma": ewma[factor][i],
                         "ml": beta[0] + features[i] @ beta[1:]}
            row = {"origin_date": dates[origin], "target_date": dates[origin + 1],
                   "origin_price": float(closes[origin]), "target_price": float(closes[origin+1]),
                   "realized_return": float(returns[origin]),
                   "realized_squared_return": float(targets[i]),
                   "features": features[i].tolist(),
                   "models": {name: _diagnostics(v, closes[origin], closes[origin+1],
                                                 config.alpha, z)
                              for name, v in forecasts.items()}}
            if is_validation:
                row["candidate_variances"] = {
                    "historical": [float(values[i]) for values in historical.values()],
                    "ewma": [float(values[i]) for values in ewma.values()]}
            rows.append(row)
        return {"forecasts": rows, "metrics": {m: _metrics(rows, m, config.alpha)
                                              for m in ("historical", "ewma", "ml")},
                "warnings": (["Fewer than 30 forecasts. Coverage estimates are noisy."]
                             if len(rows) < 30 else [])}

    validation_results = score_period(validation.start, validation.stop, is_validation=True)
    validation_results["candidate_scores"] = {"historical": window_scores, "ewma": factor_scores}
    configuration = asdict(config)
    for key in ("historical_windows", "ewma_factors"):
        configuration[key] = list(configuration[key])
    test = score_period(validation.stop, len(targets), is_validation=False)
    return {
        "configuration": configuration,
        "methodology": {
            "horizon_sessions": 1, "common_lookback": lookback,
            "trading_days_per_year": TRADING_DAYS, "feature_windows": list(FEATURE_WINDOWS),
            "target": "Next squared close-to-close log return, a noisy variance proxy.",
            "assumption": "Zero conditional daily log-return mean. The second moment equals variance.",
            "historical": "Trailing mean of squared returns through origin close, without centering.",
            "ewma": "Initialize at the first squared return. Update v = factor*v + (1-factor)*r^2 through origin.",
            "ml": "Nonnegative least squares with coefficient order [intercept, 1, 5, 22]. Constant targets use their value as the intercept and zero slopes. Coefficients need not be unique for rank-deficient features.",
            "refit": "Expanding labelled sample through origin close. Refit at the start of each scoring period and every refit_frequency origins thereafter. Test evaluation is sequential.",
            "selection": "Minimum validation MSE per baseline, grid order breaks ties. Validation candidate_variances follow grid order. Selected validation metrics are retrospective. Freeze choices before test.",
            "risk": "Next log return conditional on origin information is Normal(0, v). Annual GBM-equivalent mu = sigma^2/2. Lower price quantile = S_origin * exp(sqrt(v) * Phi^-1(1-alpha)).",
            "violation": "Strict target price < price quantile. At v=0 the model is a point mass at the origin price.",
            "quantile_loss": "Price pinball loss at probability 1-alpha, averaged across origins.",
            "units": {"daily_variance": "squared one-session log return",
                      "annualized_volatility": "year^-1/2, sqrt(252 * daily_variance)",
                      "mse": "(one-session log return)^4", "mae": "(one-session log return)^2",
                      "price_quantile": "asset price currency", "var": "asset price currency",
                      "quantile_loss": "asset price currency"}},
        "periods": {"train": period(0, validation.start),
                    "validation": period(validation.start, validation.stop),
                    "test": period(validation.stop, len(targets))},
        "selected_parameters": {"historical_window": window, "ewma_factor": factor},
        "ml_fits": ml_fits, "validation": validation_results, "test": test}
