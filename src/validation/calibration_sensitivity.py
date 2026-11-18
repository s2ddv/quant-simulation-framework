"""Deterministic GBM sensitivity to explicit historical-calibration scenarios.

This is a parameter stress grid, not an estimate of the sampling distribution
of fitted parameters. Its spreads are user assumptions, not confidence levels.
No historical data, RNG, simulation, or persistence is used in this module.
"""

import math
from numbers import Real
from statistics import NormalDist

from validation.statistics import QUANTILES


def _finite_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _positive_exp(log_value):
    try:
        value = math.exp(log_value)
    except OverflowError as exc:
        raise ValueError("sensitivity statistics exceed finite precision") from exc
    if not math.isfinite(value) or value == 0:
        raise ValueError("sensitivity statistics exceed finite precision")
    return value


def _terminal_metrics(s0, mu, sigma, horizon, alpha):
    # Work in log prices so exp(mu*T) need not itself be representable.
    log_mean = math.log(s0) + mu * horizon
    mean = s0 if mu == 0 or horizon == 0 else _positive_exp(log_mean)
    if sigma == 0 or horizon == 0:
        quantiles = {str(p): mean for p in QUANTILES}
        lower = mean
    else:
        # Scale first: sigma**2 may overflow when sigma**2*T is finite.
        scale = sigma * math.sqrt(horizon)
        location = log_mean - 0.5 * scale * scale
        quantiles = {
            str(p): _positive_exp(location + scale * NormalDist().inv_cdf(p))
            for p in QUANTILES
        }
        # Symmetry avoids rounding 1-alpha to 1 for very small alpha.
        lower = _positive_exp(location - scale * NormalDist().inv_cdf(alpha))
    return {"mean": mean, "quantiles": quantiles,
            "lower_quantile": lower, "var": s0 - lower}


def calibration_sensitivity(s0, mu, sigma, horizon, alpha=0.95,
                            mu_delta=0.02, sigma_relative_delta=0.2,
                            scenario_levels=3):
    """Return JSON-compatible analytic terminal metrics for a bounded grid.

    Parameters are fitted historical price drift ``mu`` (year^-1), volatility
    ``sigma`` (year^-1/2), initial price ``s0`` (asset currency), and ``horizon``
    (years). ``mu_delta`` is an absolute annual drift shift; e.g. 0.02 means
    two percentage points. ``sigma_relative_delta`` is a fraction in [0, 1].
    ``scenario_levels`` is an integer in {3, 5, 7, 9, 100}, excluding bool. The
    grid crosses drift shifts in [-mu_delta, mu_delta]
    with volatility multipliers in [1-sigma_relative_delta,
    1+sigma_relative_delta], including the exact unperturbed center. At 100
    levels, spacing differs slightly on either side to retain both endpoints
    and the exact baseline.
    The default 3-by-3 output preserves the legacy report schema and values.

    For each constant-parameter GBM, E[S_T] = s0*exp(mu*T) and
    Q_p = s0*exp((mu-sigma^2/2)*T + sigma*sqrt(T)*Phi^-1(p)).
    VaR_alpha = s0-Q_(1-alpha), retaining negative VaR (gains). Quantiles
    describe individual terminal prices conditional on a scenario. Ranges
    are extrema over these hypotheses, not confidence intervals for
    parameters, future prices, or a simulated mean. Spreads are not fitted
    to history and do not estimate joint parameter uncertainty or model error.

    Zero spreads, zero sigma, or zero horizon can produce duplicate scenarios;
    all scenario_levels**2 rows remain present. With sigma=0 the volatility shock
    remains zero. Finite real scalars only; s0>0, sigma/horizon/mu_delta>=0,
    0<alpha<1. Unrepresentable positive prices raise ValueError rather than
    silently returning zero or infinity. Memory is O(scenario_levels**2),
    bounded by 10,000 scalar records; no simulation matrices are allocated.
    """
    if type(scenario_levels) is not int or scenario_levels not in (3, 5, 7, 9, 100):
        raise ValueError("scenario_levels must be an integer in {3, 5, 7, 9, 100}")
    s0 = _finite_real("s0", s0)
    mu = _finite_real("mu", mu)
    sigma = _finite_real("sigma", sigma)
    horizon = _finite_real("horizon", horizon)
    alpha = _finite_real("alpha", alpha)
    mu_delta = _finite_real("mu_delta", mu_delta)
    sigma_relative_delta = _finite_real("sigma_relative_delta", sigma_relative_delta)
    if s0 <= 0 or sigma < 0 or horizon < 0 or mu_delta < 0:
        raise ValueError("require s0 > 0 and sigma, horizon, mu_delta >= 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 <= sigma_relative_delta <= 1:
        raise ValueError("sigma_relative_delta must be between 0 and 1")

    # Keep the legacy operations (including signed zero) for exact replay.
    if scenario_levels == 3:
        shifts = (-mu_delta, 0.0, mu_delta)
        multipliers = (1 - sigma_relative_delta, 1.0, 1 + sigma_relative_delta)
    elif scenario_levels == 100:
        # An even grid needs unequal counts either side of its exact baseline.
        offsets = [index / 49 for index in range(-49, 0)]
        offsets += [index / 50 for index in range(0, 51)]
        shifts = [mu_delta * offset for offset in offsets]
        multipliers = [1 + sigma_relative_delta * offset for offset in offsets]
        shifts[49], multipliers[49] = 0.0, 1.0
    else:
        half = scenario_levels // 2
        offsets = [index / half for index in range(-half, half + 1)]
        shifts = [mu_delta * offset for offset in offsets]
        multipliers = [1 + sigma_relative_delta * offset for offset in offsets]
        shifts[half], multipliers[half] = 0.0, 1.0

    scenarios = []
    for shift in shifts:
        for multiplier in multipliers:
            scenario_mu = _finite_real("scenario mu", mu + shift)
            scenario_sigma = _finite_real("scenario sigma", sigma * multiplier)
            scenarios.append({
                "mu_shift": shift, "sigma_multiplier": multiplier,
                "mu": scenario_mu, "sigma": scenario_sigma,
                **_terminal_metrics(s0, scenario_mu, scenario_sigma, horizon, alpha),
            })

    def limits(values):
        return {"min": min(values), "max": max(values)}

    configuration = {"s0": s0, "mu": mu, "sigma": sigma,
                     "horizon": horizon, "alpha": alpha,
                     "mu_delta": mu_delta,
                     "sigma_relative_delta": sigma_relative_delta}
    if scenario_levels != 3:
        configuration["scenario_levels"] = scenario_levels
    scenario_count = "nine" if scenario_levels == 3 else str(scenario_levels ** 2)
    return {
        "method": "analytic_gbm_parameter_scenario_grid",
        "assumptions": [
            "Constant-parameter normal GBM in each scenario; historical price drift is not risk-neutral drift.",
            "Drift shifts and volatility multipliers are user-selected sensitivity hypotheses, not statistically estimated uncertainty bounds.",
            f"Ranges are minima/maxima over {scenario_count} scenarios, not confidence intervals or guaranteed future price ranges.",
            "Conditional terminal quantiles describe the price distribution within each scenario.",
            "No Monte Carlo sampling is used; Monte Carlo error and the confidence interval of a simulated mean are separate quantities.",
            "The grid does not estimate parameter covariance, regime changes, or model error; zero fitted volatility stays zero under relative shocks.",
        ],
        "units": {"s0": "asset_price_currency", "mu": "year^-1",
                  "sigma": "year^-1/2", "horizon": "years",
                  "mu_delta": "year^-1", "sigma_relative_delta": "fraction",
                  "mean_quantiles_var": "asset_price_currency"},
        "configuration": configuration,
        "baseline": scenarios[(49 if scenario_levels == 100 else scenario_levels // 2)
                              * (scenario_levels + 1)],
        "scenarios": scenarios,
        "ranges": {
            "mean": limits([row["mean"] for row in scenarios]),
            "quantiles": {str(p): limits([row["quantiles"][str(p)] for row in scenarios])
                          for p in QUANTILES},
            "var": limits([row["var"] for row in scenarios]),
        },
    }
