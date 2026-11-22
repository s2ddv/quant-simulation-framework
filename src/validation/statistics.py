"""Analytical terminal GBM benchmarks and nested-sample convergence."""

import math
from numbers import Real
from statistics import NormalDist

import numpy as np

QUANTILES = (0.05, 0.5, 0.95)


def terminal_theory(s0, mu, sigma, horizon):
    """GBM moments and quantiles, calculated in log space.

    Price is in asset currency, horizon in years, mu in year^-1 and sigma
    in year^-1/2. Positive moments/quantiles must remain representable as
    float64; a genuinely zero variance is reserved for sigma=0 or T=0.
    Intermediate squares and exponentials must not restrict that range.
    """
    if (not np.isfinite([s0, mu, sigma, horizon]).all()
            or s0 <= 0 or sigma < 0 or horizon < 0):
        raise ValueError("Require finite parameters, s0 > 0, sigma/horizon >= 0.")
    s0, mu, sigma, horizon = map(float, (s0, mu, sigma, horizon))
    message = "Theoretical terminal statistics exceed finite precision."

    def positive_exp(log_value):
        try:
            value = math.exp(log_value)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(value) or value == 0:
            raise ValueError(message)
        return value

    log_mean = math.log(s0) + mu * horizon
    mean = s0 if mu == 0 or horizon == 0 else positive_exp(log_mean)
    if sigma == 0 or horizon == 0:
        return {"mean": mean, "variance": 0.0,
                "quantiles": {str(p): mean for p in QUANTILES}}

    # v = sigma^2*T may underflow even when mean^2*expm1(v) is representable.
    log_v = 2 * math.log(sigma) + math.log(horizon)
    try:
        v = math.exp(log_v)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if log_v < -36:
        # log(expm1(v)) = log(v) + O(v), below float64 relative precision.
        log_expm1_v = log_v
    elif v <= 1:
        log_expm1_v = math.log(math.expm1(v))
    else:
        log_expm1_v = v + math.log1p(-math.exp(-v))
    variance = positive_exp(2 * log_mean + log_expm1_v)
    standard_deviation_log = math.exp(log_v / 2)
    quantiles = {
        str(p): positive_exp(log_mean - v / 2
                            + standard_deviation_log * NormalDist().inv_cdf(p))
        for p in QUANTILES
    }
    return {"mean": mean, "variance": variance, "quantiles": quantiles}


def _comparison(estimate, theoretical):
    error = None if estimate is None else abs(estimate - theoretical)
    return {"estimate": estimate, "theoretical": theoretical,
            "absolute_error": error,
            "relative_error": (error / abs(theoretical)
                               if error is not None and theoretical != 0 else None)}


def compare_terminal(terminal, theory):
    """Use unbiased sample variance (ddof=1) and linear empirical quantiles."""
    terminal = np.asarray(terminal, dtype=float)
    if (terminal.ndim != 1 or terminal.size == 0
            or not np.isfinite(terminal).all() or np.any(terminal <= 0)):
        raise ValueError("terminal must be a nonempty vector of finite positive prices.")
    return {
        "mean": _comparison(float(terminal.mean()), theory["mean"]),
        "variance": _comparison(float(terminal.var(ddof=1))
                                if terminal.size > 1 else None, theory["variance"]),
        "quantiles": {
            str(p): _comparison(float(np.quantile(terminal, p, method="linear")),
                                theory["quantiles"][str(p)]) for p in QUANTILES
        },
    }


def convergence_analysis(terminal, theory):
    """Compare increasing prefixes of ONE run, not separately reseeded runs.

    The reference SE is the exact RMSE of the sample mean under fixed GBM
    parameters, not an envelope or a guarantee of monotonic observed error.
    """
    terminal = np.asarray(terminal, dtype=float)
    compare_terminal(terminal, theory)  # Validate before constructing checkpoints.
    counts = np.unique(np.geomspace(min(2, terminal.size), terminal.size,
                                   num=min(12, terminal.size)).astype(int))
    counts = np.unique(np.append(counts, terminal.size))
    return [{"n_sims": int(n), **compare_terminal(terminal[:n], theory),
             "mean_standard_error": float(np.sqrt(theory["variance"] / n))}
            for n in counts]


def terminal_quantile(s0, mu, sigma, horizon, p):
    """Analytical ``p``-quantile of the GBM terminal price ``S_T``.

    ``Q_p(S_T) = exp(ln s0 + (mu - sigma^2/2)*T + sigma*sqrt(T)*Phi^-1(p))``,
    evaluated in log space with the same expression and guards as
    ``terminal_theory`` (so p in ``QUANTILES`` reproduces its values). Price in
    asset currency, horizon T in years, mu in year^-1, sigma in year^-1/2;
    ``p`` is a real probability strictly inside (0, 1). With sigma=0 or T=0
    the law is degenerate and every quantile equals the deterministic mean.
    Unlike ``terminal_theory``, the variance is not formed, so a quantile can
    be returned even when the variance itself is not representable.

    Raises:
        ValueError: On invalid parameters or ``p`` (bool rejected), or if the
            quantile overflows or underflows to zero in float64.
    """
    for name, value in (("s0", s0), ("mu", mu), ("sigma", sigma),
                        ("horizon", horizon)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                f"{name} must be a real number (bool rejected), "
                f"got {type(value).__name__}."
            )
    if (not np.isfinite([s0, mu, sigma, horizon]).all()
            or s0 <= 0 or sigma < 0 or horizon < 0):
        raise ValueError("Require finite parameters, s0 > 0, sigma/horizon >= 0.")
    if (isinstance(p, (bool, np.bool_)) or not isinstance(p, Real)
            or not 0 < p < 1):
        raise ValueError("p must be a real number strictly between 0 and 1.")
    s0, mu, sigma, horizon, p = map(float, (s0, mu, sigma, horizon, p))
    message = "Theoretical terminal quantile exceeds finite precision."

    def positive_exp(log_value):
        try:
            value = math.exp(log_value)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(value) or value == 0:
            raise ValueError(message)
        return value

    log_mean = math.log(s0) + mu * horizon
    if sigma == 0 or horizon == 0:
        return s0 if mu == 0 or horizon == 0 else positive_exp(log_mean)
    log_v = 2 * math.log(sigma) + math.log(horizon)
    try:
        v = math.exp(log_v)
    except OverflowError as exc:
        raise ValueError(message) from exc
    standard_deviation_log = math.exp(log_v / 2)
    return positive_exp(log_mean - v / 2
                        + standard_deviation_log * NormalDist().inv_cdf(p))
