"""Variance reduction for Monte Carlo expectations under the normal GBM.

Antithetic variates (E2.1)
--------------------------
For ``n_sims = 2m`` the sampler draws ``Z`` with shape ``(steps, m)`` from the
instance's own ``np.random.Generator`` and uses ``[Z, -Z]``. Columns
``0..m-1`` hold the originals and columns ``m..2m-1`` their mirrors, so
column ``i`` and column ``i + m`` form pair ``i``. Terminal vectors use the
same order ``[originals..., mirrors...]``. An odd ``n_sims`` is rejected:
a lone unpaired path would carry a different weight in the pair estimator.
Antithetic sampling is defined only for the normal (exact GBM) mode; the
discrete Student t scenario is outside this scope and is rejected.

Each sample is still a draw from the exact GBM law, because ``-Z`` has the
same law as ``Z``. The two members of a pair are, however, *dependent*.
For an integrand ``f`` the estimator and its standard error use the pair as
the independent unit::

    Y_i = (f(S+_i) + f(S-_i)) / 2,        i = 1..m
    estimate = mean(Y),   SE = std(Y, ddof=1) / sqrt(m)

For the terminal price (``v = sigma**2 * T``, ``E S_T = S0*exp(mu*T)``)::

    Var(S_T)       = S0**2 * exp(2*mu*T) * expm1(v)
    Cov(S+, S-)    = S0**2 * exp(2*(mu - sigma**2/2)*T) - (E S_T)**2
                   = S0**2 * exp(2*mu*T) * expm1(-v)      (negative)
    Var(Y)         = (Var + Cov) / 2
    VRF            = Var / (2*Var(Y)) = 1 / (-expm1(-v))

The VRF compares, at equal cost ``2m`` exponentials, the plain mean of
``2m`` independent draws with the mean of ``m`` antithetic pairs. Examples
with ``T = 1``: sigma 0.2 -> 25.50, 0.25 -> 16.51, 0.5 -> 4.521, 1 -> 1.582.

When it does not help: the gain needs ``f`` to be monotone in ``Z``
(negative pair covariance). For even or non-monotone payoffs the pair can
be positively correlated and the VRF falls below 1; e.g. a straddle
``|S_T - K|`` with ``K`` at the median and sigma 0.2 gives VRF about 0.54
(limit 0.5 as the payoff becomes even in ``Z``).

Interval caveat: the i.i.d. intervals of ``risk.intervals`` (Wilson for the
loss probability, order-statistic VaR interval, bootstrap CVaR) assume
independent columns and are **not** valid on antithetic samples as they
stand. Use the pair as the unit (``antithetic_mean``) for expectations; the
antithetic sample remains a valid sample of the law for point estimates.

``"control"`` (E2.2) is an *estimator* technique applied after sampling,
not a sampling mode, so it is not a value of ``VarianceReduction``.

RNG streams: antithetic sampling consumes the instance's existing
``self.rng`` (PCG64 via ``default_rng``) and opens no new stream. It draws
``steps * m`` normals instead of ``steps * n_sims``, so its prices are not
elementwise comparable with ``"none"`` for the same seed.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal, TypedDict

import numpy as np

VarianceReduction = Literal["none", "antithetic"]
VARIANCE_REDUCTION_METHODS: tuple[str, ...] = ("none", "antithetic")


class AntitheticMean(TypedDict):
    estimate: float
    standard_error: float | None
    pairs: int


def validate_variance_reduction(value: object) -> str:
    """Return a supported sampling mode or raise ``ValueError``."""
    if not isinstance(value, str) or value not in VARIANCE_REDUCTION_METHODS:
        raise ValueError(
            "variance_reduction must be one of "
            + ", ".join(repr(name) for name in VARIANCE_REDUCTION_METHODS)
        )
    return value


def antithetic_mean(values: object) -> AntitheticMean:
    """Pair-based mean and standard error of an antithetic sample.

    ``values`` is a finite real 1-D array of length ``2m >= 2`` ordered as
    ``[originals..., mirrors...]`` (``values[i]`` pairs with
    ``values[i + m]``). Returns ``estimate = mean(Y)`` with
    ``Y_i = (values[i] + values[i + m]) / 2``, ``standard_error =
    std(Y, ddof=1) / sqrt(m)`` and ``pairs = m``. With one pair the sample
    standard deviation is undefined and ``standard_error`` is ``None``.

    Halving before adding keeps each ``Y_i`` finite for any finite inputs.
    The mean and standard deviation fall back to scaling by ``max |Y|``
    when the plain reductions would overflow.
    """
    array = np.asarray(values)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("values must be a real numeric array")
    array = array.astype(np.float64, copy=False)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size < 2 or array.size % 2:
        raise ValueError("values must have an even length of at least 2")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")

    m = array.size // 2
    with np.errstate(under="ignore"):
        pairs = 0.5 * array[:m] + 0.5 * array[m:]

    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(np.mean(pairs))
        spread = float(np.std(pairs, ddof=1)) if m > 1 else 0.0
    if not np.isfinite(estimate) or not np.isfinite(spread):
        scale = float(np.max(np.abs(pairs)))
        scaled = pairs / scale
        estimate = float(np.mean(scaled)) * scale
        spread = (float(np.std(scaled, ddof=1)) * scale) if m > 1 else 0.0
    if not np.isfinite(estimate) or not np.isfinite(spread):
        raise ValueError("antithetic mean exceeds finite precision")
    standard_error = spread / math.sqrt(m) if m > 1 else None
    return {"estimate": estimate, "standard_error": standard_error, "pairs": m}


def _nonnegative(name: str, value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _total_variance(sigma: float, horizon: float) -> float:
    # Scale before squaring: sigma**2 may overflow when sigma**2*T is finite.
    scale = sigma * math.sqrt(horizon)
    try:
        return scale * scale
    except OverflowError:  # pragma: no cover - float multiply returns inf
        return math.inf


def antithetic_terminal_vrf(sigma: float, horizon: float) -> float:
    """Theoretical VRF ``1 / (-expm1(-sigma**2*T))`` of the terminal mean.

    Ratio of the plain ``2m``-draw variance to the ``m``-pair antithetic
    variance of the GBM terminal-price mean (independent of ``s0`` and
    ``mu``). ``sigma`` in year^-1/2, ``horizon`` in years. The value tends to
    1 as ``sigma**2*T`` grows. When ``sigma*sqrt(T) == 0`` (or the variance
    is so small that the ratio overflows) both estimators have zero
    variance; the function returns ``math.inf``, the limit as
    ``sigma**2*T -> 0+``, and never ``nan``.
    """
    sigma = _nonnegative("sigma", sigma)
    horizon = _nonnegative("horizon", horizon)
    v = _total_variance(sigma, horizon)
    denominator = -math.expm1(-v)
    if denominator == 0:
        return math.inf
    try:
        return 1.0 / denominator
    except (OverflowError, ZeroDivisionError):  # pragma: no cover
        return math.inf


def antithetic_pair_covariance(s0: float, mu: float, sigma: float,
                               horizon: float) -> float:
    """Exact ``Cov(S+_T, S-_T) = S0**2*exp(2*mu*T)*expm1(-sigma**2*T)``.

    Equal to ``S0**2*exp(2*(mu - sigma**2/2)*T) - (S0*exp(mu*T))**2`` but
    evaluated in log space without cancellation. Price units squared; it is
    ``<= 0`` and zero only when ``sigma*sqrt(T) == 0``. A result that
    overflows, or underflows to zero although ``sigma**2*T > 0``, raises
    ``ValueError``.
    """
    if (isinstance(s0, bool) or not isinstance(s0, Real)
            or not math.isfinite(s0) or s0 <= 0):
        raise ValueError("s0 must be finite and positive")
    if isinstance(mu, bool) or not isinstance(mu, Real) or not math.isfinite(mu):
        raise ValueError("mu must be finite")
    sigma = _nonnegative("sigma", sigma)
    horizon = _nonnegative("horizon", horizon)
    v = _total_variance(sigma, horizon)
    if v == 0:
        return 0.0
    log_magnitude = (2.0 * math.log(s0) + 2.0 * float(mu) * horizon
                     + math.log(-math.expm1(-v)))
    if not math.isfinite(log_magnitude):
        raise ValueError("pair covariance exceeds finite precision")
    try:
        magnitude = math.exp(log_magnitude)
    except OverflowError:
        raise ValueError("pair covariance exceeds finite precision") from None
    if magnitude == 0:
        raise ValueError("pair covariance underflows to zero")
    return -magnitude
