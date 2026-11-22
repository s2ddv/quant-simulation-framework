"""Monte Carlo confidence intervals for risk estimators.

Scope and interpretation
------------------------
The intervals here quantify the **sampling error of the Monte Carlo
estimator**: with the model and its parameters held fixed (``s0``, ``mu``,
``sigma``, horizon, distribution), how far a statistic computed from ``n``
simulated terminal prices may lie from its exact value under that model.
They are *not*:

* percentiles of the terminal price or loss distribution (VaR itself is such
  a percentile; an interval around it describes estimator error, not spread
  of outcomes);
* parameter (calibration) uncertainty, which ``validation.calibration_sensitivity``
  addresses by re-estimating ``mu`` and ``sigma``;
* model error (whether GBM describes the asset at all).

Conventions (identical to ``risk.statistics.risk_summary``)
-----------------------------------------------------------
* Prices are in the asset's currency; ``terminal`` holds simulated prices
  ``S_T`` as a 1-D sample of independent draws, ``s0`` is the initial price.
* Loss is ``L = s0 - S_T`` (price units; negative means a gain).
* A loss event is the strict inequality ``S_T < s0``; ties are not losses.

Contents
--------
* ``prob_loss_interval``: binomial-proportion interval for ``P(S_T < s0)``,
  reusing ``validation.tail_probability`` (Wilson score or Clopper-Pearson).
* ``var_interval``: distribution-free order-statistic interval for the loss
  quantile (VaR); the point estimate is the one of ``risk_summary``.

All intervals assume i.i.d. simulated paths. Variance-reduction schemes that
correlate draws (antithetic pairs, stratification, QMC) invalidate them.

The private ``_validate_*`` helpers are shared input checks intended for the
VaR and CVaR interval functions of this module as well.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from scipy import stats

from risk.statistics import _linear_percentile
from validation.tail_probability import clopper_pearson_interval, wilson_interval

_PROPORTION_METHODS = {
    "wilson": wilson_interval,
    "clopper_pearson": clopper_pearson_interval,
}


def _validate_real(value: Any, name: str) -> float:
    """Return ``value`` as a finite float; reject bool and non-real types."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}.")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(
            f"{name} must be representable as a finite float, got a value "
            "too large to convert."
        ) from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}.")
    return result


def _validate_level(level: Any, name: str = "level") -> float:
    """Return a probability level strictly inside (0, 1)."""
    result = _validate_real(level, name)
    if not 0 < result < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {result}.")
    return result


def _validate_terminal(terminal: Any) -> np.ndarray:
    """Return a finite, non-empty 1-D float64 view/copy of ``terminal``.

    Positivity is not required, matching ``risk_summary``. Boolean, complex
    and non-numeric inputs are rejected rather than silently cast.
    """
    array = np.asarray(terminal)
    if array.dtype.kind not in "iuf":
        raise ValueError(
            f"terminal must contain real numbers, got dtype {array.dtype}."
        )
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"terminal must be a non-empty 1-D array, got shape {array.shape}."
        )
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("terminal must contain only finite prices.")
    return array


def prob_loss_interval(terminal: Any, s0: Any, level: float = 0.95,
                       method: str = "wilson") -> dict[str, Any]:
    """Confidence interval for the probability of loss ``p = P(S_T < s0)``.

    The estimator is ``p_hat = k / n`` with ``k = #{S_T < s0}`` and
    ``n = terminal.size``; it equals ``risk_summary(paths)["prob_loss"]`` when
    ``terminal = paths[-1]`` and ``s0 = paths[0, 0]``. Since ``k ~ Binom(n, p)``
    for i.i.d. paths, a binomial-proportion interval for ``p`` is used:

    * ``"wilson"``: Wilson score interval (approximate; coverage close to
      ``level``, though it oscillates for small ``n * p``; Brown, Cai and
      DasGupta, 2001);
    * ``"clopper_pearson"``: exact beta-quantile interval (coverage
      ``>= level``, conservative).

    Both return ``[0, u]`` when ``k == 0`` and ``[l, 1]`` when ``k == n``.

    Args:
        terminal: Simulated terminal prices, finite 1-D array of size ``n >= 1``.
        s0: Initial price (finite real, not bool), in the asset's currency.
        level: Two-sided confidence level, strictly in (0, 1).
        method: ``"wilson"`` or ``"clopper_pearson"``.

    Returns:
        JSON-safe dict with ``estimate``, ``lower``, ``upper`` (floats in
        [0, 1]), ``method`` (str), ``level`` (float), ``n`` and ``k`` (ints).

    Raises:
        ValueError: On invalid ``terminal``, ``s0``, ``level`` or ``method``.

    The input is not modified.
    """
    if not isinstance(method, str) or method not in _PROPORTION_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_PROPORTION_METHODS)}, got {method!r}."
        )
    prices = _validate_terminal(terminal)
    initial = _validate_real(s0, "s0")
    confidence = _validate_level(level)

    n = int(prices.size)
    k = int(np.count_nonzero(prices < initial))
    lower, upper = _PROPORTION_METHODS[method](k, n, confidence)
    return {
        "estimate": k / n,
        "lower": float(lower),
        "upper": float(upper),
        "method": method,
        "level": confidence,
        "n": n,
        "k": k,
    }


def _terminal_losses(prices: np.ndarray, initial: float) -> np.ndarray:
    """Losses ``s0 - S_T``, raising like ``risk_summary`` if not finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        losses = initial - prices
    if not np.isfinite(losses).all():
        raise ValueError("Terminal losses exceed finite precision.")
    return losses


def var_interval(terminal: Any, s0: Any, confidence: float = 0.95,
                 level: float = 0.95) -> dict[str, Any]:
    """Distribution-free confidence interval for the loss quantile (VaR).

    Target: ``q_p``, the ``p = confidence`` quantile of ``L = s0 - S_T``
    (price units, may be negative). Point estimate: ``risk_summary``'s VaR,
    i.e. NumPy's linear-interpolation percentile ``_linear_percentile``
    of the losses (identical value, same function).

    Interval: with ``a = 1 - level``, ``F`` the Binom(N, p) CDF and sorted
    losses ``L_(1) <= ... <= L_(N)``, the 1-based ranks are
    ``j = F^-1(a/2)`` and ``k = F^-1(1 - a/2) + 1`` (``F^-1`` = ``binom.ppf``),
    each clipped to [1, N]; the interval is ``[L_(j), L_(k)]``. For continuous
    L, ``B = #{L_i <= q_p} ~ Binom(N, p)`` and
    ``P(L_(j) <= q_p <= L_(k)) = P(j <= B <= k - 1) = F(k-1) - F(j-1)``,
    which is reported as ``coverage`` for the ranks actually used. Unclipped,
    ``F(j-1) < a/2`` and ``F(k-1) >= 1 - a/2``, so ``coverage > level``
    (conservative). Clipping happens when N is small for the given p
    (``saturated``); it can lower ``coverage`` below ``level`` on either side:
    an upper rank ``N + 1`` (an unbounded endpoint) becomes ``N``, and a lower
    rank 0 becomes 1, which drops the mass ``F(0) = (1-p)^N >= a/2``. The
    reported ``coverage`` is exact for continuous L in all cases; e.g. N = 1
    gives ``[L_(1), L_(1)]`` with coverage 0.

    If L has atoms (ties), the same interval still covers any p-quantile with
    probability >= ``coverage`` (``P(L < q) <= p <= P(L <= q)``), i.e. it is
    conservative. All draws must be i.i.d.; antithetic or otherwise
    correlated samples invalidate the coverage.

    Point vs interval: the linear percentile interpolates between the order
    statistics of 1-based ranks ``floor(h) + 1`` and ``ceil(h) + 1`` with
    ``h = (N - 1) p``, an O(1/N) rank offset from ``N p``. Usually
    ``j <= floor(h) + 1`` and ``ceil(h) + 1 <= k``, so the estimate lies in
    the interval, but this is not enforced: for very small N or a small
    ``level`` (ranks near the binomial median) the estimate can fall outside
    ``[lower, upper]``. No adjustment is made.

    Args:
        terminal: Simulated terminal prices, finite 1-D array, size N >= 1.
        s0: Initial price (finite real, not bool), asset currency.
        confidence: VaR level p, strictly in (0, 1) (as in ``risk_summary``).
        level: Two-sided interval level, strictly in (0, 1).

    Returns:
        JSON-safe dict: ``estimate``, ``lower``, ``upper`` (price units),
        ``lower_rank``, ``upper_rank`` (1-based ranks after clipping),
        ``saturated`` (bool), ``coverage`` (float in [0, 1]), ``method``
        (``"order_statistics"``), ``level``, ``confidence``, ``n``.

    Raises:
        ValueError: On invalid inputs or non-finite losses.

    Memory is O(N) (losses and one partitioned copy). The input is not modified.
    """
    prices = _validate_terminal(terminal)
    initial = _validate_real(s0, "s0")
    p = _validate_level(confidence, "confidence")
    interval_level = _validate_level(level)

    losses = _terminal_losses(prices, initial)
    n = int(losses.size)
    estimate = float(_linear_percentile(losses, p))

    alpha = 1.0 - interval_level
    requested_lower = int(stats.binom.ppf(alpha / 2, n, p))
    requested_upper = int(stats.binom.ppf(1 - alpha / 2, n, p)) + 1
    lower_rank = min(max(requested_lower, 1), n)
    upper_rank = min(max(requested_upper, 1), n)
    saturated = (lower_rank != requested_lower or upper_rank != requested_upper)

    selected = np.partition(losses, (lower_rank - 1, upper_rank - 1))
    lower = float(selected[lower_rank - 1])
    upper = float(selected[upper_rank - 1])
    coverage = float(stats.binom.cdf(upper_rank - 1, n, p)
                     - stats.binom.cdf(lower_rank - 1, n, p))
    coverage = min(max(coverage, 0.0), 1.0)
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "lower_rank": lower_rank,
        "upper_rank": upper_rank,
        "saturated": bool(saturated),
        "coverage": coverage,
        "method": "order_statistics",
        "level": interval_level,
        "confidence": p,
        "n": n,
    }
