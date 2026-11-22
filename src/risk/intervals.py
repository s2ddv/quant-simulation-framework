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

The private ``_validate_*`` helpers are shared input checks intended for the
VaR and CVaR interval functions of this module as well.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

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


def _validate_level(level: Any) -> float:
    """Return a confidence level strictly inside (0, 1)."""
    result = _validate_real(level, "level")
    if not 0 < result < 1:
        raise ValueError(f"level must be strictly between 0 and 1, got {result}.")
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
