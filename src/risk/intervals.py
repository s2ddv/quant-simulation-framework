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
* ``cvar_interval``: interval for the CVaR (mean loss at or beyond VaR, ties
  included), either asymptotic-normal (Brazauskas et al. 2008) or i.i.d.
  percentile bootstrap with a dedicated, seedable RNG stream.

All intervals assume i.i.d. simulated paths. Variance-reduction schemes that
correlate draws (antithetic pairs, stratification, QMC) invalidate them.

The private ``_validate_*`` helpers are shared input checks intended for the
VaR and CVaR interval functions of this module as well.
"""

from __future__ import annotations

import zlib
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy import stats

from risk.statistics import _finite_mean, _linear_percentile
from validation.tail_probability import clopper_pearson_interval, wilson_interval

# Stable integer spawn key of the bootstrap stream. NumPy's SeedSequence does
# not accept strings in ``spawn_key``, so the planned label "cvar_bootstrap"
# is mapped to its CRC-32 (approved deviation from docs/plano-expansao.md).
_CVAR_BOOTSTRAP_STREAM = zlib.crc32(b"cvar_bootstrap")
_CVAR_METHODS = ("asymptotic", "bootstrap")
# Peak working memory of one bootstrap batch (see ``_bootstrap_batch_size``).
_BOOTSTRAP_MEMORY_BUDGET = 64 * 2**20
_BOOTSTRAP_BYTES_PER_ELEMENT = 24

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
    ``h = (N - 1) p``. The interpolated 1-based rank is
    ``h + 1 = N p + (1 - p)``: an O(1) offset in rank, i.e. O(1/N) in
    probability level (``(1 - p) / N``), which for a smooth density ``f`` is
    about ``(1 - p) / (N f(q))`` in price. Usually
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


def _validate_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` as a Python int ``>= minimum``; reject bool."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _cvar_from_losses(losses: np.ndarray, confidence: float) -> tuple[float, float, np.ndarray]:
    """VaR, CVaR and tail of ``losses`` exactly as in ``risk_summary``.

    ``var`` is the linear percentile; the tail is ``losses >= var`` (ties
    included) and CVaR its overflow-safe mean (``_finite_mean``).
    """
    var = _linear_percentile(losses, confidence)
    tail = losses[losses >= var]
    return float(var), float(_finite_mean(tail)), tail


def _bootstrap_batch_size(n: int) -> int:
    """Rows per batch so that one batch stays within the memory budget.

    Per resampled element: int64 index (8 B), gathered float64 losses (8 B)
    and the float64 copy made by ``np.partition`` along axis 1 (8 B), i.e.
    24 B * rows * N. Per-row temporaries (boolean mask, tail copy: O(N)) and
    the per-row results are negligible next to that.
    """
    return max(1, _BOOTSTRAP_MEMORY_BUDGET // (_BOOTSTRAP_BYTES_PER_ELEMENT * n))


def _row_linear_percentile(samples: np.ndarray, confidence: float) -> np.ndarray:
    """``_linear_percentile`` applied to each row of a 2-D array.

    Same rank, the same two order statistics (via ``np.partition`` on axis 1)
    and the same interpolation branches, so every row matches the 1-D
    function bit for bit.
    """
    n = samples.shape[1]
    rank = (n - 1) * (confidence * 100 / 100)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    selected = np.partition(samples, (lower, upper), axis=1)
    left = selected[:, lower]
    right = selected[:, upper]
    weight = rank - lower
    with np.errstate(over="ignore", invalid="ignore"):
        difference = right - left
        unsafe = (1 - weight) * left + weight * right
        if weight >= 0.5:
            safe = right - difference * (1 - weight)
        else:
            safe = left + difference * weight
    return np.where(np.isfinite(difference), safe, unsafe)


def _bootstrap_cvars(losses: np.ndarray, confidence: float, n_bootstrap: int,
                     rng: np.random.Generator, batch_size: int) -> np.ndarray:
    """CVaR of ``n_bootstrap`` i.i.d. resamples of ``losses``, in batches.

    Indices are drawn one row (resample) at a time from the single generator,
    so the replicate sequence depends only on the generator state, never on
    ``batch_size`` or on NumPy's internal buffering across calls.
    """
    n = losses.size
    cvars = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        rows = min(batch_size, n_bootstrap - start)
        indices = np.empty((rows, n), dtype=np.int64)
        for row in range(rows):
            indices[row] = rng.integers(0, n, size=n)
        samples = losses[indices]
        del indices
        var_rows = _row_linear_percentile(samples, confidence)
        for row in range(rows):
            sample = samples[row]
            # Same tail and the same _finite_mean as risk_summary, per row,
            # so each replicate is identical to _cvar_from_losses(sample).
            cvars[start + row] = _finite_mean(sample[sample >= var_rows[row]])
    return cvars


def cvar_interval(terminal: Any, s0: Any, confidence: float = 0.95,
                  level: float = 0.95, method: str = "asymptotic", *,
                  n_bootstrap: int = 1000, seed: int | None = None,
                  batch_size: int | None = None) -> dict[str, Any]:
    """Confidence interval for the CVaR (expected shortfall) estimator.

    Definition (identical to ``risk_summary``): with losses ``L = s0 - S_T``
    and ``p = confidence``, ``var_hat`` is NumPy's linear percentile of L and
    ``CVaR_hat`` is the mean of the losses ``>= var_hat`` (ties included).
    ``estimate`` equals ``risk_summary(paths, confidence)["cvar"]`` exactly.
    Units: asset currency; values may be negative (a gain).

    ``method="asymptotic"`` (Brazauskas, Jones, Puri and Zitikis, 2008,
    J. Statist. Plann. Inference 138; Manistre and Hancock, 2005, NAAJ 9(2))::

        Var(CVaR_hat) ~ [Var(L | L >= VaR) + p (CVaR - VaR)^2] / (N (1 - p))

    with the conditional variance estimated by the ddof=1 sample variance
    of the same tail used for the mean, VaR/CVaR by their point estimates,
    and the normal interval ``estimate +- z SE``, ``z = Phi^-1(1/2 + level/2)``.
    It is a large-sample approximation that requires a finite second moment
    of L in the tail (true for GBM, where L <= s0); for heavy tails or small
    tails ``N (1 - p)`` it can under-cover. The denominator follows the plan,
    ``N (1 - p)`` (the expected tail size); the observed ``n_tail`` differs
    by O(1) for continuous L but can be much larger with ties at VaR, where
    using it would shrink the SE. With ``n_tail < 2`` the conditional
    variance is undefined: ``lower``/``upper``/``standard_error`` are None and
    ``note`` explains why. Variance terms are computed on losses scaled by
    their largest magnitude, so neither squaring nor summing overflows or
    underflows when the result is representable.

    ``method="bootstrap"``: i.i.d. percentile bootstrap. ``n_bootstrap``
    resamples of size N are drawn with replacement, the CVaR of each is
    computed with the same definition (row-wise linear percentile plus the
    ``_finite_mean`` of its ties-included tail, bitwise identical to the 1-D
    computation), and the interval is the linear ``np.quantile`` of the
    replicate CVaRs at ``a/2`` and ``1 - a/2`` (``a = 1 - level``);
    ``bootstrap_standard_error`` is their ddof=1 standard deviation. RNG:
    ``default_rng(SeedSequence(seed, spawn_key=(_CVAR_BOOTSTRAP_STREAM,)))``
    with ``_CVAR_BOOTSTRAP_STREAM = zlib.crc32(b"cvar_bootstrap")`` (an
    integer key, since NumPy rejects string spawn keys); ``seed=None`` uses
    fresh OS entropy and is reported as None (not reproducible). The
    replicates are independent of ``batch_size``, which only bounds memory:
    by default it sizes one batch at about 64 MiB (24 bytes per resampled
    element), but never below one replicate row, so for N above about 2.8
    million the batch is a single row of O(N) memory that exceeds 64 MiB.
    With ``n_tail < 2`` the numeric bounds are still returned, plus a
    ``note`` saying the interval has no meaningful coverage. The percentile
    bootstrap of a tail mean can under-cover when the tail holds few points,
    and its interval need not contain ``estimate``.

    All draws must be i.i.d. (no antithetic or otherwise correlated paths).

    Args:
        terminal: Simulated terminal prices, finite 1-D array, size N >= 1.
        s0: Initial price (finite real, not bool), asset currency.
        confidence: VaR/CVaR level p, strictly in (0, 1).
        level: Two-sided interval level, strictly in (0, 1).
        method: ``"asymptotic"`` or ``"bootstrap"``.
        n_bootstrap: Bootstrap replicates B, integer >= 2 (bootstrap only).
        seed: Integer >= 0 or None (bootstrap only).
        batch_size: Replicates per batch, integer >= 1, or None for the
            memory-budget default (bootstrap only).

    Returns:
        JSON-safe dict with ``estimate``, ``lower``, ``upper``, ``method``,
        ``level``, ``confidence``, ``n``, ``n_tail``, ``var``; plus
        ``standard_error`` (and ``note`` when undefined) for "asymptotic",
        or ``bootstrap_standard_error``, ``n_bootstrap``, ``seed`` and
        ``rng`` (and ``note`` when ``n_tail < 2``) for "bootstrap".

    Raises:
        ValueError: On invalid inputs or non-finite losses or results.

    The input is not modified.
    """
    if not isinstance(method, str) or method not in _CVAR_METHODS:
        raise ValueError(
            f"method must be one of {list(_CVAR_METHODS)}, got {method!r}."
        )
    prices = _validate_terminal(terminal)
    initial = _validate_real(s0, "s0")
    p = _validate_level(confidence, "confidence")
    interval_level = _validate_level(level)
    replicates = _validate_int(n_bootstrap, "n_bootstrap", 2)
    seed_value = None if seed is None else _validate_int(seed, "seed", 0)
    batch = None if batch_size is None else _validate_int(batch_size, "batch_size", 1)

    losses = _terminal_losses(prices, initial)
    n = int(losses.size)
    var, estimate, tail = _cvar_from_losses(losses, p)
    n_tail = int(tail.size)
    result: dict[str, Any] = {
        "estimate": estimate, "lower": None, "upper": None, "method": method,
        "level": interval_level, "confidence": p, "n": n, "n_tail": n_tail,
        "var": var,
    }
    alpha = 1.0 - interval_level

    if method == "asymptotic":
        result["standard_error"] = None
        if n_tail < 2:
            result["note"] = ("Conditional tail variance is undefined with "
                              "fewer than 2 tail losses; no interval.")
            return result
        scale = max(float(np.max(np.abs(tail))), abs(var), abs(estimate))
        if scale == 0 or bool(np.all(tail == var)):
            # Degenerate tail (e.g. sigma = 0): conditional variance and the
            # CVaR - VaR gap are exactly 0; skip the 1-ulp rounding of the mean.
            standard_error = 0.0
        else:
            tail_sd = float(np.std(tail / scale, ddof=1))
            gap = estimate / scale - var / scale
            standard_error = (float(np.hypot(tail_sd, np.sqrt(p) * gap))
                              / float(np.sqrt(n * (1 - p))) * scale)
        z = float(stats.norm.ppf(0.5 + interval_level / 2))
        with np.errstate(over="ignore", invalid="ignore"):
            lower = estimate - z * standard_error
            upper = estimate + z * standard_error
        if not np.isfinite([standard_error, lower, upper]).all():
            raise ValueError("CVaR interval exceeds finite precision.")
        result.update({"lower": float(lower), "upper": float(upper),
                       "standard_error": float(standard_error)})
        return result

    rng = np.random.default_rng(
        np.random.SeedSequence(seed_value, spawn_key=(_CVAR_BOOTSTRAP_STREAM,)))
    cvars = _bootstrap_cvars(losses, p, replicates, rng,
                             batch if batch is not None else _bootstrap_batch_size(n))
    if not np.isfinite(cvars).all():
        raise ValueError("Bootstrap CVaR replicates exceed finite precision.")
    lower, upper = np.quantile(cvars, [alpha / 2, 1 - alpha / 2], method="linear")
    scale = float(np.max(np.abs(cvars)))
    bootstrap_se = (0.0 if scale == 0
                    else float(np.std(cvars / scale, ddof=1)) * scale)
    if not np.isfinite([lower, upper, bootstrap_se]).all():
        raise ValueError("CVaR interval exceeds finite precision.")
    result.update({
        "lower": float(lower), "upper": float(upper),
        "bootstrap_standard_error": float(bootstrap_se),
        "n_bootstrap": replicates, "seed": seed_value,
        "rng": "NumPy default_rng/PCG64",
    })
    if n_tail < 2:
        result["note"] = ("Bootstrap interval from fewer than 2 tail losses "
                          "has no meaningful coverage; bounds are reported "
                          "but should not be used as a confidence interval.")
    return result
