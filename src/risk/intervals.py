"""Monte Carlo confidence intervals for simulated terminal risk metrics.

Scope and hypotheses
--------------------
The intervals below quantify **Monte Carlo sampling error only**: the
uncertainty of each estimator caused by using a finite number ``n`` of iid
simulated paths, *conditional on fixed GBM parameters* (``mu``, ``sigma``,
``s0`` and horizon). They do **not** include calibration uncertainty in the
historically estimated ``mu``/``sigma``, nor model error (GBM versus real
prices). Increasing ``n_sims`` shrinks these intervals towards zero while the
other error sources remain.

Units and conventions (identical to :func:`risk.statistics.risk_summary`):

* loss ``L = s0 - S_T`` in the asset's price currency; negative means a gain;
* VaR is the linear-interpolation percentile of the losses at ``confidence``
  and CVaR the mean of losses greater than or equal to VaR (ties included);
  both are in price units and may be negative;
* probability of loss is ``P(S_T < s0)`` (strict), dimensionless.

``confidence`` is the risk level (e.g. 0.95 VaR); ``level`` is the coverage
probability of the interval (e.g. a 95% interval). They are independent.
"""

import math
from statistics import NormalDist

import numpy as np

from risk.statistics import risk_summary

PARAMETER_SCOPE = "fixed_gbm_parameters_monte_carlo_error_only"
# 2e6 float64 = ~16 MB per array; a chunk holds ~4 such temporaries at once
# (indices, sample, quantile copy, tail products/mask), so ~66 MB peak.
_BOOTSTRAP_CHUNK_ELEMENTS = 2_000_000
# Below this many tail points the percentile bootstrap undercovers markedly.
MIN_RELIABLE_TAIL_POINTS = 50


def _validate_probability(value, name):
    if not isinstance(value, (int, float, np.floating)) or not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, exclusively.")


def _validate_losses(losses):
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty 1D array.")
    if not np.isfinite(losses).all():
        raise ValueError("losses must contain only finite values.")
    return losses


def _binomial_order_indices(n, q, alpha):
    """0-based order-statistic indices for a distribution-free quantile CI.

    With ``K ~ Binomial(n, q)`` the number of draws <= the true quantile,
    ``P(X_(k+1) > xi_q) = P(K <= k)``. The lower index is the largest ``k``
    with ``P(K <= k) <= alpha/2``; the upper index the smallest ``k`` with
    ``P(K <= k) >= 1 - alpha/2``. Returns ``(lower, upper, achieved)`` where an
    index is ``None`` when no order statistic satisfies its tail (n too small).

    The CDF is evaluated exactly (via ``lgamma``) on a window around ``n*q``.
    The window half-width ``t`` solves Bernstein's bound
    ``exp(-t^2 / (2(v + t/3))) = 1e-6 * alpha`` with ``v = n q (1-q)``, so the
    ignored mass on each side is below ``1e-6 * alpha``; work is O(sqrt(n)).
    """
    c = math.log(1.0 / (1e-6 * alpha))
    v = n * q * (1 - q)
    t = c / 3 + math.sqrt(c * c / 9 + 2 * c * v)
    lo = max(0, math.floor(n * q - t) - 1)
    hi = min(n, math.ceil(n * q + t) + 1)
    log_q, log_1q = math.log(q), math.log1p(-q)
    log_norm = math.lgamma(n + 1)
    ks = np.arange(lo, hi + 1)
    log_pmf = np.array([log_norm - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + k * log_q + (n - k) * log_1q for k in ks])
    cdf = np.minimum(np.cumsum(np.exp(log_pmf)), 1.0)

    below = np.nonzero(cdf <= alpha / 2)[0]
    lower = int(ks[below[-1]]) if below.size else None
    above = np.nonzero(cdf >= 1 - alpha / 2)[0]
    upper = int(ks[above[0]]) if above.size and ks[above[0]] <= n - 1 else None

    cdf_lower = 0.0 if lower is None else float(cdf[lower - lo])
    cdf_upper = 1.0 if upper is None else float(cdf[upper - lo])
    return lower, upper, cdf_upper - cdf_lower


def var_confidence_interval(losses, confidence=0.95, *, level=0.95):
    """Distribution-free CI for the VaR (loss quantile) from order statistics.

    For iid draws from a continuous loss law, ``[L_(l), L_(u)]`` covers the
    true ``confidence``-quantile with probability ``B(u-1) - B(l-1) >= level``
    (``B`` the Binomial(n, confidence) CDF), exactly for every ``n``; with
    atoms (ties) the closed interval remains conservative. When ``n`` is too
    small for a tail, that bound is ``None`` (unbounded) and
    ``achieved_coverage`` accounts for it. Bounds are in price units.
    """
    losses = _validate_losses(losses)
    _validate_probability(confidence, "confidence")
    _validate_probability(level, "level")
    n = losses.size
    lower_idx, upper_idx, achieved = _binomial_order_indices(
        n, float(confidence), 1 - float(level))
    wanted = [i for i in (lower_idx, upper_idx) if i is not None]
    selected = np.partition(losses, wanted) if wanted else losses
    return {
        "level": float(level), "confidence": float(confidence),
        "unit": "asset_price_currency",
        "method": "order_statistic_binomial_exact",
        "parameter_scope": PARAMETER_SCOPE,
        "n": int(n),
        "lower": None if lower_idx is None else float(selected[lower_idx]),
        "upper": None if upper_idx is None else float(selected[upper_idx]),
        "lower_order_statistic": None if lower_idx is None else lower_idx + 1,
        "upper_order_statistic": None if upper_idx is None else upper_idx + 1,
        "achieved_coverage": float(achieved),
    }


def cvar_bootstrap_interval(losses, confidence=0.95, *, level=0.95,
                            n_bootstrap=1000, seed=0):
    """Percentile-bootstrap CI for CVaR (mean of losses >= VaR, ties included).

    Each resample draws ``n`` losses with replacement using
    ``numpy.random.default_rng(seed)`` (PCG64), recomputes the linear VaR and
    the tail mean, and the interval is the ``(1-level)/2`` and
    ``(1+level)/2`` linear percentiles of the replicates. Coverage is
    asymptotic and tends to be below nominal when the tail holds few points
    (``n * (1 - confidence)`` small); bounds are ``None`` for ``n < 2``. The
    same seed reproduces results only for the same NumPy RNG stream and
    sample. Values are scaled by ``max|L|`` internally so no step overflows.
    """
    losses = _validate_losses(losses)
    _validate_probability(confidence, "confidence")
    _validate_probability(level, "level")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) \
            or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2.")
    n = losses.size
    result = {
        "level": float(level), "confidence": float(confidence),
        "unit": "asset_price_currency", "method": "percentile_bootstrap",
        "parameter_scope": PARAMETER_SCOPE, "n": int(n),
        "n_bootstrap": int(n_bootstrap), "seed": seed,
        "rng": "NumPy default_rng/PCG64, integers(0, n) row-chunked",
        "lower": None, "upper": None,
        "tail_points": float(n * (1 - confidence)),
    }
    if result["tail_points"] < MIN_RELIABLE_TAIL_POINTS:
        result["warning"] = (
            f"fewer than {MIN_RELIABLE_TAIL_POINTS} expected tail points; the "
            "percentile bootstrap can cover well below the nominal level")
    if n < 2:
        return result

    scale = float(np.max(np.abs(losses)))
    if scale == 0.0:
        result.update({"lower": 0.0, "upper": 0.0})
        return result
    normalized = losses / scale
    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_bootstrap))
    chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // n)
    for start in range(0, int(n_bootstrap), chunk):
        rows = min(chunk, int(n_bootstrap) - start)
        sample = normalized[rng.integers(0, n, size=(rows, n))]
        var = np.quantile(sample, confidence, axis=1, method="linear")
        tail = sample >= var[:, None]
        replicates[start:start + rows] = (np.sum(sample * tail, axis=1)
                                          / np.sum(tail, axis=1))
    alpha = 1 - float(level)
    lower, upper = np.quantile(replicates, [alpha / 2, 1 - alpha / 2],
                               method="linear") * scale
    if not np.isfinite([lower, upper]).all():
        raise ValueError("CVaR confidence interval exceeds finite precision.")
    result.update({"lower": float(lower), "upper": float(upper)})
    return result


def prob_loss_wilson_interval(n_losses, n, *, level=0.95):
    """Wilson score CI for the probability of loss ``P(S_T < s0)``.

    ``n_losses`` is the count of simulated paths with ``S_T < s0`` out of
    ``n`` iid paths. Endpoints are
    ``(p + z^2/(2n) +/- z sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)`` with
    ``z = Phi^-1((1+level)/2)``; they stay inside [0, 1] and are defined for
    ``p = 0`` or ``1``. Coverage is approximate (close to nominal for
    moderate n); a zero count does not establish zero probability.
    """
    for value, name in ((n_losses, "n_losses"), (n, "n")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
    if n < 1 or not 0 <= n_losses <= n:
        raise ValueError("Require n >= 1 and 0 <= n_losses <= n.")
    _validate_probability(level, "level")
    n, n_losses = int(n), int(n_losses)
    z = NormalDist().inv_cdf((1 + float(level)) / 2)
    p = n_losses / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "level": float(level), "unit": "probability",
        "method": "wilson_score", "parameter_scope": PARAMETER_SCOPE,
        "n": n, "estimate": p,
        "lower": 0.0 if n_losses == 0 else max(0.0, center - half),
        "upper": 1.0 if n_losses == n else min(1.0, center + half),
    }


def risk_confidence_intervals(paths, confidence=0.95, *, level=0.95,
                              n_bootstrap=1000, seed=0):
    """Point estimates from ``risk_summary`` plus Monte Carlo CIs.

    Args:
        paths: Finite prices shaped ``(steps + 1, n_sims)`` with common s0.
        confidence: VaR/CVaR loss percentile level, in (0, 1).
        level: Two-sided coverage of each interval, in (0, 1).
        n_bootstrap: CVaR bootstrap replicates (>= 2).
        seed: Seed for the CVaR bootstrap RNG.

    Returns:
        ``{"var": ..., "cvar": ..., "prob_loss": ...}``; each entry holds the
        ``estimate`` (identical to ``risk_summary``) and its interval. The
        intervals are marginal (not simultaneous) and exclude calibration
        and model uncertainty. All three assume i.i.d. simulated terminal
        prices: do not apply them to antithetic samples, whose pairs are
        dependent. The CVaR entry carries ``tail_points`` and a ``warning``
        when the tail is small.
    """
    summary = risk_summary(paths, confidence)
    paths = np.asarray(paths, dtype=float)
    s0, terminal = paths[0, 0], paths[-1]
    losses = s0 - terminal
    var = var_confidence_interval(losses, confidence, level=level)
    cvar = cvar_bootstrap_interval(losses, confidence, level=level,
                                   n_bootstrap=n_bootstrap, seed=seed)
    prob = prob_loss_wilson_interval(int(np.count_nonzero(terminal < s0)),
                                     terminal.size, level=level)
    var["estimate"] = summary["var"]
    cvar["estimate"] = summary["cvar"]
    return {"var": var, "cvar": cvar, "prob_loss": prob}
