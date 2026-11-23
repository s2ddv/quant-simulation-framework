"""European options under the Merton (1976) jump-diffusion: Poisson series of BSM prices.

Model under the pricing measure Q (constant parameters, continuous dividend
yield ``q``)::

    dS/S_- = (r - q - lambda*k) dt + sigma dW^Q + (J - 1) dN,
    ln J ~ N(m_J, delta^2),  N ~ Poisson(lambda),  k = E[J - 1] = exp(m_J + delta^2/2) - 1

Conditional on ``N_T = n`` the log-price is normal, which gives the series::

    V = sum_n w_n * BSM(S, K, T, r_n, q, sigma_n, kind)
    w_n      = exp(-lambda' T) (lambda' T)^n / n!,   lambda' = lambda (1 + k)
    sigma_n^2 = sigma^2 + n delta^2 / T
    r_n      = r - lambda k + n ln(1 + k) / T,       ln(1 + k) = m_J + delta^2/2

The BSM terms are :func:`pricing.black_scholes.bs_price` (not reimplemented).

Measure (read before use). Merton's hypothesis is used: jump risk is
diversifiable (zero beta) and therefore NOT priced, so ``lambda``, ``m_J`` and
``delta`` are the same under P and Q and only the diffusion drift is changed
to ``r - q - lambda*k``. The market with jumps is incomplete: the result is
not a replication price but the expectation under this particular Q. It
equals the Monte Carlo price obtained by simulating the jump-diffusion with
PRICE drift ``mu = r - q`` (keeping ``lambda, m_J, delta``) and discounting at
``r``. Pricing with the historical drift ``mu`` is forbidden (plan §2.1);
historical ``sigma`` or jump parameters are proxies and must be labelled via
:func:`merton_pricing_metadata`.

Units: ``S``, ``K`` and prices in the asset's currency; ``T`` in years;
``r``, ``q`` and ``lambda`` in year^-1; ``sigma`` in year^-1/2; ``m_J`` and
``delta`` in log-price units per jump.

Truncation. Terms are summed for ``n`` in ``[n_lo, n_hi]`` where the Poisson
mass outside the range is below ``tol`` both for ``Pois(lambda' T)`` and for
``Pois(lambda T)``. Since every call term satisfies ``BSM_call <= S e^{-qT}``
and ``w_n * BSM_put_n <= K e^{-rT} Pois(n; lambda T)``, the truncation error is
at most ``tol * S e^{-qT}`` (call) or ``tol * K e^{-rT}`` (put).

Sources: Merton, R. C. "Option pricing when underlying stock returns are
discontinuous". *Journal of Financial Economics* 3:125–144, 1976; Hull, J.
*Options, Futures, and Other Derivatives* (chapter on jump-diffusion models).
The formula was derived here and verified numerically (limits and parity in
the tests); equation and page numbers were not checked in the originals.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from pricing.black_scholes import (
    FloatArray,
    OptionKind,
    SigmaSource,
    _kind,
    _real,
    _strike,
    bs_price,
    pricing_metadata,
)

MAX_SERIES_TERMS = 200_000


def _series_range(lam_prime_t: float, lam_t: float, tol: float) -> tuple[int, int]:
    """Smallest ``[n_lo, n_hi]`` leaving Poisson mass < tol outside, for both means."""
    los, his = [], []
    for mean in (lam_prime_t, lam_t):
        lo_m = int(poisson.ppf(0.5 * tol, mean))   # P(N < lo_m) < tol/2
        hi_m = int(poisson.isf(0.5 * tol, mean))   # P(N > hi_m) <= tol/2
        while poisson.sf(hi_m, mean) > 0.5 * tol:
            hi_m += 1
        los.append(lo_m)
        his.append(hi_m)
    lo, hi = min(los), max(his)
    return max(lo, 0), hi


def merton_price(S: float, K: Any, T: float, r: float, q: float, sigma: float,
                 jump_intensity: float, jump_mean: float, jump_std: float,
                 kind: OptionKind, *, tol: float = 1e-14) -> float | FloatArray:
    """Merton (1976) European call/put price (see the module docstring).

    Parameters
    ----------
    S : spot (> 0). K : strike (> 0); a scalar returns ``float``, an array
        (``ndim >= 1``) returns an ndarray of the same shape.
    T : years (>= 0). r, q : continuous rates, year^-1 (may be negative).
    sigma : diffusion volatility, year^-1/2 (>= 0).
    jump_intensity : lambda, jumps per year (>= 0).
    jump_mean, jump_std : mean (real) and std (>= 0) of the log jump size.
    kind : ``"call"`` or ``"put"``.
    tol : Poisson mass neglected (0 < tol < 1); the truncation error is at
        most ``tol*S*e^{-qT}`` (call) or ``tol*K*e^{-rT}`` (put).

    Special cases: ``T == 0`` returns the payoff; ``lambda == 0`` returns
    ``bs_price`` exactly; ``sigma == 0`` with ``delta > 0`` is regular for
    ``n >= 1`` and ``bs_price`` handles the zero-variance ``n = 0`` term;
    ``delta == 0`` gives deterministic jump sizes. Invalid inputs, more than
    ``MAX_SERIES_TERMS`` terms, or non-finite intermediate values raise
    ``ValueError``; ``nan``/``inf`` are never returned.
    """
    s = _real("S", S, positive=True)
    k_arr, scalar = _strike(K)
    t = _real("T", T, nonnegative=True)
    rr = _real("r", r)
    qq = _real("q", q)
    sig = _real("sigma", sigma, nonnegative=True)
    lam = _real("jump_intensity", jump_intensity, nonnegative=True)
    m_j = _real("jump_mean", jump_mean)
    delta = _real("jump_std", jump_std, nonnegative=True)
    kd = _kind(kind)
    eps = _real("tol", tol, positive=True)
    if not eps < 1.0:
        raise ValueError("tol must satisfy 0 < tol < 1")
    k_scalar_or_array: Any = float(k_arr) if scalar else k_arr

    if t == 0.0 or lam == 0.0:
        # No jump can occur (or no time for one): plain BSM / payoff.
        return bs_price(s, k_scalar_or_array, t, rr, qq, sig, kd)

    log1pk = m_j + 0.5 * delta * delta              # ln(1 + k), exact
    if log1pk > 700.0:
        raise ValueError("jump_mean + jump_std^2/2 is too large: 1 + k overflows")
    k = math.expm1(log1pk)
    lam_t = lam * t
    lam_prime_t = lam_t * math.exp(log1pk)
    if not (math.isfinite(lam_t) and math.isfinite(lam_prime_t)):
        raise ValueError("lambda*T*(1 + k) is not finite")
    if lam_prime_t == 0.0:
        return bs_price(s, k_scalar_or_array, t, rr, qq, sig, kd)
    n_lo, n_hi = _series_range(lam_prime_t, lam_t, eps)
    if n_hi - n_lo + 1 > MAX_SERIES_TERMS:
        raise ValueError(
            f"Merton series needs {n_hi - n_lo + 1} terms (lambda*T too large); "
            f"limit is MAX_SERIES_TERMS = {MAX_SERIES_TERMS}")

    log_lpt = math.log(lam_prime_t)
    total = np.zeros_like(k_arr, dtype=np.float64)
    for n in range(n_lo, n_hi + 1):
        log_w = -lam_prime_t + n * log_lpt - float(gammaln(n + 1.0))
        w = math.exp(log_w)
        if w == 0.0:
            continue
        sigma_n = math.sqrt(sig * sig + n * delta * delta / t)
        r_n = rr - lam * k + n * log1pk / t
        total = total + w * np.asarray(bs_price(s, k_scalar_or_array, t, r_n, qq, sigma_n, kd))
    out = np.maximum(total, 0.0) + 0.0
    if not np.all(np.isfinite(out)):
        raise ValueError("Merton price is not finite for these inputs")
    return float(out) if scalar else out


def merton_pricing_metadata(r: float, q: float, sigma: float,
                            sigma_source: SigmaSource, jump_intensity: float,
                            jump_mean: float, jump_std: float) -> dict[str, Any]:
    """:func:`pricing_metadata` extended for :func:`merton_price`.

    Records ``model = "merton_jump_diffusion"``, ``measure = "risk_neutral"``,
    the jump parameters and ``jump_risk = "unpriced"`` (Merton's hypothesis:
    jump parameters identical under P and Q; incomplete market, not a
    replication price).
    """
    meta = pricing_metadata(r, q, sigma, sigma_source)
    meta.update({
        "model": "merton_jump_diffusion",
        "measure": "risk_neutral",
        "jump_intensity": _real("jump_intensity", jump_intensity, nonnegative=True),
        "jump_mean": _real("jump_mean", jump_mean),
        "jump_std": _real("jump_std", jump_std, nonnegative=True),
        "jump_risk": "unpriced",
    })
    return meta
