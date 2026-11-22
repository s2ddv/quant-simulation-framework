"""Analytic Black–Scholes–Merton prices, Greeks and implied volatility.

Model (risk-neutral measure Q, constant parameters, continuous dividend
yield)::

    dS = (r - q) S dt + sigma S dW^Q

European options on one asset, exercised only at ``T``::

    F  = S exp((r - q) T)                               (forward price)
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
    C  = S e^{-qT} Phi(d1)  - K e^{-rT} Phi(d2)
    P  = K e^{-rT} Phi(-d2) - S e^{-qT} Phi(-d1)

Units: ``S`` and ``K`` in the asset's currency; ``T`` in years; ``r`` and ``q``
continuously compounded rates in year^-1 (negative values are allowed);
``sigma`` in year^-1/2. Prices are in the asset's currency.

Measure. These are risk-neutral prices: the drift is ``r - q``, never the
historical price drift ``mu``. A historical volatility used as ``sigma`` is a
proxy for, not an estimate of, implied volatility; record the origin with
:func:`pricing_metadata`. Only continuous dividend yields are supported.

Sources (formulas were checked numerically against the reference values
listed in the tests; the page and example numbers below were NOT verified in
the original texts):

* Black, F.; Scholes, M. "The pricing of options and corporate liabilities".
  *Journal of Political Economy* 81(3), 1973 — no-dividend formula.
* Merton, R. C. "Theory of rational option pricing". *Bell Journal of
  Economics and Management Science* 4(1):141–183, 1973 — continuous dividend
  yield.
* Hull, J. *Options, Futures, and Other Derivatives*, 9th ed.: ch. 15 (BSM
  formula; example S=42, K=40 — example number probably 15.6, not verified),
  ch. 17 (options on indices with yield q), ch. 19 ("The Greek letters").
  Page and example numbers not verified in the source.
* Haug, E. G. *The Complete Guide to Option Pricing Formulas*, 2nd ed., 2007,
  ch. 1 (generalized BSM with cost of carry b = r - q) and ch. 2 (Greeks).
  Section and page numbers not verified in the source.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import ndtr

OptionKind = Literal["call", "put"]
SigmaSource = Literal["historical_proxy", "user_supplied"]
FloatArray = npt.NDArray[np.float64]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_KINDS = ("call", "put")
_SIGMA_SOURCES = ("historical_proxy", "user_supplied")

GREEK_UNITS: dict[str, str] = {
    "delta": "dV/dS, currency per currency of underlying (dimensionless)",
    "gamma": "d2V/dS2, per unit of underlying currency",
    "vega": "dV/dsigma, currency per 1.00 (absolute) change of sigma, not per 1%",
    "theta": "dV/dt calendar = -dV/dT, currency per year",
    "rho": "dV/dr, currency per 1.00 (absolute) change of r, not per 1%",
}


# ---------------------------------------------------------------------------
# Input validation (same conventions as simulation.base: bool rejected, Real
# required, finiteness checked).
# ---------------------------------------------------------------------------

def _real(name: str, value: Any, *, positive: bool = False,
          nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number (bool is rejected)")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _strike(K: Any) -> tuple[FloatArray, bool]:
    """Return strikes as a float64 array and whether the input was scalar."""
    if isinstance(K, bool) or isinstance(K, Real):
        return np.array(_real("K", K, positive=True), dtype=np.float64), True
    arr = np.asarray(K)
    if arr.dtype == np.bool_ or arr.dtype.kind not in "iuf":
        raise ValueError("K must be a real number or an array of real numbers (bool is rejected)")
    if arr.ndim == 0:
        raise ValueError("K must be a Python/NumPy real scalar or an array with ndim >= 1")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("K must be finite")
    if np.any(arr <= 0.0):
        raise ValueError("K must be positive")
    return arr, False


def _kind(kind: Any) -> OptionKind:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("kind must be 'call' or 'put'")
    return "call" if kind == "call" else "put"


def _validate(S: Any, K: Any, T: Any, r: Any, q: Any, sigma: Any,
              kind: Any) -> tuple[float, FloatArray, bool, float, float, float, float, OptionKind]:
    s = _real("S", S, positive=True)
    k, scalar = _strike(K)
    t = _real("T", T, nonnegative=True)
    rr = _real("r", r)
    qq = _real("q", q)
    sig = _real("sigma", sigma, nonnegative=True)
    return s, k, scalar, t, rr, qq, sig, _kind(kind)


def _require_finite(name: str, value: FloatArray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(
            f"{name} is not finite for these inputs (overflow/underflow from "
            "extreme but finite parameters)")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class _Terms:
    """Shared intermediate quantities, computed once per call."""

    def __init__(self, S: float, K: FloatArray, T: float, r: float, q: float,
                 sigma: float) -> None:
        with np.errstate(over="ignore", under="ignore"):
            self.disc_s = float(S * np.exp(-q * T))
            self.disc_k: FloatArray = K * np.exp(-r * T)
        if not math.isfinite(self.disc_s):
            raise ValueError("S*exp(-q*T) is not finite for these inputs")
        _require_finite("K*exp(-r*T)", self.disc_k)
        # log(F/K), computed in log space so S/K itself never overflows.
        self.log_fk: FloatArray = (math.log(S) + (r - q) * T) - np.log(K)
        self.vol_sqrt_t = sigma * math.sqrt(T)
        # Zero variance (T == 0, sigma == 0 or sigma*sqrt(T) underflowing to 0).
        self.degenerate = self.vol_sqrt_t == 0.0
        if self.degenerate:
            self.d1 = self.d2 = None
        else:
            with np.errstate(over="ignore", divide="ignore"):
                d1 = self.log_fk / self.vol_sqrt_t + 0.5 * self.vol_sqrt_t
            self.d1 = d1
            self.d2 = d1 - self.vol_sqrt_t


def _nonnegative(value: FloatArray) -> FloatArray:
    # Rounding (cancellation in deep out-of-the-money options) may produce
    # tiny negatives or -0.0; the exact price is >= 0, so clip and add +0.0.
    return np.maximum(value, 0.0) + 0.0


def _price_array(S: float, K: FloatArray, T: float, r: float, q: float,
                 sigma: float, kind: OptionKind) -> FloatArray:
    t = _Terms(S, K, T, r, q, sigma)
    if t.degenerate:
        # Discounted payoff on the deterministic forward:
        # call = e^{-rT} max(F - K, 0) = max(S e^{-qT} - K e^{-rT}, 0).
        # The indicator uses log(F/K) so that F == K gives exactly 0.0.
        if kind == "call":
            value = np.where(t.log_fk > 0.0, t.disc_s - t.disc_k, 0.0)
        else:
            value = np.where(t.log_fk < 0.0, t.disc_k - t.disc_s, 0.0)
    else:
        assert t.d1 is not None and t.d2 is not None
        # ndtr(+/-d) directly (no 1 - ndtr) to keep tail accuracy; the put
        # uses its own formula, not put-call parity.
        if kind == "call":
            value = t.disc_s * ndtr(t.d1) - t.disc_k * ndtr(t.d2)
        else:
            value = t.disc_k * ndtr(-t.d2) - t.disc_s * ndtr(-t.d1)
    out = _nonnegative(np.asarray(value, dtype=np.float64))
    _require_finite("option price", out)
    return out


@overload
def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             kind: OptionKind) -> float: ...
@overload
def bs_price(S: float, K: npt.ArrayLike, T: float, r: float, q: float,
             sigma: float, kind: OptionKind) -> float | FloatArray: ...
def bs_price(S: float, K: Any, T: float, r: float, q: float, sigma: float,
             kind: OptionKind) -> float | FloatArray:
    """European option price under Black–Scholes–Merton with dividend yield.

    Parameters
    ----------
    S : spot price (> 0, currency).
    K : strike (> 0, currency). A real scalar returns ``float``; a sequence or
        array (``ndim >= 1``) returns a ``float64`` ndarray of the same shape
        (vectorization in ``K`` only).
    T : time to expiry in years (>= 0).
    r, q : continuously compounded risk-free rate and dividend yield, year^-1
        (finite, may be negative).
    sigma : volatility, year^-1/2 (>= 0).
    kind : ``"call"`` or ``"put"``.

    Contract
    --------
    * ``T > 0`` and ``sigma > 0``: Merton (1973) formula (module docstring).
    * ``T == 0`` or ``sigma == 0`` (zero variance): discounted payoff on the
      deterministic forward, ``max(S e^{-qT} - K e^{-rT}, 0)`` for a call and
      ``max(K e^{-rT} - S e^{-qT}, 0)`` for a put; exactly ``0.0`` when
      ``F == K`` (tested as ``log(F/K) == 0``). No division by zero.
    * The result is always ``>= 0`` and never ``-0.0`` (rounding negatives are
      clipped). Invalid or non-finite inputs, and results that overflow, raise
      ``ValueError``; ``nan``/``inf`` are never returned.
    """
    s, k, scalar, t, rr, qq, sig, kd = _validate(S, K, T, r, q, sigma, kind)
    out = _price_array(s, k, t, rr, qq, sig, kd)
    return float(out) if scalar else out


def bs_greeks(S: float, K: Any, T: float, r: float, q: float, sigma: float,
              kind: OptionKind) -> dict[str, Any]:
    """Analytic BSM Greeks with continuous dividend yield ``q``.

    Returns a dict with ``delta, gamma, vega, theta, rho`` (``float`` for a
    scalar ``K``, ndarrays for an array ``K``) and ``units`` (see
    ``GREEK_UNITS``): vega and rho per 1.00 absolute change of sigma and r (not
    per 1%), theta per year as the calendar-time derivative
    ``dV/dt = -dV/dT``. Formulas (Hull 9th ed. ch. 19; Haug 2nd ed. ch. 2;
    numbers not verified in the source), with ``phi`` the standard normal
    density::

        delta_call = e^{-qT} Phi(d1)          delta_put = -e^{-qT} Phi(-d1)
        gamma      = e^{-qT} phi(d1) / (S sigma sqrt(T))
        vega       = S e^{-qT} phi(d1) sqrt(T)
        theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                     - r K e^{-rT} Phi(d2) + q S e^{-qT} Phi(d1)
        theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                     + r K e^{-rT} Phi(-d2) - q S e^{-qT} Phi(-d1)
        rho_call   = K T e^{-rT} Phi(d2)      rho_put = -K T e^{-rT} Phi(-d2)

    Zero variance (``T == 0`` or ``sigma == 0``): the Greeks of the
    discounted deterministic payoff, with ``1{.}`` the indicator and
    ``F = S e^{(r-q)T}``::

        call: delta = e^{-qT} 1{F>K},  rho = K T e^{-rT} 1{F>K},
              theta = (q S e^{-qT} - r K e^{-rT}) 1{F>K}
        put:  delta = -e^{-qT} 1{F<K}, rho = -K T e^{-rT} 1{F<K},
              theta = (r K e^{-rT} - q S e^{-qT}) 1{F<K}
        gamma = vega = 0.

    At ``T == 0`` this gives ``rho = 0`` and ``theta = (qS - rK) 1{F>K}``
    (call); it is the derivative of the zero-variance value, not the limit
    ``T -> 0+`` for ``sigma > 0`` (which diverges at the money). At the exact
    boundary ``F == K`` (``log(F/K) == 0``) the payoff is not differentiable
    and ``ValueError("greeks undefined at zero variance with forward ==
    strike")`` is raised (for an array ``K``, if any element is on it).
    """
    s, k, scalar, t, rr, qq, sig, kd = _validate(S, K, T, r, q, sigma, kind)
    terms = _Terms(s, k, t, rr, qq, sig)
    ds = terms.disc_s
    dk = terms.disc_k
    if terms.degenerate:
        if np.any(terms.log_fk == 0.0):
            raise ValueError("greeks undefined at zero variance with forward == strike")
        itm = (terms.log_fk > 0.0) if kd == "call" else (terms.log_fk < 0.0)
        w = itm.astype(np.float64)
        zero = np.zeros_like(k)
        if kd == "call":
            delta = ds / s * w
            theta = (qq * ds - rr * dk) * w
            rho = dk * t * w
        else:
            delta = -ds / s * w
            theta = (rr * dk - qq * ds) * w
            rho = -dk * t * w
        values = {"delta": delta, "gamma": zero, "vega": zero.copy(),
                  "theta": theta, "rho": rho}
    else:
        assert terms.d1 is not None and terms.d2 is not None
        d1, d2 = terms.d1, terms.d2
        eqt = ds / s  # e^{-qT}
        pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
        sqrt_t = math.sqrt(t)
        with np.errstate(over="ignore"):
            gamma = eqt * pdf / (s * terms.vol_sqrt_t)
        vega = ds * pdf * sqrt_t
        decay = -ds * pdf * sig / (2.0 * sqrt_t)
        if kd == "call":
            nd1, nd2 = ndtr(d1), ndtr(d2)
            delta = eqt * nd1
            theta = decay - rr * dk * nd2 + qq * ds * nd1
            rho = k * t * np.exp(-rr * t) * nd2
        else:
            nmd1, nmd2 = ndtr(-d1), ndtr(-d2)
            delta = -eqt * nmd1
            theta = decay + rr * dk * nmd2 - qq * ds * nmd1
            rho = -k * t * np.exp(-rr * t) * nmd2
        values = {"delta": delta, "gamma": gamma, "vega": vega,
                  "theta": theta, "rho": rho}
    result: dict[str, Any] = {}
    for name, value in values.items():
        arr = np.asarray(value, dtype=np.float64) + 0.0
        _require_finite(name, arr)
        result[name] = float(arr) if scalar else arr
    result["units"] = dict(GREEK_UNITS)
    return result


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       q: float, kind: OptionKind, *,
                       bracket: tuple[float, float] = (1e-6, 5.0)) -> float:
    """Black–Scholes–Merton implied volatility (year^-1/2) by Brent's method.

    Solves ``bs_price(S, K, T, r, q, sigma, kind) == price`` for ``sigma`` in
    ``bracket`` with ``scipy.optimize.brentq`` (scalar ``K`` only).

    Checks, in order (each raises ``ValueError`` with a distinct message):

    1. invalid inputs, ``T == 0`` (price does not depend on sigma);
    2. no-arbitrage bounds, which must hold strictly:
       call ``max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}``,
       put ``max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}``.
       Equality at the lower bound corresponds to ``sigma = 0`` and at the
       upper bound to ``sigma -> inf``; neither is a finite positive implied
       volatility, so both are rejected;
    3. a price inside the bounds but outside ``[bs(lo), bs(hi)]`` is not
       reachable within the bracket.

    Tolerances: ``xtol = 1e-12``, ``rtol = 4*eps`` (SciPy's minimum),
    ``maxiter = 200`` and convergence is checked. The root is then located to
    about 1e-12 in sigma, so the round-trip error ``sigma -> price -> sigma``
    is dominated by the conditioning of the problem: a price rounding error
    of order ``eps * max(S, K)`` moves sigma by about that amount divided by
    vega. Where vega is tiny (deep in/out of the money, very short ``T``) the
    recovered sigma reproduces the price but may differ from the original
    sigma by more than 1e-8.
    """
    s, k, scalar, t, rr, qq, _, kd = _validate(S, K, T, r, q, 0.0, kind)
    if not scalar:
        raise ValueError("implied_volatility accepts a scalar K only")
    p = _real("price", price)
    if t == 0.0:
        raise ValueError("implied volatility is undefined for T == 0")
    if (not isinstance(bracket, tuple) or len(bracket) != 2):
        raise ValueError("bracket must be a tuple (lo, hi)")
    lo = _real("bracket[0]", bracket[0], positive=True)
    hi = _real("bracket[1]", bracket[1], positive=True)
    if not lo < hi:
        raise ValueError("bracket must satisfy 0 < lo < hi")

    kk = float(k)
    terms = _Terms(s, k, t, rr, qq, 0.0)
    ds, dk = terms.disc_s, float(terms.disc_k)
    if kd == "call":
        lower, upper = max(ds - dk, 0.0), ds
    else:
        lower, upper = max(dk - ds, 0.0), dk
    if not lower < p < upper:
        raise ValueError(
            f"price {p!r} violates the no-arbitrage bounds ({lower!r}, {upper!r}) "
            f"for a {kd}; equality corresponds to sigma = 0 or sigma -> inf")

    def objective(sig: float) -> float:
        return float(_price_array(s, k, t, rr, qq, sig, kd)) - p

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError(
            f"price {p!r} is within the no-arbitrage bounds but not reachable "
            f"within the bracket [{lo!r}, {hi!r}] (prices "
            f"[{f_lo + p!r}, {f_hi + p!r}]); widen the bracket")
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    root, info = brentq(objective, lo, hi, xtol=1e-12,
                        rtol=4.0 * np.finfo(float).eps, maxiter=200,
                        full_output=True)
    if not info.converged:
        raise ValueError(f"brentq did not converge for K={kk!r}: {info.flag}")
    return float(root)


def pricing_metadata(r: float, q: float, sigma: float,
                     sigma_source: SigmaSource) -> dict[str, Any]:
    """Metadata every pricing result must carry (plan §2.1).

    ``measure`` is always ``"risk_neutral"`` (drift ``r - q``); ``sigma_source``
    must be ``"historical_proxy"`` (historical volatility used as a proxy for
    implied volatility) or ``"user_supplied"``.
    """
    if not isinstance(sigma_source, str) or sigma_source not in _SIGMA_SOURCES:
        raise ValueError("sigma_source must be 'historical_proxy' or 'user_supplied'")
    return {
        "measure": "risk_neutral",
        "model": "black_scholes_merton",
        "dividends": "continuous_yield",
        "r": _real("r", r),
        "q": _real("q", q),
        "sigma": _real("sigma", sigma, nonnegative=True),
        "sigma_source": sigma_source,
    }
