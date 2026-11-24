"""Analytical benchmarks for the Heston (1993) stochastic-volatility model.

Model (constant parameters)::

    dS = m S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     d<W1, W2> = rho dt

``m`` is the price drift: ``mu`` (physical measure P, year^-1) for moments, or
``r - q`` (risk-neutral measure Q) for prices. Units: price in the asset's
currency, time in years, ``v``, ``v0`` and ``theta`` in year^-1 (instantaneous
variance), ``kappa`` in year^-1, ``xi`` (vol of variance) in year^-1,
``rho`` dimensionless, ``r``/``q`` continuously compounded in year^-1.

Contents
--------
* CIR variance moments (``heston_variance_mean``, ``heston_variance_var``)::

      E[v_t]   = theta + (v0 - theta) e^{-kappa t}
      Var[v_t] = v0 xi^2 e^{-kappa t}(1 - e^{-kappa t})/kappa
                 + theta xi^2 (1 - e^{-kappa t})^2 / (2 kappa)

  (Cox, Ingersoll and Ross 1985, Econometrica 53(2):385-407, conditional
  moments of the square-root process.) ``1 - e^{-kappa t}`` is evaluated as
  ``-expm1(-kappa t)``, so small ``kappa t`` loses no precision.
* ``E[S_T] = S0 e^{m T}`` (``heston_expected_terminal``): ``S e^{-m t}`` is a
  true martingale for the Heston model (``sqrt(v)`` has finite exponential
  moments of the integrated variance). Simulation schemes match it only when
  they are discrete martingales (QE with martingale correction, log-Euler
  full truncation).
* Feller condition ``2 kappa theta >= xi^2`` (``feller_condition``): reported,
  never required. When violated, zero is accessible for ``v``.
* Characteristic function of ``ln S_T`` under Q in the "little Heston trap"
  form (Albrecher, Mayer, Schoutens and Tistaert 2007, "The little Heston
  trap", Wilmott Magazine, January 2007, pp. 83-92)::

      b = kappa - rho xi i u
      d = sqrt(b^2 + xi^2 (i u + u^2))        (principal root, Re d >= 0)
      g = (b - d)/(b + d),   E = exp(-d T)
      C = (kappa theta / xi^2) [ (b - d) T - 2 ln((1 - g E)/(1 - g)) ]
      D = ((b - d)/xi^2) (1 - E)/(1 - g E)
      phi(u) = exp(i u (ln S0 + (r - q) T) + C + D v0)

  Stable evaluation: ``b - d = -xi^2 (i u + u^2)/(b + d)`` (no cancellation;
  ``Re(b + d) >= kappa > 0``) and ``ln((1 - gE)/(1 - g)) =
  log1p(g (1 - E)/(1 - g))`` (one complex log of the quotient, on its
  principal branch; ``|g|`` may exceed 1). At ``i u + u^2 = 0``
  (``u = 0`` or ``u = -i``) ``b - d = 0`` is set exactly, which also removes
  the ``0/0`` of ``u = -i`` when ``kappa < rho xi``.
* European prices by the Lewis (2001) formula centred on the forward
  (Lewis, A. L., "A simple option formula for general jump-diffusion and
  other exponential Levy processes", 2001; also Lewis, "Option Valuation
  under Stochastic Volatility", 2000)::

      F = S e^{(r - q) T},   k = ln(K / F),   psi = CF of ln(S_T / F)
      Call = e^{-rT} [ F - (sqrt(F K)/pi) Int_0^inf Re(e^{-i u k} psi(u - i/2))/(u^2 + 1/4) du ]
      Put  = Call - e^{-rT} (F - K)                      (put-call parity)

* Independent check by Gil-Pelaez inversion in the probabilities of Heston
  (1993), "A closed-form solution for options with stochastic volatility with
  applications to bond and currency options", Review of Financial Studies
  6(2):327-343 (``heston_price_gil_pelaez``)::

      P2 = 1/2 + (1/pi) Int_0^inf Re[e^{-i u k} psi(u)/(i u)] du
      P1 = 1/2 + (1/pi) Int_0^inf Re[e^{-i u k} psi(u - i)/(i u psi(-i))] du
      Call = e^{-rT} (F P1 - K P2),   psi(-i) = E[S_T/F] = 1.

All formulas were derived/transcribed from memory and **checked numerically**
(Lewis vs Gil-Pelaez, two quadrature levels, Black-Scholes limit, Andersen
2008 benchmark cases); the equation numbers and pages of the sources were
**not verified** in the originals (sources inaccessible in this session).

Quadrature (``_integrate``)
---------------------------
Composite Gauss-Legendre with ``nodes`` points per panel on geometric panels
``[0, 1/4], [1/4, 1/2], [1/2, 1], ..., [U/2, U]``. Each panel is accepted
when the rule on its two halves agrees with the whole-panel rule to
``max(tol, 1e-14 |I|)``; otherwise it is bisected recursively (at most 30
levels, then ``ValueError``). This catches wide panels that hold too many
oscillations of ``e^{-iuk} psi(u)`` (short ``T`` away from the money with
large ``xi`` and ``|rho|`` near 1, where a fixed 32-point rule per geometric
panel was measured to err by ~1e-5). After each panel (and at
least three panels) integration stops when the integrand envelope (modulus
without the unit-modulus oscillation ``e^{-iuk}``) times ``U`` is below
``tol``; if ``U`` would exceed ``U_MAX = 1e6`` a ``ValueError`` is raised
(the characteristic function decays too slowly, e.g. extremely short ``T``
with tiny variance). A uniform rule on a fixed ``[0, U]`` fails for short
maturities or large ``xi``; the geometric panels resolve both the slow
decay and the oscillation scale. With the defaults (``nodes=32``,
``tol=1e-12``) the documented absolute accuracy is about ``1e-7`` for
``S0 = 100`` (measured: agreement with ``nodes=64, tol=1e-14`` and with the
Gil-Pelaez formula to ``<= 1e-8`` for ``T`` from 1/252 to 15 years,
including ``kappa = 5, xi = 3, rho = -0.99``).

Degenerate cases
----------------
* ``xi == 0``: the variance is deterministic,
  ``v_t = theta + (v0 - theta) e^{-kappa t}``, and the price is exactly
  Black-Scholes with ``sigma^2 T = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa``
  (``pricing.black_scholes.bs_price``). The CF has the matching Gaussian form.
* ``0 < xi < XI_MIN = 1e-4``: ``heston_price`` and ``heston_price_gil_pelaez``
  raise ``ValueError``. The ``kappa theta / xi^2`` scaling makes the price
  ill conditioned there (measured by the researcher: errors 1e-4 at
  ``xi = 1e-6`` and 0.6 at ``xi = 1e-8``); the exact limit is ``xi = 0``.
  Near the limit the price differs from Black-Scholes by ``O(xi)``.
* ``T == 0``: discounted intrinsic value ``max(+/-(S - K), 0)`` (same
  convention as ``bs_price``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

OptionKind = Literal["call", "put"]
ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
FloatArray = npt.NDArray[np.float64]

#: Smallest positive ``xi`` accepted by the pricers (see module docstring).
XI_MIN = 1e-4
#: Upper integration limit; reaching it raises ``ValueError``.
U_MAX = 1e6
#: Default absolute stopping tolerance of the integrand envelope.
DEFAULT_TOL = 1e-12
#: Default Gauss-Legendre nodes per panel.
DEFAULT_NODES = 32
#: Documented absolute price accuracy for S0 = 100 with the defaults.
PRICE_ABS_TOL_AT_100 = 1e-7
#: Negative prices with ``|price| <= NEGATIVE_CLAMP_REL * e^{-rT} max(F, K)``
#: (1e-6 for F, K ~ 100, i.e. 10x the documented accuracy) are quadrature
#: noise and clamped to +0.0; larger negatives raise ``ValueError``.
NEGATIVE_CLAMP_REL = 1e-8
_MIN_PANELS = 3
_MAX_DEPTH = 30
_FIRST_PANEL = 0.25


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number (bool is rejected)")
    try:
        out = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be representable as a finite float") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive(name: str, value: Any) -> float:
    out = _real(name, value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative(name: str, value: Any) -> float:
    out = _real(name, value)
    if out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _variance_params(v0: Any, kappa: Any, theta: Any,
                     xi: Any = 0.0) -> tuple[float, float, float, float]:
    return (_nonnegative("v0", v0), _positive("kappa", kappa),
            _positive("theta", theta), _nonnegative("xi", xi))


def _rho(rho: Any) -> float:
    out = _real("rho", rho)
    if not -1.0 <= out <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    return out


def _times(t: Any) -> tuple[FloatArray, bool]:
    """Nonnegative finite times as a float64 array; flag True for scalars."""
    if isinstance(t, (bool, np.bool_)):
        raise ValueError("t must be real (bool is rejected)")
    if isinstance(t, Real):
        return np.asarray([_nonnegative("t", t)], dtype=np.float64), True
    arr = np.asarray(t)
    if arr.dtype == np.bool_ or arr.dtype.kind not in "iuf":
        raise ValueError("t must be a real number or an array of real numbers")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("t must be finite and nonnegative")
    return arr, False


def _kind(kind: Any) -> OptionKind:
    if not isinstance(kind, str) or kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return "call" if kind == "call" else "put"


# ---------------------------------------------------------------------------
# Moments and Feller
# ---------------------------------------------------------------------------

def heston_variance_mean(v0: float, kappa: float, theta: float, t: Any) -> Any:
    """``E[v_t] = theta + (v0 - theta) e^{-kappa t}`` (year^-1).

    CIR conditional mean (Cox, Ingersoll and Ross 1985). ``t`` (years,
    ``>= 0``) may be a scalar (returns ``float``) or an array (returns a
    float64 array of the same shape).
    """
    v0_f, k, th, _ = _variance_params(v0, kappa, theta)
    times, scalar = _times(t)
    with np.errstate(under="ignore"):
        out = th + (v0_f - th) * np.exp(-k * times)
    return float(out[0]) if scalar else out


def heston_variance_var(v0: float, kappa: float, theta: float, xi: float,
                        t: Any) -> Any:
    """``Var[v_t]`` of the CIR variance (year^-2).

    ``v0 xi^2 e^{-kt}(1 - e^{-kt})/k + theta xi^2 (1 - e^{-kt})^2/(2k)``
    (Cox, Ingersoll and Ross 1985), with ``1 - e^{-kt} = -expm1(-kt)`` so
    that ``kappa t -> 0`` is accurate. Scalar or array ``t`` as in
    ``heston_variance_mean``. ``xi == 0`` gives 0.
    """
    v0_f, k, th, x = _variance_params(v0, kappa, theta, xi)
    times, scalar = _times(t)
    with np.errstate(under="ignore"):
        e = np.exp(-k * times)
        one_minus = -np.expm1(-k * times)
        ratio = one_minus / k  # (1 - e^{-kt})/k, accurate for small k t
        out = x * x * (v0_f * e * ratio + th * one_minus * ratio / 2.0)
    if not np.all(np.isfinite(out)):
        raise ValueError("variance of v_t exceeds finite precision")
    return float(out[0]) if scalar else out


def heston_expected_terminal(s0: float, mu: float, T: float) -> float:
    """``E[S_T] = S0 e^{mu T}`` (``mu`` = price drift; ``r - q`` under Q)."""
    s = _positive("s0", s0)
    m = _real("mu", mu)
    t = _nonnegative("T", T)
    log_value = math.log(s) + m * t
    if log_value > math.log(np.finfo(np.float64).max):
        raise ValueError("E[S_T] overflows float64")
    return math.exp(log_value)


def feller_condition(kappa: float, theta: float, xi: float) -> dict[str, Any]:
    """Feller condition ``2 kappa theta >= xi^2`` (reported, not required).

    Returns ``{"satisfied", "two_kappa_theta", "xi_squared", "margin"}`` with
    ``margin = 2 kappa theta - xi^2`` (year^-2); ``satisfied`` means zero is
    not reached by the exact CIR process.
    """
    k = _positive("kappa", kappa)
    th = _positive("theta", theta)
    x = _nonnegative("xi", xi)
    two_kt = 2.0 * k * th
    xi2 = x * x
    if not (math.isfinite(two_kt) and math.isfinite(xi2)):
        raise ValueError("Feller terms exceed finite precision")
    return {"satisfied": bool(two_kt >= xi2), "two_kappa_theta": two_kt,
            "xi_squared": xi2, "margin": two_kt - xi2}


def heston_integrated_variance(v0: float, kappa: float, theta: float,
                               T: float) -> float:
    """Deterministic (``xi = 0``) integrated variance ``Int_0^T v_t dt``.

    ``theta T + (v0 - theta)(1 - e^{-kappa T})/kappa`` (dimensionless).
    """
    v0_f, k, th, _ = _variance_params(v0, kappa, theta)
    t = _nonnegative("T", T)
    return th * t + (v0_f - th) * (-math.expm1(-k * t)) / k


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def _cf_core(u: ComplexArray, T: float, v0: float, kappa: float, theta: float,
             xi: float, rho: float) -> ComplexArray:
    """``E[exp(i u X)]`` for ``X = ln(S_T/F)`` (risk-neutral, forward-centred).

    Inputs are validated floats; ``u`` complex. Returns complex128.
    """
    iu = 1j * u
    w = iu + u * u  # i u + u^2
    if xi == 0.0:
        iv = theta * T + (v0 - theta) * (-math.expm1(-kappa * T)) / kappa
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.exp(-0.5 * w * iv)
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        b = kappa - rho * xi * iu
        d = np.sqrt(b * b + (xi * xi) * w)  # principal root, Re d >= 0
        bpd = b + d
        zero = w == 0
        safe_bpd = np.where(zero, 1.0, bpd)
        bmd_over_xi2 = np.where(zero, 0.0, -w / safe_bpd)  # (b - d)/xi^2
        bmd = (xi * xi) * bmd_over_xi2
        g = bmd / safe_bpd
        one_minus_e = -np.expm1(-d * T)  # 1 - E
        e = 1.0 - one_minus_e
        log_q = np.log1p(g * one_minus_e / (1.0 - g))  # ln((1 - gE)/(1 - g))
        c = (kappa * theta) * (bmd_over_xi2 * T - 2.0 * log_q / (xi * xi))
        dd = bmd_over_xi2 * one_minus_e / (1.0 - g * e)
        out = np.exp(c + dd * v0)
    return np.where(zero, 1.0 + 0.0j, out)


def heston_characteristic_function(u: Any, T: float, v0: float, kappa: float,
                                   theta: float, xi: float, rho: float, *,
                                   s0: float = 1.0, r: float = 0.0,
                                   q: float = 0.0) -> Any:
    """Risk-neutral CF ``phi(u) = E^Q[exp(i u ln S_T)]``, "little Heston trap".

    ``u`` may be a real or complex scalar or array (vectorized); the result
    is complex (scalar or complex128 array of the same shape). See the module
    docstring for the formula (Albrecher et al. 2007). ``xi == 0`` uses the
    exact Gaussian CF with the deterministic integrated variance. Values of
    ``u`` with ``Im u`` outside the strip where ``E[S_T^{-Im u}]`` is finite
    may overflow; non-finite results raise ``ValueError``.
    """
    t = _positive("T", T)
    v0_f, k, th, x = _variance_params(v0, kappa, theta, xi)
    rh = _rho(rho)
    s = _positive("s0", s0)
    rr = _real("r", r)
    qq = _real("q", q)
    arr = np.asarray(u)
    if arr.dtype == np.bool_ or arr.dtype.kind not in "iufc":
        raise ValueError("u must be numeric")
    uc = arr.astype(np.complex128)
    if not np.all(np.isfinite(uc)):
        raise ValueError("u must be finite")
    log_f = math.log(s) + (rr - qq) * t
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(1j * uc * log_f) * _cf_core(uc.reshape(-1), t, v0_f, k, th, x,
                                                 rh).reshape(uc.shape)
    if not np.all(np.isfinite(out)):
        raise ValueError("characteristic function is not finite for these inputs")
    return complex(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def _integrate(integrand: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
               nodes: int, tol: float) -> float:
    """Int_0^inf f(u) du on geometric Gauss-Legendre panels (see module doc).

    ``integrand(u)`` returns ``(f(u), envelope(u))`` where ``envelope`` is a
    nonnegative bound of ``|f|`` without the unit-modulus oscillation.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)

    def rule(lo: float, hi: float) -> float:
        half = 0.5 * (hi - lo)
        values, _ = integrand(lo + half * (x + 1.0))
        if not np.all(np.isfinite(values)):
            raise ValueError("pricing integrand is not finite")
        return half * float(np.dot(w, values))

    def panel(lo: float, hi: float, whole: float, depth: int) -> float:
        # Bisection check: a wide panel may hold more oscillations of
        # e^{-iuk} psi(u) than ``nodes`` points resolve (short T, far from
        # the money, large xi with |rho| near 1).
        mid = 0.5 * (lo + hi)
        left, right = rule(lo, mid), rule(mid, hi)
        halves = left + right
        if abs(halves - whole) <= max(tol, 1e-14 * abs(halves)):
            return halves
        if depth >= _MAX_DEPTH:
            raise ValueError("characteristic-function integral did not converge "
                             f"(panel [{lo:g}, {hi:g}] needs more than "
                             f"{_MAX_DEPTH} bisections)")
        return (panel(lo, mid, left, depth + 1)
                + panel(mid, hi, right, depth + 1))

    total = 0.0
    lo, hi = 0.0, _FIRST_PANEL
    panels = 0
    while True:
        total += panel(lo, hi, rule(lo, hi), 0)
        panels += 1
        _, env = integrand(np.asarray([hi], dtype=np.float64))
        if panels >= _MIN_PANELS and float(env[0]) * hi < tol:
            return total
        lo, hi = hi, 2.0 * hi
        if hi > U_MAX:
            raise ValueError(
                f"characteristic-function integral did not converge before "
                f"U_MAX = {U_MAX:g} (decay too slow: maturity or variance too "
                "small for the Fourier pricer)")


def _quad_params(nodes: Any, tol: Any) -> tuple[int, float]:
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)) \
            or nodes < 4:
        raise ValueError("nodes must be an integer >= 4")
    t = _positive("tol", tol)
    return int(nodes), t


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class _Inputs:
    def __init__(self, S: Any, K: Any, T: Any, r: Any, q: Any, v0: Any,
                 kappa: Any, theta: Any, xi: Any, rho: Any, kind: Any) -> None:
        self.S = _positive("S", S)
        self.K = _positive("K", K)
        self.T = _nonnegative("T", T)
        self.r = _real("r", r)
        self.q = _real("q", q)
        self.v0, self.kappa, self.theta, self.xi = _variance_params(v0, kappa, theta, xi)
        self.rho = _rho(rho)
        self.kind = _kind(kind)
        if 0.0 < self.xi < XI_MIN:
            raise ValueError(
                f"0 < xi < {XI_MIN:g} is ill conditioned for the Fourier pricer "
                "(kappa*theta/xi^2 scaling); use xi = 0 (exact Black-Scholes "
                f"limit) or xi >= {XI_MIN:g}")
        with np.errstate(over="ignore", under="ignore"):
            self.disc = math.exp(-self.r * self.T)
            log_f = math.log(self.S) + (self.r - self.q) * self.T
        if not math.isfinite(self.disc) or self.disc == 0.0 or abs(log_f) > 700:
            raise ValueError("discount factor or forward exceeds finite precision")
        self.F = math.exp(log_f)
        self.k = math.log(self.K) - log_f  # ln(K/F)

    def special(self) -> float | None:
        """Price for ``T == 0`` or ``xi == 0`` (exact), else None."""
        if self.T == 0.0 or self.xi == 0.0:
            from pricing.black_scholes import bs_price
            sigma = 0.0
            if self.T > 0.0:
                iv = heston_integrated_variance(self.v0, self.kappa, self.theta, self.T)
                sigma = math.sqrt(max(iv, 0.0) / self.T)
            return float(bs_price(self.S, self.K, self.T, self.r, self.q, sigma, self.kind))
        return None

    def psi(self, u: ComplexArray) -> ComplexArray:
        return _cf_core(u, self.T, self.v0, self.kappa, self.theta, self.xi, self.rho)

    def finish(self, call: float) -> float:
        """Apply put-call parity and the nonnegativity clamp."""
        value = call if self.kind == "call" else call - self.disc * (self.F - self.K)
        if not math.isfinite(value):
            raise ValueError("Heston price is not finite")
        if value < 0.0:
            limit = NEGATIVE_CLAMP_REL * self.disc * max(self.F, self.K)
            if value < -limit:
                raise ValueError(
                    f"Heston price {value!r} is negative beyond the quadrature "
                    f"tolerance {limit:g}; parameters outside the reliable range")
            value = 0.0
        return value + 0.0  # never -0.0


def heston_price(S: float, K: float, T: float, r: float, q: float, v0: float,
                 kappa: float, theta: float, xi: float, rho: float,
                 kind: OptionKind, *, nodes: int = DEFAULT_NODES,
                 tol: float = DEFAULT_TOL) -> float:
    """European Heston price (risk-neutral), Lewis formula centred on ``F``.

    Parameters: spot ``S`` and strike ``K`` (> 0, currency); ``T`` years
    (``>= 0``); ``r``, ``q`` year^-1; ``v0 >= 0``, ``theta > 0`` (year^-1),
    ``kappa > 0`` (year^-1), ``xi >= 0`` (year^-1; ``0 < xi < 1e-4``
    rejected), ``rho`` in ``[-1, 1]``; ``kind`` ``"call"``/``"put"``.
    ``nodes``/``tol`` control the quadrature (module docstring); use two
    levels to check convergence. The call is integrated; the put follows by
    parity. Accuracy with the defaults: about ``1e-7`` absolute at
    ``S0 = 100`` (``PRICE_ABS_TOL_AT_100``). The result is ``>= 0`` and
    never ``-0.0``: negatives within ``NEGATIVE_CLAMP_REL * e^{-rT} max(F, K)``
    are clamped, larger ones raise ``ValueError``.
    """
    inp = _Inputs(S, K, T, r, q, v0, kappa, theta, xi, rho, kind)
    m, t = _quad_params(nodes, tol)
    special = inp.special()
    if special is not None:
        return special
    k = inp.k

    def integrand(u: FloatArray) -> tuple[FloatArray, FloatArray]:
        z = inp.psi(u - 0.5j)
        denom = u * u + 0.25
        values = np.real(np.exp(-1j * u * k) * z) / denom
        return values, np.abs(z) / denom

    integral = _integrate(integrand, m, t)
    call = inp.disc * (inp.F - math.sqrt(inp.F * inp.K) / math.pi * integral)
    return inp.finish(call)


def heston_price_gil_pelaez(S: float, K: float, T: float, r: float, q: float,
                            v0: float, kappa: float, theta: float, xi: float,
                            rho: float, kind: OptionKind, *,
                            nodes: int = DEFAULT_NODES,
                            tol: float = DEFAULT_TOL) -> float:
    """Independent check: Heston (1993) probabilities ``P1``, ``P2``.

    Same contract and quadrature as ``heston_price`` but a different
    integrand (Gil-Pelaez inversion, module docstring). The integrands are
    finite at ``u -> 0`` but the Gauss nodes never touch 0. Agreement with
    ``heston_price`` is a check of the CF and of the quadrature.
    """
    inp = _Inputs(S, K, T, r, q, v0, kappa, theta, xi, rho, kind)
    m, t = _quad_params(nodes, tol)
    special = inp.special()
    if special is not None:
        return special
    k = inp.k

    def p2_integrand(u: FloatArray) -> tuple[FloatArray, FloatArray]:
        z = inp.psi(u.astype(np.complex128))
        values = np.real(np.exp(-1j * u * k) * z / (1j * u))
        return values, np.abs(z) / u

    def p1_integrand(u: FloatArray) -> tuple[FloatArray, FloatArray]:
        z = inp.psi(u - 1j)  # psi(-i) = E[S_T/F] = 1
        values = np.real(np.exp(-1j * u * k) * z / (1j * u))
        return values, np.abs(z) / u

    p1 = 0.5 + _integrate(p1_integrand, m, t) / math.pi
    p2 = 0.5 + _integrate(p2_integrand, m, t) / math.pi
    call = inp.disc * (inp.F * p1 - inp.K * p2)
    return inp.finish(call)
