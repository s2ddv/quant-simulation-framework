"""Tests for the Heston (1993) analytical benchmarks ``validation.heston_theory`` (E5.1).

Model under the risk-neutral measure Q (prices) or P (moments)::

    dS = m S dt + sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,
    d<W1, W2> = rho dt,   m = r - q (Q) or mu (P).

Units: price in the asset's currency, time in years, ``v``, ``v0``, ``theta``,
``kappa``, ``xi``, ``r``, ``q`` in year^-1, ``rho`` dimensionless.

Independent references written here (not copied from the module under test):

* Exact CIR law (Cox, Ingersoll and Ross 1985): ``v_t = c X`` with ``X`` a
  noncentral chi-square with ``delta = 4 kappa theta / xi^2`` degrees of
  freedom and noncentrality ``lambda = v0 e^{-kappa t} / c``,
  ``c = xi^2 (1 - e^{-kappa t}) / (4 kappa)``. Its closed moments
  ``E = c (delta + lambda)``, ``Var = 2 c^2 (delta + 2 lambda)`` are compared
  algebraically, and exact draws (``numpy.random.Generator.noncentral_chisquare``)
  statistically at 4 SE.
* Characteristic function of ``ln S_T`` written twice in this file: the
  "little Heston trap" form (Albrecher et al. 2007) with a plain complex
  ``log`` of the quotient (no ``log1p`` rewriting, no ``b - d`` rewriting), and
  the original Heston (1993) form with ``e^{+dT}``, which has a branch-cut
  discontinuity for long maturities and is therefore only used at short ``T``.
* Prices by Gil-Pelaez inversion of ``P1``, ``P2`` integrated with
  ``scipy.integrate.quad`` (adaptive QUADPACK, a different rule from the
  geometric Gauss-Legendre panels of the module) on ``[0, U]``, with ``U``
  chosen where ``|phi(U)| / U < 1e-16``.
* Black-Scholes written here with ``scipy.special.ndtr`` (``xi = 0`` branch)
  and ``pricing.black_scholes.bs_price`` (``xi -> 0`` limit, as requested).
* Andersen (2008) benchmark cases I-III (S0 = 100, r = q = 0, v0 = theta):
  the parameters and the reference prices (7 decimals) are those of the
  researcher's opinion (``parecer_e5``), which transcribed Andersen's table 1
  from memory and recomputed the prices (Lewis = Gil-Pelaez to 1e-8); only
  case I, K = 100 (13.0847) was matched against the published value. They are
  therefore treated as computed references, not as a verified transcription.

Tolerances: deterministic comparisons use the module's documented absolute
price accuracy ``1e-7`` at S0 = 100 (table: ``1e-6``, i.e. the table's
rounding ``5e-8`` plus that accuracy with margin); the statistical CIR check
uses ``|estimate - theory| <= 4 SE`` (plan section 2.5).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr

from pricing.black_scholes import bs_price
from validation import heston_theory as ht
from validation.heston_theory import OptionKind

K_SE = 4.0
PRICE_TOL = 1e-7          # module's documented accuracy at S0 = 100
TABLE_TOL = 1e-6          # 7-decimal table: rounding 5e-8 + PRICE_TOL, margin
S0 = 100.0
STRIKES = (70.0, 100.0, 140.0)

# Andersen (2008) cases: (kappa, theta = v0, xi, rho, T).
CASES: dict[str, tuple[float, float, float, float, float]] = {
    "I": (0.5, 0.04, 1.0, -0.9, 10.0),
    "II": (0.3, 0.04, 0.9, -0.5, 15.0),
    "III": (1.0, 0.09, 1.0, -0.3, 5.0),
}
TABLE: dict[str, tuple[float, float, float]] = {
    "I": (35.8497697, 13.0846701, 0.2957744),
    "II": (37.1696647, 16.6492229, 5.1381905),
    "III": (38.7720441, 21.7952877, 9.9830678),
}


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------
def cf_trap_ref(u: Any, S: float, T: float, r: float, q: float, v0: float,
                kappa: float, theta: float, xi: float, rho: float) -> np.ndarray:
    """``E^Q[exp(i u ln S_T)]``, Albrecher et al. (2007) form, plain formulas."""
    uc = np.asarray(u, dtype=complex)
    iu = 1j * uc
    b = kappa - rho * xi * iu
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + uc * uc))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    c = (r - q) * iu * T + kappa * theta / xi ** 2 * (
        (b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    dd = (b - d) / xi ** 2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * math.log(S) + c + dd * v0)


def cf_heston93_ref(u: Any, S: float, T: float, r: float, q: float, v0: float,
                    kappa: float, theta: float, xi: float, rho: float) -> np.ndarray:
    """Original Heston (1993) form (``f_2``, ``e^{+dT}``); branch cut at long T."""
    uc = np.asarray(u, dtype=complex)
    iu = 1j * uc
    d = np.sqrt((rho * xi * iu - kappa) ** 2 - xi ** 2 * (-iu - uc * uc))
    bp = kappa - rho * xi * iu + d
    g = bp / (kappa - rho * xi * iu - d)
    e = np.exp(d * T)
    c = (r - q) * iu * T + kappa * theta / xi ** 2 * (
        bp * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    dd = bp / xi ** 2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * math.log(S) + c + dd * v0)


def price_quad_ref(S: float, K: float, T: float, r: float, q: float, v0: float,
                   kappa: float, theta: float, xi: float, rho: float,
                   kind: str = "call") -> float:
    """Gil-Pelaez ``P1``, ``P2`` with ``scipy.integrate.quad`` and the trap CF."""
    args = (S, T, r, q, v0, kappa, theta, xi, rho)
    fwd = S * math.exp((r - q) * T)
    lk = math.log(K)
    upper = 1.0
    while abs(complex(cf_trap_ref(upper, *args))) / upper > 1e-16:
        upper *= 1.25

    def f2(u: float) -> float:
        return float((np.exp(-1j * u * lk) * cf_trap_ref(u, *args) / (1j * u)).real)

    def f1(u: float) -> float:
        return float((np.exp(-1j * u * lk) * cf_trap_ref(u - 1j, *args)
                      / (1j * u * fwd)).real)

    opts: dict[str, Any] = dict(limit=4000, epsabs=1e-13, epsrel=1e-13)
    p1 = 0.5 + integrate.quad(f1, 0.0, upper, **opts)[0] / math.pi
    p2 = 0.5 + integrate.quad(f2, 0.0, upper, **opts)[0] / math.pi
    disc = math.exp(-r * T)
    call = disc * (fwd * p1 - K * p2)
    return call if kind == "call" else call - disc * (fwd - K)


def bs_ref(S: float, K: float, T: float, r: float, q: float, total_var: float,
           kind: str) -> float:
    """Black-Scholes-Merton with total variance ``sigma^2 T`` (ndtr)."""
    sd = math.sqrt(total_var)
    d1 = (math.log(S / K) + (r - q) * T + 0.5 * total_var) / sd
    d2 = d1 - sd
    call = S * math.exp(-q * T) * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)
    return call if kind == "call" else call - S * math.exp(-q * T) + K * math.exp(-r * T)


def _case_args(case: str) -> tuple[float, float, float, float, float, float]:
    kappa, theta, xi, rho, T = CASES[case]
    return T, theta, kappa, theta, xi, rho  # T, v0, kappa, theta, xi, rho


# ---------------------------------------------------------------------------
# CIR moments
# ---------------------------------------------------------------------------
CIR_PARAMS = [
    dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.0),    # case I: Feller violated
    dict(v0=0.09, kappa=2.0, theta=0.04, xi=0.3),    # Feller holds, v0 != theta
    dict(v0=0.0, kappa=1.0, theta=0.09, xi=1.0),     # v0 = 0 (lambda = 0)
]


def _ncx2_params(v0: float, kappa: float, theta: float, xi: float,
                 t: float) -> tuple[float, float, float]:
    c = xi * xi * (-math.expm1(-kappa * t)) / (4.0 * kappa)
    return c, 4.0 * kappa * theta / (xi * xi), v0 * math.exp(-kappa * t) / c


@pytest.mark.parametrize("p", CIR_PARAMS)
@pytest.mark.parametrize("t", [0.01, 0.5, 2.0, 10.0])
def test_cir_moments_match_noncentral_chisquare_closed_form(p: dict[str, float],
                                                            t: float) -> None:
    c, delta, lam = _ncx2_params(p["v0"], p["kappa"], p["theta"], p["xi"], t)
    mean = ht.heston_variance_mean(p["v0"], p["kappa"], p["theta"], t)
    var = ht.heston_variance_var(p["v0"], p["kappa"], p["theta"], p["xi"], t)
    assert mean == pytest.approx(c * (delta + lam), rel=1e-12)
    assert var == pytest.approx(2.0 * c * c * (delta + 2.0 * lam), rel=1e-12)


@pytest.mark.parametrize("idx", range(len(CIR_PARAMS)))
def test_cir_moments_vs_exact_sampling(idx: int) -> None:
    # N = 200_000 exact draws per time: the SE of the mean is std/sqrt(N);
    # the SE of the sample variance sqrt((m4 - s^4)/N) uses the sample fourth
    # central moment (all CIR moments are finite). 4 SE per assertion.
    p = CIR_PARAMS[idx]
    rng = np.random.default_rng(np.random.SeedSequence(5101 + idx))
    n = 200_000
    for t in (0.25, 2.0):
        c, delta, lam = _ncx2_params(p["v0"], p["kappa"], p["theta"], p["xi"], t)
        draws = c * rng.noncentral_chisquare(delta, lam, n) if lam > 0 \
            else c * rng.chisquare(delta, n)
        m = float(draws.mean())
        se_m = float(draws.std(ddof=1)) / math.sqrt(n)
        cen = draws - m
        s2 = float(np.mean(cen ** 2))
        se_v = math.sqrt((float(np.mean(cen ** 4)) - s2 * s2) / n)
        mean = ht.heston_variance_mean(p["v0"], p["kappa"], p["theta"], t)
        var = ht.heston_variance_var(p["v0"], p["kappa"], p["theta"], p["xi"], t)
        assert abs(m - mean) <= K_SE * se_m, (t, m, mean, se_m)
        assert abs(s2 * n / (n - 1) - var) <= K_SE * se_v, (t, s2, var, se_v)


def test_cir_moments_limits_and_arrays() -> None:
    assert ht.heston_variance_mean(0.09, 2.0, 0.04, 0.0) == 0.09
    assert ht.heston_variance_var(0.09, 2.0, 0.04, 0.3, 0.0) == 0.0
    assert ht.heston_variance_var(0.09, 2.0, 0.04, 0.0, 1.0) == 0.0
    # t -> infinity: stationary Gamma law, mean theta, var theta xi^2/(2 kappa).
    assert ht.heston_variance_mean(0.09, 2.0, 0.04, 1e3) == pytest.approx(0.04, rel=1e-14)
    assert ht.heston_variance_var(0.09, 2.0, 0.04, 0.3, 1e3) == pytest.approx(
        0.04 * 0.09 / 4.0, rel=1e-14)
    # Small kappa t: series E ~ v0 + kappa(theta - v0)t; Var ~ v0 xi^2 t.
    t = 1e-9
    assert ht.heston_variance_var(0.09, 2.0, 0.04, 0.3, t) == pytest.approx(
        0.09 * 0.09 * t, rel=1e-8)
    ts = np.array([[0.0, 0.5], [1.0, 3.0]])
    arr = ht.heston_variance_mean(0.09, 2.0, 0.04, ts)
    assert isinstance(arr, np.ndarray) and arr.shape == (2, 2)
    np.testing.assert_allclose(arr, 0.04 + 0.05 * np.exp(-2.0 * ts), rtol=1e-14)
    varr = ht.heston_variance_var(0.09, 2.0, 0.04, 0.3, ts)
    assert varr.shape == (2, 2)
    assert float(varr[1, 0]) == ht.heston_variance_var(0.09, 2.0, 0.04, 0.3, 1.0)
    assert isinstance(ht.heston_variance_mean(0.09, 2.0, 0.04, 1), float)


def test_expected_terminal_and_integrated_variance() -> None:
    assert ht.heston_expected_terminal(100.0, 0.05, 2.0) == pytest.approx(
        100.0 * math.exp(0.1), rel=1e-15)
    # exp(log(S0)) rounds by a few ulp; equality is not part of the contract.
    assert ht.heston_expected_terminal(100.0, 0.05, 0.0) == pytest.approx(100.0, rel=1e-15)
    with pytest.raises(ValueError):
        ht.heston_expected_terminal(100.0, 1e3, 1e3)  # overflows float64
    iv = ht.heston_integrated_variance(0.09, 2.0, 0.04, 3.0)
    # Direct integral of the ODE solution theta + (v0 - theta) e^{-kappa t}.
    ref = integrate.quad(lambda s: 0.04 + 0.05 * math.exp(-2.0 * s), 0.0, 3.0,
                         epsabs=1e-15)[0]
    assert iv == pytest.approx(ref, rel=1e-12)
    assert ht.heston_integrated_variance(0.04, 2.0, 0.04, 3.0) == pytest.approx(0.12)


def test_feller_condition() -> None:
    for case, (kappa, theta, xi, _, _) in CASES.items():
        out = ht.feller_condition(kappa, theta, xi)
        assert out["satisfied"] is False, case       # cases I-III violate Feller
        assert out["margin"] == pytest.approx(2 * kappa * theta - xi * xi)
        assert out["two_kappa_theta"] == pytest.approx(2 * kappa * theta)
        assert out["xi_squared"] == pytest.approx(xi * xi)
    ok = ht.feller_condition(2.0, 0.04, 0.3)
    assert ok["satisfied"] is True and ok["margin"] == pytest.approx(0.16 - 0.09)
    edge = ht.feller_condition(0.5, 1.0, 1.0)        # 2 kappa theta == xi^2
    assert edge["satisfied"] is True and edge["margin"] == 0.0
    assert ht.feller_condition(1.0, 0.04, 0.0)["satisfied"] is True


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------
CF_ARGS = dict(T=0.5, v0=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
               s0=100.0, r=0.02, q=0.01)


def _cf(u: Any, **kw: Any) -> Any:
    args = dict(CF_ARGS)
    args.update(kw)
    return ht.heston_characteristic_function(
        u, args["T"], args["v0"], args["kappa"], args["theta"], args["xi"],
        args["rho"], s0=args["s0"], r=args["r"], q=args["q"])


@pytest.mark.parametrize("kw", [{}, dict(T=10.0, v0=0.04, kappa=0.5, xi=1.0, rho=-0.9),
                                dict(rho=0.9, kappa=0.2, xi=1.5),   # kappa < rho xi
                                dict(xi=0.0)])
def test_cf_identities(kw: dict[str, float]) -> None:
    args = dict(CF_ARGS)
    args.update(kw)
    assert _cf(0.0, **kw) == pytest.approx(1.0 + 0.0j, abs=1e-15)
    fwd = args["s0"] * math.exp((args["r"] - args["q"]) * args["T"])
    # phi(-i) = E[S_T] = F (martingale under Q).
    assert _cf(-1j, **kw) == pytest.approx(fwd + 0.0j, rel=1e-12)
    u = np.linspace(0.05, 30.0, 200)
    np.testing.assert_allclose(_cf(-u, **kw), np.conj(_cf(u, **kw)), rtol=0, atol=1e-14)
    assert np.all(np.abs(_cf(u, **kw)) <= 1.0 + 1e-12)   # |CF| <= 1 for real u


def test_cf_matches_independent_forms() -> None:
    a = CF_ARGS
    args = (a["s0"], a["T"], a["r"], a["q"], a["v0"], a["kappa"], a["theta"],
            a["xi"], a["rho"])
    u = np.concatenate([np.linspace(-40, -0.05, 120), np.linspace(0.05, 40, 120)])
    got = _cf(u)
    # Short T: the original Heston (1993) form has no branch-cut jump here.
    np.testing.assert_allclose(got, cf_heston93_ref(u, *args), rtol=0, atol=1e-13)
    np.testing.assert_allclose(got, cf_trap_ref(u, *args), rtol=0, atol=1e-13)
    # Complex arguments of the pricing contours.
    for shift in (-0.5j, -1j):
        np.testing.assert_allclose(_cf(u + shift), cf_trap_ref(u + shift, *args),
                                   rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case", list(CASES))
def test_cf_long_maturity_matches_trap_and_is_continuous(case: str) -> None:
    T, v0, kappa, theta, xi, rho = _case_args(case)
    u = np.linspace(0.01, 60.0, 6000)
    got = ht.heston_characteristic_function(u, T, v0, kappa, theta, xi, rho, s0=S0)
    np.testing.assert_allclose(got, cf_trap_ref(u, S0, T, 0.0, 0.0, v0, kappa, theta,
                                                xi, rho), rtol=0, atol=1e-12)
    # Continuity in u (no 2*pi*i jump): steps of 0.01 change phi by << 1.
    assert float(np.max(np.abs(np.diff(got)))) < 0.05
    if case == "I":
        # The original form jumps for case I (why the trap form is required).
        u2 = np.linspace(0.1, 30.0, 300)
        old = cf_heston93_ref(u2, S0, T, 0.0, 0.0, v0, kappa, theta, xi, rho)
        new = ht.heston_characteristic_function(u2, T, v0, kappa, theta, xi, rho, s0=S0)
        assert float(np.max(np.abs(old - new))) > 0.1


def test_cf_zero_xi_is_gaussian() -> None:
    a = CF_ARGS
    u = np.linspace(-20, 20, 81)
    iv = ht.heston_integrated_variance(a["v0"], a["kappa"], a["theta"], a["T"])
    mean = math.log(a["s0"]) + (a["r"] - a["q"]) * a["T"] - 0.5 * iv
    ref = np.exp(1j * u * mean - 0.5 * u * u * iv)
    np.testing.assert_allclose(_cf(u, xi=0.0), ref, rtol=0, atol=1e-14)


def test_cf_shape_and_validation() -> None:
    assert isinstance(_cf(1.5), complex)
    assert _cf(np.ones((2, 3))).shape == (2, 3)
    for bad in ("a", np.array([True]), np.inf, np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            _cf(bad)
    with pytest.raises(ValueError):
        _cf(1.0, T=0.0)


# ---------------------------------------------------------------------------
# Prices: Andersen cases I-III
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def quad_prices() -> dict[tuple[str, float], float]:
    out = {}
    for case in CASES:
        T, v0, kappa, theta, xi, rho = _case_args(case)
        for K in STRIKES:
            out[(case, K)] = price_quad_ref(S0, K, T, 0.0, 0.0, v0, kappa, theta, xi, rho)
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_andersen_cases_vs_quad_table_and_gil_pelaez(
        case: str, quad_prices: dict[tuple[str, float], float]) -> None:
    T, v0, kappa, theta, xi, rho = _case_args(case)
    for K, tab in zip(STRIKES, TABLE[case]):
        lewis = ht.heston_price(S0, K, T, 0.0, 0.0, v0, kappa, theta, xi, rho, "call")
        gp = ht.heston_price_gil_pelaez(S0, K, T, 0.0, 0.0, v0, kappa, theta, xi, rho,
                                        "call")
        ref = quad_prices[(case, K)]
        assert abs(lewis - ref) <= PRICE_TOL, (case, K, lewis, ref)
        assert abs(lewis - tab) <= TABLE_TOL, (case, K, lewis, tab)
        assert abs(lewis - gp) <= PRICE_TOL, (case, K, lewis, gp)
        fine = ht.heston_price(S0, K, T, 0.0, 0.0, v0, kappa, theta, xi, rho, "call",
                               nodes=64, tol=1e-14)
        assert abs(lewis - fine) <= PRICE_TOL, (case, K, lewis, fine)


def test_case_i_atm_matches_published_four_decimals() -> None:
    T, v0, kappa, theta, xi, rho = _case_args("I")
    price = ht.heston_price(S0, 100.0, T, 0.0, 0.0, v0, kappa, theta, xi, rho, "call")
    assert round(price, 4) == 13.0847


@pytest.mark.parametrize("r, q", [(0.03, 0.01), (-0.01, 0.04)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_prices_with_rates_vs_quad(r: float, q: float, kind: OptionKind) -> None:
    T, v0, kappa, theta, xi, rho = _case_args("III")
    for K in (60.0, 100.0, 150.0):
        got = ht.heston_price(S0, K, T, r, q, 0.06, kappa, theta, xi, rho, kind)
        ref = price_quad_ref(S0, K, T, r, q, 0.06, kappa, theta, xi, rho, kind)
        assert abs(got - ref) <= PRICE_TOL, (K, got, ref)


@pytest.mark.parametrize("case", list(CASES))
def test_put_call_parity(case: str) -> None:
    T, v0, kappa, theta, xi, rho = _case_args(case)
    r, q = 0.02, 0.01
    fwd = S0 * math.exp((r - q) * T)
    disc = math.exp(-r * T)
    for K in (50.0, 100.0, 200.0):
        c = ht.heston_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, "call")
        p = ht.heston_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, "put")
        assert c - p == pytest.approx(disc * (fwd - K), abs=PRICE_TOL)
        cg = ht.heston_price_gil_pelaez(S0, K, T, r, q, v0, kappa, theta, xi, rho, "call")
        pg = ht.heston_price_gil_pelaez(S0, K, T, r, q, v0, kappa, theta, xi, rho, "put")
        assert abs(cg - c) <= PRICE_TOL and abs(pg - p) <= PRICE_TOL


@pytest.mark.parametrize("case", list(CASES))
def test_monotone_convex_and_bounded_in_strike(case: str) -> None:
    T, v0, kappa, theta, xi, rho = _case_args(case)
    r, q = 0.02, 0.01
    strikes = np.arange(40.0, 260.0, 5.0)
    calls = np.array([ht.heston_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, "call")
                      for K in strikes])
    disc = math.exp(-r * T)
    fwd = S0 * math.exp((r - q) * T)
    assert np.all(np.diff(calls) <= PRICE_TOL)                     # nonincreasing
    assert np.all(np.diff(calls, 2) >= -2 * PRICE_TOL)             # convex
    # Slope bounds: -e^{-rT} <= dC/dK <= 0.
    assert np.all(np.diff(calls) / 5.0 >= -disc - PRICE_TOL)
    # Static bounds.
    assert np.all(calls >= np.maximum(disc * (fwd - strikes), 0.0) - PRICE_TOL)
    assert np.all(calls <= S0 * math.exp(-q * T) + PRICE_TOL)


# ---------------------------------------------------------------------------
# Black-Scholes limits and degenerate inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("xi", [1e-2, 1e-4])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_black_scholes_limit_small_xi(xi: float, kind: OptionKind) -> None:
    # v0 = theta: the xi -> 0 law is GBM with sigma = sqrt(theta); the price
    # error is O(xi) (researcher: 0.023 at 1e-2, 2.2e-4 at 1e-4). Tolerance
    # 5 xi S0 as recommended in the opinion.
    theta, T, r, q = 0.04, 1.0, 0.03, 0.01
    for K in (80.0, 100.0, 125.0):
        h = ht.heston_price(S0, K, T, r, q, theta, 1.5, theta, xi, -0.6, kind)
        bs = float(bs_price(S0, K, T, r, q, math.sqrt(theta), kind))
        assert abs(h - bs) <= 5 * xi * S0, (K, h, bs)
        assert abs(h - bs_ref(S0, K, T, r, q, theta * T, kind)) <= 5 * xi * S0


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_xi_is_black_scholes_with_integrated_variance(kind: OptionKind) -> None:
    v0, kappa, theta, T, r, q = 0.09, 2.0, 0.04, 1.5, 0.03, 0.01
    iv = theta * T + (v0 - theta) * (1.0 - math.exp(-kappa * T)) / kappa
    for K in (70.0, 100.0, 130.0):
        for fn in (ht.heston_price, ht.heston_price_gil_pelaez):
            got = fn(S0, K, T, r, q, v0, kappa, theta, 0.0, 0.3, kind)
            assert got == pytest.approx(bs_ref(S0, K, T, r, q, iv, kind), abs=1e-10)


@pytest.mark.parametrize("xi", [1e-5, 9.99e-5, 1e-8])
def test_tiny_positive_xi_rejected(xi: float) -> None:
    for fn in (ht.heston_price, ht.heston_price_gil_pelaez):
        with pytest.raises(ValueError, match="xi"):
            fn(S0, 100.0, 1.0, 0.0, 0.0, 0.04, 1.0, 0.04, xi, -0.5, "call")
    # The boundary itself is accepted.
    assert ht.heston_price(S0, 100.0, 1.0, 0.0, 0.0, 0.04, 1.0, 0.04, 1e-4, -0.5,
                           "call") > 0


def test_short_maturity_one_day() -> None:
    T = 1.0 / 252.0
    args = (T, 0.03, 0.01, 0.04, 2.0, 0.04, 0.5, -0.7)
    for K in (90.0, 97.0, 100.0, 103.0, 110.0):
        kinds: tuple[OptionKind, ...] = ("call", "put")
        for kind in kinds:
            got = ht.heston_price(S0, K, *args, kind)
            ref = price_quad_ref(S0, K, *args, kind=kind)
            gp = ht.heston_price_gil_pelaez(S0, K, *args, kind)
            assert abs(got - ref) <= PRICE_TOL, (K, kind, got, ref)
            assert abs(got - gp) <= PRICE_TOL
            assert got >= 0.0
    # Near expiry the ATM price is close to Black-Scholes at sqrt(v0) (O(T)
    # corrections): one-day ATM ~ 0.506, within 1% of BS.
    atm = ht.heston_price(S0, 100.0, *args, "call")
    assert atm == pytest.approx(bs_ref(S0, 100.0, T, 0.03, 0.01, 0.04 * T, "call"),
                                rel=1e-2)


def test_zero_maturity_is_intrinsic() -> None:
    for fn in (ht.heston_price, ht.heston_price_gil_pelaez):
        assert fn(S0, 90.0, 0.0, 0.03, 0.0, 0.04, 1.0, 0.04, 0.5, -0.5, "call") == 10.0
        assert fn(S0, 90.0, 0.0, 0.03, 0.0, 0.04, 1.0, 0.04, 0.5, -0.5, "put") == 0.0
        assert fn(S0, 110.0, 0.0, 0.03, 0.0, 0.04, 1.0, 0.04, 0.5, -0.5, "put") == 10.0


def test_deep_otm_price_is_nonnegative_and_not_negative_zero() -> None:
    T, v0, kappa, theta, xi, rho = _case_args("I")
    p = ht.heston_price(S0, 1e4, 1.0, 0.0, 0.0, v0, kappa, theta, xi, rho, "call")
    assert p >= 0.0 and math.copysign(1.0, p) == 1.0
    assert p <= 1e-6


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
BASE: dict[str, Any] = dict(S=100.0, K=100.0, T=1.0, r=0.01, q=0.0, v0=0.04, kappa=1.0,
            theta=0.04, xi=0.5, rho=-0.5, kind="call")
BAD_INPUTS = [
    ("S", 0.0), ("S", -1.0), ("S", True), ("S", math.nan), ("K", 0.0),
    ("K", math.inf), ("T", -1.0), ("T", math.nan), ("r", math.nan), ("q", True),
    ("v0", -0.01), ("v0", True), ("kappa", 0.0), ("kappa", -1.0), ("theta", 0.0),
    ("xi", -0.1), ("xi", math.nan), ("rho", 1.01), ("rho", -1.5), ("rho", np.bool_(True)),
    ("kind", "straddle"), ("kind", None), ("v0", "0.04"), ("r", 1e400),
]


@pytest.mark.parametrize("name, value", BAD_INPUTS)
def test_price_input_validation(name: str, value: Any) -> None:
    args = dict(BASE)
    args[name] = value
    for fn in (ht.heston_price, ht.heston_price_gil_pelaez):
        with pytest.raises(ValueError):
            fn(**args)


@pytest.mark.parametrize("nodes, tol", [(3, 1e-12), (True, 1e-12), (32.0, 1e-12),
                                        (32, 0.0), (32, -1.0), (32, math.nan)])
def test_quadrature_parameter_validation(nodes: Any, tol: Any) -> None:
    with pytest.raises(ValueError):
        ht.heston_price(**BASE, nodes=nodes, tol=tol)


def test_moment_and_feller_input_validation() -> None:
    with pytest.raises(ValueError):
        ht.heston_variance_mean(-0.1, 1.0, 0.04, 1.0)
    with pytest.raises(ValueError):
        ht.heston_variance_mean(0.04, 0.0, 0.04, 1.0)
    with pytest.raises(ValueError):
        ht.heston_variance_mean(0.04, 1.0, 0.04, -1.0)
    with pytest.raises(ValueError):
        ht.heston_variance_mean(0.04, 1.0, 0.04, True)
    with pytest.raises(ValueError):
        ht.heston_variance_mean(0.04, 1.0, 0.04, np.array([0.5, np.nan]))
    with pytest.raises(ValueError):
        ht.heston_variance_var(0.04, 1.0, 0.04, -0.3, 1.0)
    with pytest.raises(ValueError):
        ht.feller_condition(True, 0.04, 0.3)
    with pytest.raises(ValueError):
        ht.feller_condition(1.0, -0.04, 0.3)
    with pytest.raises(ValueError):
        ht.heston_expected_terminal(0.0, 0.05, 1.0)


# ---------------------------------------------------------------------------
# Regressions of the independent review of d64c716 (P3-3, P3-5)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("xi", [1e-5, 9.99e-5, 1e-8])
def test_cf_rejects_tiny_positive_xi(xi: float) -> None:
    # P3-3: at xi = 1e-8 the CF erred by ~0.04 against the xi = 0 limit.
    with pytest.raises(ValueError, match="xi"):
        ht.heston_characteristic_function(1.3, 1.0, 0.04, 1.5, 0.04, xi, -0.7)
    for ok in (0.0, ht.XI_MIN):
        assert np.isfinite(ht.heston_characteristic_function(1.3, 1.0, 0.04, 1.5, 0.04,
                                                             ok, -0.7))


# kappa < rho xi with long T: Re b = kappa - rho xi < 0 on the P1 contour
# u = x - i, and the share-measure law of ln S_T is extremely wide, so the
# P1 integrand varies on the scale x ~ 1e-9.
_KAPPA_LT_RHO_XI = [(15.0, 0.5, 2.0, 0.9), (20.0, 0.2, 1.5, 0.95)]


@pytest.mark.parametrize("T, kappa, xi, rho", _KAPPA_LT_RHO_XI)
def test_p1_integrand_is_smooth_near_u_minus_i(T: float, kappa: float, xi: float,
                                               rho: float) -> None:
    # P3-5: cancellation in i u + u^2, in b + d (Re b < 0), in E = 1 - (1 - E)
    # and in log1p(q) with q near -1 made the P1 integrand noisy (relative
    # third differences 5e-9 .. 1.3e-7 here); smooth now (measured ~6e-12).
    for c in (1e-8, 1e-6, 1e-4):
        x = np.linspace(c, 1.01 * c, 101)
        f = (ht.heston_characteristic_function(x - 1j, T, 0.04, kappa, 0.04, xi, rho)
             / (1j * x)).real
        assert float(np.max(np.abs(np.diff(f, 3)))) <= 1e-10 * float(np.max(np.abs(f)))


@pytest.mark.parametrize("T, kappa, xi, rho", _KAPPA_LT_RHO_XI)
def test_gil_pelaez_converges_when_kappa_below_rho_xi(T: float, kappa: float,
                                                      xi: float, rho: float) -> None:
    # P3-5: Gil-Pelaez raised ValueError (30 bisections) for these inputs.
    for K in STRIKES:
        for kind in ("call", "put"):
            lewis = ht.heston_price(S0, K, T, 0.01, 0.0, 0.04, kappa, 0.04, xi, rho, kind)
            gp = ht.heston_price_gil_pelaez(S0, K, T, 0.01, 0.0, 0.04, kappa, 0.04, xi,
                                            rho, kind)
            assert abs(gp - lewis) <= PRICE_TOL, (K, kind, gp, lewis)


def test_cf_stable_form_matches_plain_trap_where_well_conditioned() -> None:
    # The b +/- d switch, u (u + i), E = exp(-dT) and the log of the quotient
    # are the same algebra as the plain trap formula: compare where the plain
    # formula is accurate (|x| >= 0.05), on the pricing contours, including
    # Re b < 0 (kappa < rho xi) on Im u = -1/2 and -1.
    u = np.concatenate([np.linspace(-30, -0.05, 90), np.linspace(0.05, 30, 90)])
    for (T, kappa, xi, rho) in [(2.0, 0.5, 2.0, 0.9), (5.0, 0.2, 1.5, 0.95),
                                (1.0, 2.0, 0.5, -0.7)]:
        for shift in (0.0, -0.5j, -1j):
            got = ht.heston_characteristic_function(u + shift, T, 0.04, kappa, 0.04, xi,
                                                    rho, s0=S0)
            ref = cf_trap_ref(u + shift, S0, T, 0.0, 0.0, 0.04, kappa, 0.04, xi, rho)
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-13)
