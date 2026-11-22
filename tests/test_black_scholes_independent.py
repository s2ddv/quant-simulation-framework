"""Independent checks of the BSM pricing module (E3.1/E3.2, testador_quant).

References here do not reuse the closed-form formulas under test:

* prices are recomputed as the discounted risk-neutral expectation of the
  payoff, integrated numerically against the lognormal law of S_T under Q
  (``scipy.integrate.quad``);
* delta is the integral of ``dS_T/dS`` over the exercise region (derivative
  under the integral sign); gamma is the boundary (Leibniz) term of that
  derivative; vega, rho and theta are central differences of the integral;
* the returned Greeks are also checked against the BSM PDE with yield q;
* structural properties (monotonicity, convexity in K, Bates/Haug put-call
  symmetry, continuity of the zero-variance limits) need no formula at all.

Model (Q): ``S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)``, Z ~ N(0,1).
Units: S, K in currency; T in years; r, q in year^-1; sigma in year^-1/2.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pricing import bs_greeks, bs_price, implied_volatility, pricing_metadata

EPS = np.finfo(float).eps
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) * _INV_SQRT_2PI


def _exercise_boundary(S, K, T, r, q, sig):
    """z* such that S_T(z*) == K, with S_T(z) = S exp(m + s z)."""
    m = (r - q - 0.5 * sig * sig) * T
    s = sig * math.sqrt(T)
    return m, s, (math.log(K / S) - m) / s


def _integral_price(S, K, T, r, q, sig, kind):
    """e^{-rT} E^Q[payoff(S_T)] by adaptive quadrature over Z.

    The integrand is smooth on the exercise region, so the domain is cut at
    the exercise boundary z*. Beyond |z - z*| = 40 + |z*| the Gaussian weight
    is below e^{-800}, negligible against any tolerance used here.
    Returns (price, abserr) where abserr is quad's own error estimate.
    """
    m, s, zs = _exercise_boundary(S, K, T, r, q, sig)
    disc = math.exp(-r * T)
    width = 40.0 + abs(zs)

    def payoff_density(z):
        st = S * math.exp(m + s * z)
        val = st - K if kind == "call" else K - st
        return val * _phi(z)

    a, b = (zs, zs + width) if kind == "call" else (zs - width, zs)
    val, err = quad(payoff_density, a, b, epsabs=0.0, epsrel=1e-13, limit=200)
    return disc * val, disc * err


def _integral_delta_gamma(S, K, T, r, q, sig, kind):
    """Delta and gamma from differentiating the expectation in S.

    delta = e^{-rT} ∫_{exercise} (S_T/S) phi(z) dz (sign -1 for the put); the
    boundary term vanishes because the payoff is 0 at z*. Differentiating
    again only the moving boundary contributes (S_T/S is S-independent):
    gamma = e^{-rT} (K/S) phi(z*) / (S s), for both kinds.
    """
    m, s, zs = _exercise_boundary(S, K, T, r, q, sig)
    disc = math.exp(-r * T)
    width = 40.0 + abs(zs)

    def ratio(z):
        return math.exp(m + s * z) * _phi(z)

    if kind == "call":
        val, _ = quad(ratio, zs, zs + width, epsabs=0.0, epsrel=1e-13, limit=200)
        delta = disc * val
    else:
        val, _ = quad(ratio, zs - width, zs, epsabs=0.0, epsrel=1e-13, limit=200)
        delta = -disc * val
    gamma = disc * (K / S) * _phi(zs) / (S * s)
    return delta, gamma


# Grid with q != 0, negative r, deep/near-the-money strikes, short and long T.
GRID = [
    (100.0, K, T, r, q, sig)
    for K in (60.0, 97.0, 100.0, 125.0)
    for T in (0.02, 0.75, 3.0)
    for r, q in ((-0.02, 0.03), (0.06, 0.015))
    for sig in (0.08, 0.45)
]
KINDS = ("call", "put")


# ---------------------------------------------------------------------------
# (a) price vs numerical integral
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_price_matches_risk_neutral_integral(S, K, T, r, q, sig, kind):
    ref, err = _integral_price(S, K, T, r, q, sig, kind)
    got = bs_price(S, K, T, r, q, sig, kind)
    # 1e-8 relative is the requested accuracy; the quadrature itself is
    # requested at 1e-13 (its estimate `err` is added). Deep OTM prices can be
    # ~1e-30, where relative accuracy of a difference of O(S) terms is not
    # meaningful: the absolute floor 1e-13*S is ~500 eps of the leg size.
    assert abs(got - ref) <= 1e-8 * abs(ref) + err + 1e-13 * S


# ---------------------------------------------------------------------------
# (b) Greeks vs derivatives of the integral, and the BSM PDE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_delta_gamma_match_integral_derivatives(S, K, T, r, q, sig, kind):
    g = bs_greeks(S, K, T, r, q, sig, kind)
    d_ref, g_ref = _integral_delta_gamma(S, K, T, r, q, sig, kind)
    # Delta is a quadrature at epsrel 1e-13 of an O(1) quantity: 1e-10 abs.
    assert abs(g["delta"] - d_ref) <= 1e-10
    # Gamma reference is a closed boundary term (exp/log only): relative
    # 1e-10 covers rounding in d1 for |d1| up to ~30; abs floor for tails.
    assert abs(g["gamma"] - g_ref) <= 1e-10 * abs(g_ref) + 1e-15


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_vega_rho_theta_match_integral_differences(S, K, T, r, q, sig, kind):
    g = bs_greeks(S, K, T, r, q, sig, kind)

    def V(**kw):
        p = {"S": S, "K": K, "T": T, "r": r, "q": q, "sig": sig}
        p.update(kw)
        return _integral_price(p["S"], p["K"], p["T"], p["r"], p["q"], p["sig"], kind)[0]

    # Central differences: truncation O(h^2 * V''') and rounding
    # O(1e-13 * V / h). h = 1e-4 (relative for sigma, T; absolute for r)
    # gives both ~1e-8 of the scale; tolerance 1e-6 * scale leaves margin.
    scale = S + K
    hs = 1e-4 * sig
    vega = (V(sig=sig + hs) - V(sig=sig - hs)) / (2 * hs)
    hr = 1e-4
    rho = (V(r=r + hr) - V(r=r - hr)) / (2 * hr)
    ht = 1e-4 * T
    theta = -(V(T=T + ht) - V(T=T - ht)) / (2 * ht)
    # Vega and theta scale like S*sqrt(T), S/sqrt(T); use T-aware scales.
    assert abs(g["vega"] - vega) <= 1e-6 * scale * max(1.0, math.sqrt(T))
    assert abs(g["rho"] - rho) <= 1e-6 * scale * max(1.0, T)
    assert abs(g["theta"] - theta) <= 1e-6 * scale / math.sqrt(T)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_greeks_satisfy_bsm_pde(S, K, T, r, q, sig, kind):
    # theta_t + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0,
    # with theta_t = dV/dt (calendar time) = the returned theta.
    g = bs_greeks(S, K, T, r, q, sig, kind)
    V = bs_price(S, K, T, r, q, sig, kind)
    terms = [g["theta"], 0.5 * sig**2 * S**2 * g["gamma"],
             (r - q) * S * g["delta"], -r * V]
    # Each term carries O(eps) relative rounding; 1e-9 of the largest term
    # (plus an eps-level floor in currency) is ~10^6 eps of slack.
    scale = max(abs(t) for t in terms) + EPS * S
    assert abs(sum(terms)) <= 1e-9 * scale


# ---------------------------------------------------------------------------
# (c) monotonicity and convexity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,q", [(-0.02, 0.03), (0.06, 0.015)])
def test_monotonic_in_spot_strike_and_vol(r, q):
    T, sig = 0.75, 0.3
    spots = np.linspace(60.0, 140.0, 81)
    c_s = np.array([bs_price(s, 100.0, T, r, q, sig, "call") for s in spots])
    p_s = np.array([bs_price(s, 100.0, T, r, q, sig, "put") for s in spots])
    # On this range all prices are >= 1e-3, far from underflow, so strict
    # monotonicity is expected (delta never vanishes for sigma*sqrt(T)=0.26).
    assert np.all(np.diff(c_s) > 0) and np.all(np.diff(p_s) < 0)

    vols = np.linspace(0.02, 1.5, 75)
    for K in (80.0, 100.0, 120.0):
        c_v = np.array([bs_price(100.0, K, T, r, q, v, "call") for v in vols])
        p_v = np.array([bs_price(100.0, K, T, r, q, v, "put") for v in vols])
        # Vega > 0; at sigma=0.02 deep ITM/OTM vega can underflow towards 0,
        # so allow ties at the level of eps*S but forbid any decrease.
        assert np.all(np.diff(c_v) >= -4 * EPS * 100.0)
        assert np.all(np.diff(p_v) >= -4 * EPS * 100.0)
        assert c_v[-1] > c_v[0] and p_v[-1] > p_v[0]

    strikes = np.linspace(50.0, 150.0, 101)
    c_k = bs_price(100.0, strikes, T, r, q, sig, "call")
    p_k = bs_price(100.0, strikes, T, r, q, sig, "put")
    assert np.all(np.diff(c_k) < 0) and np.all(np.diff(p_k) > 0)
    # Convexity in K (Breeden–Litzenberger: second difference = e^{-rT} x
    # density x dK^2 >= 0). Rounding in each price is <= ~eps*S, so a
    # second difference may be negative by at most ~4 eps*S.
    for prices in (c_k, p_k):
        assert np.all(np.diff(prices, 2) >= -8 * EPS * 100.0)
    # Slope bounds: -e^{-rT} <= dC/dK <= 0 and 0 <= dP/dK <= e^{-rT}.
    dk = strikes[1] - strikes[0]
    disc = math.exp(-r * T)
    assert np.all(np.diff(c_k) / dk >= -disc - 1e-12)
    assert np.all(np.diff(p_k) / dk <= disc + 1e-12)


# ---------------------------------------------------------------------------
# (d) continuity of the zero-variance limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("K", [80.0, 99.0, 100.0, 101.0, 125.0])
@pytest.mark.parametrize("r,q", [(-0.02, 0.03), (0.06, 0.015), (0.03, 0.03)])
def test_sigma_to_zero_converges_to_forward_payoff(K, r, q, kind):
    S, T = 100.0, 1.3
    base = bs_price(S, K, T, r, q, 0.0, kind)
    # Independent reference for sigma = 0 (contract): discounted payoff on F.
    fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    exp_base = max(fwd, 0.0) if kind == "call" else max(-fwd, 0.0)
    assert abs(base - exp_base) <= 4 * EPS * S
    tiny = bs_price(S, K, T, r, q, 1e-12, kind)
    # |V(sigma) - V(0)| <= S e^{-qT} sigma sqrt(T) phi(0) (vega is bounded by
    # its ATM value), i.e. ~5e-11 here; plus eps-level rounding of the legs.
    assert abs(tiny - base) <= S * 1e-12 * math.sqrt(T) + 8 * EPS * S


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_small_maturity_converges_to_intrinsic(K, kind):
    S, r, q, sig = 100.0, 0.05, -0.01, 0.4
    at0 = bs_price(S, K, 0.0, r, q, sig, kind)
    intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    assert at0 == intrinsic
    errs = []
    for T in (1e-4, 1e-6, 1e-8, 1e-10):
        v = bs_price(S, K, T, r, q, sig, kind)
        # Bound: carry/discount terms (|r|+|q|) * max(S,K) * T plus the
        # time value <= S sigma sqrt(T) phi(0) (ATM worst case).
        bound = (abs(r) + abs(q)) * max(S, K) * T * 2 + S * sig * math.sqrt(T) + 8 * EPS * S
        errs.append(abs(v - intrinsic))
        assert errs[-1] <= bound
    # Away from the money the time value vanishes faster than any power;
    # at the money it is ~S sigma sqrt(T)/sqrt(2 pi). Either way the
    # sequence over T decreasing by 100x must shrink (deterministic formula).
    assert errs[-1] < errs[0] or errs[0] <= 8 * EPS * S


# ---------------------------------------------------------------------------
# (e) put-call symmetry (Bates 1991; Haug 2007): C(S,K,r,q) = P(K,S,q,r)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_put_call_symmetry(S, K, T, r, q, sig):
    c = bs_price(S, K, T, r, q, sig, "call")
    p = bs_price(K, S, T, q, r, sig, "put")
    # Same Phi arguments up to rounding of log/exp: ~10 eps of the legs.
    assert abs(c - p) <= 64 * EPS * max(S, K)
    # The symmetry also maps the put to the call.
    p2 = bs_price(S, K, T, r, q, sig, "put")
    c2 = bs_price(K, S, T, q, r, sig, "call")
    assert abs(p2 - c2) <= 64 * EPS * max(S, K)


# ---------------------------------------------------------------------------
# (f) implied volatility from integral prices, and exact-boundary rejections
# ---------------------------------------------------------------------------

IV_CASES = [
    (100.0, K, T, r, q, sig)
    for K in (80.0, 100.0, 115.0)
    for T in (0.1, 1.0)
    for r, q in ((-0.02, 0.03), (0.06, 0.015))
    for sig in (0.12, 0.6)
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", IV_CASES)
def test_implied_volatility_from_integral_prices(S, K, T, r, q, sig, kind):
    price, err = _integral_price(S, K, T, r, q, sig, kind)
    # Vega from the integral (central difference) so the tolerance does not
    # depend on the module under test.
    h = 1e-4 * sig
    vega = (_integral_price(S, K, T, r, q, sig + h, kind)[0]
            - _integral_price(S, K, T, r, q, sig - h, kind)[0]) / (2 * h)
    est = implied_volatility(price, S, K, T, r, q, kind)
    # Conditioning: d sigma ~ d price / vega. Price uncertainty = quad error
    # estimate + 1e-13 relative + eps-level pricing rounding; 10x margin, plus
    # brentq's xtol 1e-12.
    dprice = err + 1e-13 * price + 8 * EPS * max(S, K)
    assert abs(est - sig) <= 10 * dprice / vega + 1e-11


@pytest.mark.parametrize("kind", KINDS)
def test_implied_volatility_rejects_exact_bounds(kind):
    S, K, T, r, q = 100.0, 90.0 if kind == "call" else 110.0, 0.8, 0.04, 0.01
    # Bounds computed as in the contract, with float64 exp.
    ds = S * float(np.exp(-q * T))
    dk = K * float(np.exp(-r * T))
    lower = max(ds - dk, 0.0) if kind == "call" else max(dk - ds, 0.0)
    upper = ds if kind == "call" else dk
    for p in (lower, upper, np.nextafter(lower, -1.0), np.nextafter(upper, np.inf)):
        with pytest.raises(ValueError):
            implied_volatility(float(p), S, K, T, r, q, kind)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(lower, S, K, T, r, q, kind)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(upper, S, K, T, r, q, kind)
    # OTM option: lower bound is 0; price 0 must be rejected (sigma = 0).
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(0.0, S, 150.0 if kind == "call" else 50.0, T, r, q, kind)


def test_implied_volatility_other_rejections():
    S, K, r, q = 100.0, 100.0, 0.03, 0.01
    with pytest.raises(ValueError):
        implied_volatility(5.0, S, K, 0.0, r, q, "call")  # T == 0
    # Inside the bounds but below bs(lo): bracket starting at 0.2 cannot
    # reach the price of sigma = 0.1 (integral reference).
    p01, _ = _integral_price(S, K, 1.0, r, q, 0.1, "call")
    with pytest.raises(ValueError, match="not reachable"):
        implied_volatility(p01, S, K, 1.0, r, q, "call", bracket=(0.2, 5.0))
    # Above bs(hi) with the default bracket (hi = 5): price of sigma = 7.
    p7, _ = _integral_price(S, K, 1.0, r, q, 7.0, "call")
    with pytest.raises(ValueError, match="not reachable"):
        implied_volatility(p7, S, K, 1.0, r, q, "call")
    with pytest.raises(ValueError):
        implied_volatility(5.0, S, [90.0, 100.0], 1.0, r, q, "call")


# ---------------------------------------------------------------------------
# (g) vectorization in K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("T,sig", [(0.9, 0.25), (0.0, 0.25), (0.9, 0.0)])
def test_vectorized_strikes_match_scalar(T, sig, kind):
    S, r, q = 100.0, -0.01, 0.02
    # Exclude strikes where zero-variance greeks are undefined (F == K):
    # F = 100 e^{-0.027} or 100 at T=0; none of these strikes hit it exactly.
    Ks = np.array([[40.0, 85.5, 99.0], [101.0, 130.0, 250.0]])
    vp = bs_price(S, Ks, T, r, q, sig, kind)
    vg = bs_greeks(S, Ks, T, r, q, sig, kind)
    assert isinstance(vp, np.ndarray) and vp.shape == Ks.shape and vp.dtype == np.float64
    for idx in np.ndindex(Ks.shape):
        k = float(Ks[idx])
        sp = bs_price(S, k, T, r, q, sig, kind)
        sg = bs_greeks(S, k, T, r, q, sig, kind)
        assert isinstance(sp, float)
        # Same formulas; array and 0-d NumPy loops may differ by a few ulp
        # (SIMD exp/log), hence 1e-14 relative instead of bit equality.
        assert vp[idx] == pytest.approx(sp, rel=1e-14, abs=1e-300)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert vg[name].shape == Ks.shape
            assert vg[name][idx] == pytest.approx(sg[name], rel=1e-14, abs=1e-300)
    # Lists are accepted like arrays.
    lst = bs_price(S, [85.5, 130.0], T, r, q, sig, kind)
    assert np.allclose(lst, [vp[0, 1], vp[1, 1]], rtol=1e-14, atol=0.0)


def test_degenerate_greeks_reject_forward_equal_strike():
    # r == q => F == S; sigma = 0 and T = 0 are both zero variance.
    with pytest.raises(ValueError):
        bs_greeks(100.0, 100.0, 1.0, 0.02, 0.02, 0.0, "call")
    with pytest.raises(ValueError):
        bs_greeks(100.0, 100.0, 0.0, 0.05, 0.01, 0.3, "put")
    with pytest.raises(ValueError):
        bs_greeks(100.0, [90.0, 100.0], 0.0, 0.05, 0.01, 0.3, "call")
    # Price at F == K under zero variance is exactly +0.0 (not -0.0).
    v = bs_price(100.0, 100.0, 1.0, 0.02, 0.02, 0.0, "put")
    assert v == 0.0 and math.copysign(1.0, v) == 1.0


def test_prices_never_negative_zero():
    Ks = np.array([1.0, 50.0, 100.0, 400.0, 5000.0])
    for kind in KINDS:
        for T, sig in ((0.01, 0.05), (2.0, 0.3), (0.0, 0.3), (1.0, 0.0)):
            v = bs_price(100.0, Ks, T, 0.03, 0.01, sig, kind)
            assert np.all(v >= 0.0)
            assert np.all(np.copysign(1.0, v) == 1.0)


# ---------------------------------------------------------------------------
# (h) invalid inputs
# ---------------------------------------------------------------------------

VALID = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "q": 0.01, "sigma": 0.2,
         "kind": "call"}
BAD = [
    ("S", True), ("S", math.nan), ("S", math.inf), ("S", 0.0), ("S", -1.0),
    ("K", False), ("K", math.nan), ("K", -math.inf), ("K", 0.0), ("K", -5.0),
    ("K", [100.0, math.nan]), ("K", [100.0, 0.0]), ("K", np.array([True])),
    ("K", "100"),
    ("T", True), ("T", math.nan), ("T", math.inf), ("T", -1e-12),
    ("r", True), ("r", math.nan), ("r", -math.inf),
    ("q", False), ("q", math.nan), ("q", math.inf),
    ("sigma", True), ("sigma", math.nan), ("sigma", math.inf), ("sigma", -1e-12),
    ("kind", "Call"), ("kind", "straddle"), ("kind", None), ("kind", 1),
]


@pytest.mark.parametrize("name,value", BAD)
def test_invalid_inputs_raise_value_error(name, value):
    args = dict(VALID)
    args[name] = value
    pos = (args["S"], args["K"], args["T"], args["r"], args["q"], args["sigma"],
           args["kind"])
    with pytest.raises(ValueError):
        bs_price(*pos)
    with pytest.raises(ValueError):
        bs_greeks(*pos)
    if name != "sigma":  # implied_volatility takes a price instead of sigma
        with pytest.raises(ValueError):
            implied_volatility(10.0, args["S"], args["K"], args["T"], args["r"],
                               args["q"], args["kind"])


@pytest.mark.parametrize("price", [True, math.nan, math.inf, -1.0])
def test_implied_volatility_invalid_price(price):
    with pytest.raises(ValueError):
        implied_volatility(price, 100.0, 100.0, 1.0, 0.03, 0.01, "call")


@pytest.mark.parametrize("source", ["historical", "implied", "", None, 1])
def test_pricing_metadata_rejects_invalid_sigma_source(source):
    with pytest.raises(ValueError):
        pricing_metadata(0.03, 0.01, 0.2, source)


def test_pricing_metadata_contract():
    for source in ("historical_proxy", "user_supplied"):
        md = pricing_metadata(-0.01, 0.02, 0.25, source)
        assert md["measure"] == "risk_neutral"
        assert md["sigma_source"] == source
        assert (md["r"], md["q"], md["sigma"]) == (-0.01, 0.02, 0.25)
    for bad in ((True, 0.0, 0.2), (0.0, math.nan, 0.2), (0.0, 0.0, -0.1)):
        with pytest.raises(ValueError):
            pricing_metadata(*bad, "user_supplied")


# ---------------------------------------------------------------------------
# (h) zero-variance Greeks of in-the-money options (P2-1) and exact
#     implied-volatility bounds (P2-2)
# ---------------------------------------------------------------------------

def _deterministic_value(S, K, T, r, q, kind):
    """Discounted payoff on the deterministic forward (zero variance).

    call: max(S e^{-qT} - K e^{-rT}, 0); put: max(K e^{-rT} - S e^{-qT}, 0).
    Written from the definition, without calling the module under test.
    """
    ds = S * math.exp(-q * T)
    dk = K * math.exp(-r * T)
    return max(ds - dk, 0.0) if kind == "call" else max(dk - ds, 0.0)


def _fd_greeks_zero_variance(S, K, T, r, q, kind):
    """Finite-difference delta, rho and theta (= -dV/dT) of the payoff above.

    Away from F == K the value is a smooth sum of exponentials, so:
    * delta: central difference in S, step hS = 1e-4 S. V is linear in S on
      the exercise side: no truncation error, only rounding ~ eps |V| / hS.
    * rho: central difference in r, hr = 1e-5. Truncation <= hr^2/6 |V_rrr|
      = hr^2/6 T^3 K e^{-rT} (< 1e-8 for the inputs used), rounding
      ~ eps |V| / hr (< 1e-9).
    * theta: central difference in T (hT = 1e-5) when T > 0; at T == 0 the
      second-order one-sided right difference (-3 V0 + 4 V1 - V2) / (2 hT),
      truncation <= hT^2/3 |V_TTT| = hT^2/3 |r^3 K + q^3 S| (tiny).
    """
    hS, hr, hT = 1e-4 * S, 1e-5, 1e-5

    def V(**kw):
        a = dict(S=S, K=K, T=T, r=r, q=q, kind=kind)
        a.update(kw)
        return _deterministic_value(**a)

    delta = (V(S=S + hS) - V(S=S - hS)) / (2.0 * hS)
    rho = (V(r=r + hr) - V(r=r - hr)) / (2.0 * hr)
    if T > 0.0:
        theta = -(V(T=T + hT) - V(T=T - hT)) / (2.0 * hT)
    else:
        theta = -(-3.0 * V(T=0.0) + 4.0 * V(T=hT) - V(T=2.0 * hT)) / (2.0 * hT)
    return delta, rho, theta


# (S, K, T, r, q, sigma): sigma = 0 with T > 0, and T == 0 with sigma > 0 or
# sigma = 0; r and q nonzero, of both signs. Strikes are deep enough in the
# money that the FD steps never cross F == K.
_ZV_ITM_CASES = [
    (100.0, 120.0, 0.75, 0.03, 0.02, 0.0),
    (100.0, 120.0, 0.75, -0.015, 0.04, 0.0),
    (100.0, 120.0, 0.0, 0.05, 0.02, 0.3),
    (100.0, 120.0, 0.0, -0.02, 0.035, 0.0),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,q,sig", _ZV_ITM_CASES)
def test_zero_variance_itm_greeks_match_finite_differences(S, K, T, r, q, sig, kind):
    # Put ITM uses K = 120 > F; call ITM mirrors the strike to K' = S^2 / K
    # (= 83.3) < F. Sign flips of delta/theta/rho change values by
    # 2 |Greek| (>= 0.9 here, except rho == 0 at T == 0), far above the
    # 1e-6 tolerance justified in _fd_greeks_zero_variance.
    k = K if kind == "put" else S * S / K
    g = bs_greeks(S, k, T, r, q, sig, kind)
    delta, rho, theta = _fd_greeks_zero_variance(S, k, T, r, q, kind)
    assert _deterministic_value(S, k, T, r, q, kind) > 1.0  # really ITM
    assert g["delta"] == pytest.approx(delta, abs=1e-6)
    assert g["rho"] == pytest.approx(rho, abs=1e-6)
    assert g["theta"] == pytest.approx(theta, abs=1e-6)
    assert g["gamma"] == 0.0 and g["vega"] == 0.0
    # Signs, independent of any tolerance: a put has delta < 0; with T > 0
    # its rho < 0 (call: delta > 0, rho > 0); at T == 0 rho is exactly 0.
    sgn = 1.0 if kind == "call" else -1.0
    assert sgn * g["delta"] > 0.5
    if T > 0.0:
        assert sgn * g["rho"] > 1.0
    else:
        assert g["rho"] == 0.0
    # Theta must have the sign of the FD reference (|theta| > 0.1 here, so a
    # sign flip cannot hide inside the tolerance).
    assert abs(theta) > 0.1 and math.copysign(1.0, g["theta"]) == math.copysign(1.0, theta)
    # Price consistency with the same reference (value, not just Greeks).
    assert bs_price(S, k, T, r, q, sig, kind) == pytest.approx(
        _deterministic_value(S, k, T, r, q, kind), rel=1e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_implied_volatility_exact_bounds_use_no_arbitrage_message(kind):
    # Distinguishes the no-arbitrage check from the "not reachable" one: a
    # price exactly at a bound must fail in step 2, not fall through to the
    # bracket check (whose message also contains "no-arbitrage").
    S, T, r, q = 100.0, 0.8, 0.04, 0.01
    K = 90.0 if kind == "call" else 110.0  # ITM: lower bound > 0
    ds = S * float(np.exp(-q * T))
    dk = K * float(np.exp(-r * T))
    lower = max(ds - dk, 0.0) if kind == "call" else max(dk - ds, 0.0)
    upper = ds if kind == "call" else dk
    assert lower > 0.0
    for p in (lower, upper):
        with pytest.raises(ValueError, match="violates the no-arbitrage"):
            implied_volatility(p, S, K, T, r, q, kind)
    # OTM: lower bound is exactly 0.
    k_otm = 150.0 if kind == "call" else 50.0
    with pytest.raises(ValueError, match="violates the no-arbitrage"):
        implied_volatility(0.0, S, k_otm, T, r, q, kind)
    upper_otm = ds if kind == "call" else k_otm * float(np.exp(-r * T))
    with pytest.raises(ValueError, match="violates the no-arbitrage"):
        implied_volatility(upper_otm, S, k_otm, T, r, q, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_implied_volatility_not_reachable_message(kind):
    # Strictly inside the bounds but above bs(hi = 5): price of sigma = 7
    # from the integral reference. Must be the bracket error, not the
    # no-arbitrage one.
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    p7, _ = _integral_price(S, K, T, r, q, 7.0, kind)
    upper = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    assert p7 < upper
    with pytest.raises(ValueError, match="not reachable") as exc:
        implied_volatility(p7, S, K, T, r, q, kind)
    assert "violates" not in str(exc.value)
