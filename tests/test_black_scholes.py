"""Offline checks for the analytic Black–Scholes–Merton module (E3.1, E3.2)."""

import itertools
import math
import warnings

import numpy as np
import pytest

from pricing import bs_greeks, bs_price, implied_volatility, pricing_metadata

EPS = np.finfo(float).eps

# (S, K, T, r, q, sigma) grid with q != 0, negative rates and short/long T.
GRID = [
    (100.0, K, T, r, q, sig)
    for K in (70.0, 95.0, 100.0, 110.0, 140.0)
    for T in (0.05, 0.5, 2.0)
    for r, q in ((0.05, 0.0), (0.03, 0.02), (-0.01, 0.04))
    for sig in (0.1, 0.35)
]


@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_put_call_parity(S, K, T, r, q, sig):
    c = bs_price(S, K, T, r, q, sig, "call")
    p = bs_price(S, K, T, r, q, sig, "put")
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    # Relative to the scale of the legs, since rhs itself can be ~0.
    scale = max(S * math.exp(-q * T), K * math.exp(-r * T))
    assert abs((c - p) - rhs) <= 1e-10 * scale


def test_hull_reference_values():
    # Hull, Options, Futures, and Other Derivatives, 9th ed., ch. 15 (BSM);
    # example number (probably 15.6) and page not verified in the source.
    # Printed values to 2 decimals; high-precision values from the E3.0 note.
    c = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "call")
    p = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "put")
    assert abs(c - 4.76) <= 5e-3 and abs(p - 0.81) <= 5e-3
    assert c == pytest.approx(4.759422392871532, rel=1e-12)
    assert p == pytest.approx(0.8085993729000922, rel=1e-12)
    # Hull 9th ed., ch. 17 (index with dividend yield q); printed 51.83.
    c_idx = bs_price(930, 900, 2 / 12, 0.08, 0.03, 0.20, "call")
    assert abs(c_idx - 51.83) <= 5e-3
    assert c_idx == pytest.approx(51.83295679649086, rel=1e-12)


def test_hull_greeks_example():
    # Hull 9th ed., ch. 19 ("The Greek letters"); printed values, example
    # number and page not verified in the source.
    g = bs_greeks(49, 50, 20 / 52, 0.05, 0.0, 0.20, "call")
    assert g["delta"] == pytest.approx(0.522, rel=5e-3)
    assert g["gamma"] == pytest.approx(0.066, rel=1e-2)  # printed with 2 sig. digits
    assert g["vega"] == pytest.approx(12.1, rel=5e-3)
    assert g["theta"] == pytest.approx(-4.31, rel=5e-3)
    assert g["rho"] == pytest.approx(8.91, rel=5e-3)
    assert set(g["units"]) == {"delta", "gamma", "vega", "theta", "rho"}


@pytest.mark.parametrize("S,K,T,r,q,sig", GRID)
def test_no_arbitrage_bounds(S, K, T, r, q, sig):
    ds, dk = S * math.exp(-q * T), K * math.exp(-r * T)
    c = bs_price(S, K, T, r, q, sig, "call")
    p = bs_price(S, K, T, r, q, sig, "put")
    tol = 1e-12 * max(ds, dk)
    assert max(ds - dk, 0.0) - tol <= c <= ds + tol
    assert max(dk - ds, 0.0) - tol <= p <= dk + tol


def test_degenerate_cases_without_numpy_warnings():
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        S, K, r, q = 100.0, 90.0, 0.05, 0.02
        # sigma == 0, T > 0: discounted payoff on the deterministic forward.
        T = 1.0
        fwd_call = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        assert bs_price(S, K, T, r, q, 0.0, "call") == pytest.approx(fwd_call, rel=1e-15)
        assert bs_price(S, K, T, r, q, 0.0, "put") == 0.0
        assert bs_price(S, 120.0, T, r, q, 0.0, "put") == pytest.approx(
            120 * math.exp(-r * T) - S * math.exp(-q * T), rel=1e-15)
        # T == 0: intrinsic value.
        assert bs_price(S, K, 0.0, r, q, 0.3, "call") == 10.0
        assert bs_price(S, K, 0.0, r, q, 0.3, "put") == 0.0
        # Forward == strike exactly: 0.0, and greeks are undefined.
        assert bs_price(S, S, 0.0, r, q, 0.3, "call") == 0.0
        with pytest.raises(ValueError, match="forward == strike"):
            bs_greeks(S, S, 0.0, r, q, 0.3, "call")
        # Zero-variance greeks (contract of the principal).
        g = bs_greeks(S, K, T, r, q, 0.0, "call")
        assert g["delta"] == pytest.approx(math.exp(-q * T))
        assert g["gamma"] == 0.0 and g["vega"] == 0.0
        assert g["theta"] == pytest.approx(q * S * math.exp(-q * T) - r * K * math.exp(-r * T))
        assert g["rho"] == pytest.approx(K * T * math.exp(-r * T))
        gp = bs_greeks(S, K, T, r, q, 0.0, "put")
        assert gp["delta"] == 0.0 and gp["theta"] == 0.0 and gp["rho"] == 0.0
        g0 = bs_greeks(S, K, 0.0, r, q, 0.3, "call")
        assert g0["delta"] == 1.0 and g0["rho"] == 0.0
        assert g0["theta"] == pytest.approx(q * S - r * K)


def test_zero_variance_limit_matches_small_sigma():
    # sigma -> 0+ limit of the general formula equals the degenerate branch.
    for kind in ("call", "put"):
        for K in (80.0, 120.0):
            assert bs_price(100, K, 1.0, 0.03, 0.01, 1e-9, kind) == pytest.approx(
                bs_price(100, K, 1.0, 0.03, 0.01, 0.0, kind), abs=1e-12)


def test_deep_out_of_the_money_is_nonnegative_not_negative_zero():
    with np.errstate(all="raise"):
        for K in (1e-3, 1e6):
            for kind in ("call", "put"):
                v = bs_price(100.0, K, 0.01, 0.05, 0.0, 0.1, kind)
                assert v >= 0.0 and math.copysign(1.0, v) == 1.0


def test_vectorized_strike_matches_scalar():
    Ks = np.array([60.0, 90.0, 100.0, 130.0])
    vec = bs_price(100.0, Ks, 0.7, 0.02, 0.01, 0.25, "put")
    assert isinstance(vec, np.ndarray) and vec.shape == Ks.shape
    scal = [bs_price(100.0, float(k), 0.7, 0.02, 0.01, 0.25, "put") for k in Ks]
    np.testing.assert_array_equal(vec, scal)
    g = bs_greeks(100.0, Ks, 0.7, 0.02, 0.01, 0.25, "call")
    assert g["delta"].shape == Ks.shape


GREEK_CASES = [
    (100.0, K, T, 0.04, 0.02, sig, kind)
    for K in (85.0, 100.0, 120.0)
    for T in (0.25, 1.5)
    for sig in (0.15, 0.4)
    for kind in ("call", "put")
]


@pytest.mark.parametrize("S,K,T,r,q,sig,kind", GREEK_CASES)
def test_greeks_match_central_differences(S, K, T, r, q, sig, kind):
    # Central differences have truncation error O(h^2 * V''') and rounding
    # error ~ eps * V / h. With h = 1e-5 * scale the truncation term is
    # ~1e-10 relative and rounding ~1e-11 * V / (h), both well below the
    # tolerances. Gamma (second difference) has rounding ~ eps * V / h^2,
    # so it uses a larger h = 1e-4 * S (rounding ~1e-8, truncation ~1e-8).
    g = bs_greeks(S, K, T, r, q, sig, kind)

    def v(**kw):
        a = {"S": S, "T": T, "r": r, "sigma": sig}
        a.update(kw)
        return bs_price(a["S"], K, a["T"], a["r"], q, a["sigma"], kind)

    h = 1e-5 * S
    assert g["delta"] == pytest.approx((v(S=S + h) - v(S=S - h)) / (2 * h), rel=1e-6, abs=1e-8)
    h = 1e-4 * S
    fd_gamma = (v(S=S + h) - 2 * v() + v(S=S - h)) / h**2
    assert g["gamma"] == pytest.approx(fd_gamma, rel=1e-4, abs=1e-7)
    h = 1e-5 * max(sig, 1e-2)
    assert g["vega"] == pytest.approx((v(sigma=sig + h) - v(sigma=sig - h)) / (2 * h), rel=1e-6)
    h = 1e-5
    assert g["rho"] == pytest.approx((v(r=r + h) - v(r=r - h)) / (2 * h), rel=1e-6)
    h = 1e-5 * T
    fd_theta = -(v(T=T + h) - v(T=T - h)) / (2 * h)
    assert g["theta"] == pytest.approx(fd_theta, rel=1e-6, abs=1e-8 * S)


IV_CASES = list(itertools.product(
    (0.8, 0.9, 1.0, 1.1, 1.25), (0.1, 0.5, 1.0, 2.0), (0.1, 0.3, 0.6), ("call", "put")))


@pytest.mark.parametrize("m,T,sig,kind", IV_CASES)
def test_implied_volatility_round_trip(m, T, sig, kind):
    S, K, r, q = 100.0, 100.0 * m, 0.03, 0.01
    price = bs_price(S, K, T, r, q, sig, kind)
    iv = implied_volatility(price, S, K, T, r, q, kind)
    vega = bs_greeks(S, K, T, r, q, sig, kind)["vega"]
    # A price rounding error ~ eps*max(S, K) moves sigma by about that over
    # vega. Where that sensitivity is < 1e-10 the 1e-8 round-trip bound has
    # a 100x margin; otherwise the problem is ill-conditioned and only the
    # price is required to be reproduced.
    if EPS * max(S, K) / vega < 1e-10:
        assert abs(iv - sig) < 1e-8
    else:
        assert abs(bs_price(S, K, T, r, q, iv, kind) - price) <= 1e-12 * max(S, K)


def test_implied_volatility_small_vega_reproduces_price():
    S, K, T, r, q = 100.0, 60.0, 0.1, 0.03, 0.0
    price = bs_price(S, K, T, r, q, 0.3, "put")
    vega = bs_greeks(S, K, T, r, q, 0.3, "put")["vega"]
    assert EPS * max(S, K) / vega >= 1e-10  # genuinely ill-conditioned case
    iv = implied_volatility(price, S, K, T, r, q, "put")
    assert abs(bs_price(S, K, T, r, q, iv, "put") - price) <= 1e-12 * max(S, K)


def test_implied_volatility_rejections():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    ds, dk = S * math.exp(-q * T), K * math.exp(-r * T)
    with pytest.raises(ValueError, match="T == 0"):
        implied_volatility(5.0, S, K, 0.0, r, q, "call")
    for bad in (ds, ds + 1.0, max(ds - dk, 0.0), -1.0):
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_volatility(bad, S, K, T, r, q, "call")
    for bad in (dk, 0.0):
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_volatility(bad, S, K, T, r, q, "put")
    # Inside the bounds but above bs(sigma=5): not reachable in the bracket.
    high = bs_price(S, K, T, r, q, 6.0, "call")
    assert high < ds
    with pytest.raises(ValueError, match="not reachable within the bracket"):
        implied_volatility(high, S, K, T, r, q, "call")
    assert implied_volatility(high, S, K, T, r, q, "call", bracket=(1e-6, 10.0)) == pytest.approx(6.0, rel=1e-9)
    with pytest.raises(ValueError, match="bracket"):
        implied_volatility(5.0, S, K, T, r, q, "call", bracket=(1.0, 0.5))


@pytest.mark.parametrize("args,msg", [
    ((0.0, 100, 1, 0.0, 0.0, 0.2, "call"), "S must be positive"),
    ((100, -1, 1, 0.0, 0.0, 0.2, "call"), "K must be positive"),
    ((100, 100, -1, 0.0, 0.0, 0.2, "call"), "T must be nonnegative"),
    ((100, 100, 1, 0.0, 0.0, -0.2, "call"), "sigma must be nonnegative"),
    ((100, 100, 1, float("nan"), 0.0, 0.2, "call"), "r must be finite"),
    ((100, 100, 1, 0.0, float("inf"), 0.2, "call"), "q must be finite"),
    ((True, 100, 1, 0.0, 0.0, 0.2, "call"), "bool"),
    ((100, 100, 1, 0.0, 0.0, 0.2, "straddle"), "kind"),
    ((100, [100, float("nan")], 1, 0.0, 0.0, 0.2, "call"), "K must be finite"),
    ((100, np.array([True]), 1, 0.0, 0.0, 0.2, "call"), "bool"),
    ((100, "100", 1, 0.0, 0.0, 0.2, "call"), "K must be a real"),
])
def test_invalid_inputs(args, msg):
    with pytest.raises(ValueError, match=msg):
        bs_price(*args)
    with pytest.raises(ValueError, match=msg):
        bs_greeks(*args)


def test_overflow_raises_instead_of_inf():
    with pytest.raises(ValueError, match="not finite"):
        bs_price(1e300, 1.0, 100.0, 0.0, -10.0, 0.2, "call")


def test_pricing_metadata():
    meta = pricing_metadata(0.03, 0.01, 0.2, "historical_proxy")
    assert meta == {"measure": "risk_neutral", "model": "black_scholes_merton",
                    "dividends": "continuous_yield", "r": 0.03, "q": 0.01,
                    "sigma": 0.2, "sigma_source": "historical_proxy"}
    with pytest.raises(ValueError, match="sigma_source"):
        pricing_metadata(0.03, 0.01, 0.2, "implied")
