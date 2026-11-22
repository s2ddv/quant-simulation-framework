"""Offline checks of the Merton (1976) series price (E4.4).

The Monte Carlo comparison against MertonJumpSimulator under Q is pending
(left to the test specialist once src/simulation/jump_diffusion.py is stable).
"""

import itertools
import math

import numpy as np
import pytest

from pricing.black_scholes import bs_price
from pricing.merton import merton_price, merton_pricing_metadata

GRID = list(itertools.product(
    (70.0, 100.0, 130.0),            # K
    (0.1, 1.0, 3.0),                 # T
    ((0.05, 0.0), (-0.01, 0.03)),    # (r, q)
    (0.0, 0.25),                     # sigma
    ((1.0, -0.1, 0.15), (5.0, 0.05, 0.0), (0.3, -0.4, 0.3)),  # (lambda, m_J, delta)
))


@pytest.mark.parametrize("K,T,rq,sigma", [(K, T, rq, s) for K in (70.0, 100.0, 130.0)
                                          for T in (0.1, 2.0)
                                          for rq in ((0.05, 0.0), (-0.01, 0.03))
                                          for s in (0.0, 0.3)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_intensity_is_black_scholes(K, T, rq, sigma, kind):
    r, q = rq
    ref = bs_price(100.0, K, T, r, q, sigma, kind)
    got = merton_price(100.0, K, T, r, q, sigma, 0.0, -0.2, 0.3, kind)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("K,T,rq,sigma,jumps", GRID)
def test_put_call_parity(K, T, rq, sigma, jumps):
    r, q = rq
    S = 100.0
    c = merton_price(S, K, T, r, q, sigma, *jumps, "call")
    p = merton_price(S, K, T, r, q, sigma, *jumps, "put")
    forward_gap = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert c - p == pytest.approx(forward_gap, abs=1e-11 * max(S, K))
    assert c >= max(forward_gap, 0.0) - 1e-12 and p >= max(-forward_gap, 0.0) - 1e-12


def test_monotone_in_strike_and_vectorized():
    strikes = np.linspace(60.0, 140.0, 17)
    calls = merton_price(100.0, strikes, 1.0, 0.03, 0.01, 0.2, 2.0, -0.05, 0.1, "call")
    puts = merton_price(100.0, strikes, 1.0, 0.03, 0.01, 0.2, 2.0, -0.05, 0.1, "put")
    assert isinstance(calls, np.ndarray) and calls.shape == strikes.shape
    assert np.all(np.diff(calls) < 0.0) and np.all(np.diff(puts) > 0.0)
    scalar = [merton_price(100.0, float(k), 1.0, 0.03, 0.01, 0.2, 2.0, -0.05, 0.1, "call")
              for k in strikes]
    np.testing.assert_allclose(calls, scalar, rtol=1e-14)


def test_monotone_in_diffusion_volatility():
    prices = [merton_price(100.0, 100.0, 1.0, 0.03, 0.0, s, 1.0, -0.1, 0.1, "call")
              for s in (0.0, 0.1, 0.2, 0.4)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_degenerate_cases():
    # T = 0: payoff.
    assert merton_price(100.0, 90.0, 0.0, 0.05, 0.0, 0.2, 3.0, -0.1, 0.1, "call") == 10.0
    assert merton_price(100.0, 90.0, 0.0, 0.05, 0.0, 0.2, 3.0, -0.1, 0.1, "put") == 0.0
    # sigma = 0 and delta = 0: deterministic log-jumps m_J; the price is the
    # Poisson mixture of discounted intrinsic values on each forward.
    S, K, T, r, q, lam, m_j = 100.0, 100.0, 1.0, 0.02, 0.0, 2.0, -0.1
    k = math.expm1(m_j)
    expected = sum(
        math.exp(-lam * T) * (lam * T) ** n / math.factorial(n) * math.exp(-r * T)
        * max(S * math.exp((r - q - lam * k) * T + n * m_j) - K, 0.0)
        for n in range(60))
    got = merton_price(S, K, T, r, q, 0.0, lam, m_j, 0.0, "call")
    assert got == pytest.approx(expected, rel=1e-12)
    # sigma = 0 with delta > 0 is regular and positive.
    assert merton_price(S, K, T, r, q, 0.0, lam, m_j, 0.2, "put") > 0.0


@pytest.mark.parametrize("bad", [
    {"jump_intensity": -1.0}, {"jump_std": -0.1}, {"sigma": -0.2}, {"T": -1.0},
    {"jump_mean": math.nan}, {"tol": 0.0}, {"tol": 1.0}, {"kind": "straddle"},
    {"K": True},
])
def test_invalid_inputs_rejected(bad):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "q": 0.0, "sigma": 0.2,
            "jump_intensity": 1.0, "jump_mean": -0.1, "jump_std": 0.1, "kind": "call"}
    tol = bad.pop("tol", 1e-14)
    args.update(bad)
    with pytest.raises(ValueError):
        merton_price(**args, tol=tol)


def test_metadata():
    meta = merton_pricing_metadata(0.03, 0.01, 0.2, "historical_proxy", 1.0, -0.1, 0.1)
    assert meta["measure"] == "risk_neutral"
    assert meta["model"] == "merton_jump_diffusion"
    assert meta["jump_risk"] == "unpriced"
    assert meta["sigma_source"] == "historical_proxy"
    assert (meta["jump_intensity"], meta["jump_mean"], meta["jump_std"]) == (1.0, -0.1, 0.1)
    with pytest.raises(ValueError):
        merton_pricing_metadata(0.03, 0.01, 0.2, "historical_proxy", -1.0, 0.0, 0.1)
