"""Tests for the control-variate estimator (E2.2, offline, seeded).

Model (constant parameters, exact lognormal terminal law under P)::

    S_T = S0 * exp((mu - sigma^2/2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
    E[S_T] = S0 * exp(mu * T)

Estimator under test (docs/plano-expansao.md E2.2), target ``Y``, control
``X = S_T`` with known mean ``E X``::

    theta = mean(Y) - beta * (mean(X) - E X)

References written here, independent of ``simulation.variance_reduction``:

* call ``Y = (S_T - K)+`` under P::

      E[Y]  = S0 e^{mu T} Phi(d1) - K Phi(d2)
      d1    = (ln(S0/K) + (mu + sigma^2/2) T) / (sigma sqrt T),  d2 = d1 - sigma sqrt T

  and, for ``rho^2``, the lognormal partial moments
  ``E[S 1{S>K}] = S0 e^{mu T} Phi(d1)`` and
  ``E[S^2 1{S>K}] = S0^2 e^{(2 mu + sigma^2) T} Phi(d1 + sigma sqrt T)``;
* any other payoff moment by Gauss-Kronrod quadrature over ``Z`` (SciPy).

Units: prices in currency, T in years, mu in 1/year, sigma in 1/sqrt(year).
Terminal samples are drawn directly from ``Z`` with ``numpy.default_rng``
(PCG64) and fixed seeds; one integration test uses the production sampler.

Tolerances follow docs/plano-expansao.md section 2.5: Monte Carlo means are
checked at 4 standard errors (false alarm about 6e-5 per assertion under
asymptotic normality); other tolerances are justified inline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, stats

from simulation.gbm import simulate_gbm
from simulation.variance_reduction import (
    control_variate_mean,
    gbm_terminal_control_mean,
)

S0, MU, SIGMA, T = 100.0, 0.05, 0.2, 1.0
K_SE = 4.0
KEYS = {"estimate", "standard_error", "beta", "rho2", "vrf", "n", "n_pilot",
        "beta_source", "bias", "warning"}
EX = S0 * math.exp(MU * T)  # independent of gbm_terminal_control_mean


# --- independent references -------------------------------------------------

def _terminal(rng: np.random.Generator, n: int, sigma: float = SIGMA) -> np.ndarray:
    z = rng.standard_normal(n)
    return S0 * np.exp((MU - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)


def _call(x: np.ndarray, strike: float = 100.0) -> np.ndarray:
    return np.maximum(x - strike, 0.0)


def _put(x: np.ndarray, strike: float = 60.0) -> np.ndarray:
    return np.maximum(strike - x, 0.0)


def _d1_d2(strike: float) -> tuple[float, float]:
    vol = SIGMA * math.sqrt(T)
    d1 = (math.log(S0 / strike) + (MU + 0.5 * SIGMA**2) * T) / vol
    return d1, d1 - vol


def _call_mean_analytic(strike: float = 100.0) -> float:
    d1, d2 = _d1_d2(strike)
    return S0 * math.exp(MU * T) * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)


def _call_rho2_analytic(strike: float = 100.0) -> float:
    """rho^2 of (S_T, (S_T - K)+) from closed-form lognormal partial moments."""
    d1, d2 = _d1_d2(strike)
    vol = SIGMA * math.sqrt(T)
    p = stats.norm.cdf(d2)
    m1 = S0 * math.exp(MU * T) * stats.norm.cdf(d1)                     # E[S 1]
    m2 = S0**2 * math.exp((2 * MU + SIGMA**2) * T) * stats.norm.cdf(d1 + vol)  # E[S^2 1]
    ey = m1 - strike * p
    ey2 = m2 - 2 * strike * m1 + strike**2 * p
    var_x = S0**2 * math.exp(2 * MU * T) * math.expm1(SIGMA**2 * T)
    cov = (m2 - strike * m1) - EX * ey
    return cov**2 / (var_x * (ey2 - ey**2))


def _expect(f: Callable[[float], float]) -> float:
    """E[f(S_T)] by quadrature over the standard normal Z."""
    loc = math.log(S0) + (MU - 0.5 * SIGMA**2) * T
    vol = SIGMA * math.sqrt(T)
    value, _ = integrate.quad(lambda z: f(math.exp(loc + vol * z)) * stats.norm.pdf(z),
                              -12, 12, points=[0.0], limit=400, epsabs=1e-13)
    return value


def _rho2_quadrature(payoff: Callable[[float], float]) -> float:
    ey = _expect(payoff)
    var_y = _expect(lambda s: payoff(s) ** 2) - ey**2
    var_x = _expect(lambda s: s * s) - EX**2
    cov = _expect(lambda s: s * payoff(s)) - EX * ey
    return cov**2 / (var_x * var_y)


def _assert_contract(result: dict) -> None:
    assert set(result) == KEYS
    for key in ("estimate", "standard_error", "beta", "rho2", "vrf"):
        value = result[key]
        assert value is None or (isinstance(value, float) and math.isfinite(value)), key
    assert result["standard_error"] is not None and result["standard_error"] >= 0
    assert result["beta"] is not None
    json.dumps(result, allow_nan=False)


ALL_MODES = [
    pytest.param({}, id="same_sample"),
    pytest.param({"pilot_fraction": 0.3}, id="pilot"),
    pytest.param({"beta": 1.0}, id="fixed"),
]


# --- references self-check --------------------------------------------------

def test_independent_references_agree():
    # closed form vs quadrature (both written here) for the call; the plan's
    # parecer gives VRF ~ 6.88 for this call and ~ 1.01 for the put K = 60.
    assert _call_mean_analytic() == pytest.approx(_expect(lambda s: max(s - 100.0, 0.0)),
                                                  rel=1e-9)
    rho2 = _call_rho2_analytic()
    assert rho2 == pytest.approx(_rho2_quadrature(lambda s: max(s - 100.0, 0.0)), rel=1e-8)
    assert 1 / (1 - rho2) == pytest.approx(6.88, abs=0.01)
    put_rho2 = _rho2_quadrature(lambda s: max(60.0 - s, 0.0))
    assert 1 / (1 - put_rho2) == pytest.approx(1.01, abs=0.01)


# --- degenerate cases -------------------------------------------------------

@pytest.mark.parametrize("kwargs", ALL_MODES)
def test_target_equal_to_control_recovers_mean_with_zero_se(kwargs):
    x = _terminal(np.random.default_rng(101), 50)
    with np.errstate(all="raise"):  # no division by zero, no invalid op
        result = control_variate_mean(x, x, EX, **kwargs)
    _assert_contract(result)
    assert result["estimate"] == pytest.approx(EX, rel=1e-12)
    assert result["standard_error"] <= 1e-12 * EX
    assert result["beta"] == pytest.approx(1.0, rel=1e-12)
    assert result["rho2"] == pytest.approx(1.0, abs=1e-12)
    assert result["warning"] is None
    # vrf is undefined (zero residual variance) or a huge meaningless number
    assert result["vrf"] is None or result["vrf"] > 1e10


def test_same_sample_target_equal_to_control_has_exactly_zero_residual():
    # documented: power-of-two rescaling makes the Y = X residual exactly 0
    x = _terminal(np.random.default_rng(102), 50)
    result = control_variate_mean(x, x, EX)
    assert result["standard_error"] == 0.0
    assert result["vrf"] is None
    assert result["rho2"] == 1.0


@pytest.mark.parametrize("kwargs", [pytest.param({}, id="same_sample"),
                                    pytest.param({"pilot_fraction": 0.5}, id="pilot")])
def test_zero_sigma_constant_control_gives_zero_beta_and_warning(kwargs):
    # sigma = 0: S_T is deterministic, so X and Y = (X - 100)+ are constant
    paths = simulate_gbm(S0, MU, 0.0, T, steps=4, n_sims=20, seed=5)
    x = paths[-1]
    assert np.ptp(x) == 0
    y = _call(x)
    with np.errstate(all="raise"):
        result = control_variate_mean(y, x, EX, **kwargs)
    _assert_contract(result)
    assert result["beta"] == 0.0
    assert result["warning"]
    assert result["rho2"] is None and result["vrf"] is None
    assert result["estimate"] == pytest.approx(float(y[0]), rel=1e-12)
    assert result["standard_error"] == 0.0


@pytest.mark.parametrize("kwargs", [pytest.param({}, id="same_sample"),
                                    pytest.param({"pilot_fraction": 0.5}, id="pilot")])
def test_constant_control_with_noisy_target_is_plain_mean(kwargs):
    rng = np.random.default_rng(6)
    x = np.full(40, 3.0)
    y = rng.normal(5.0, 2.0, size=40)
    result = control_variate_mean(y, x, 3.0, **kwargs)
    _assert_contract(result)
    assert result["beta"] == 0.0 and result["warning"]
    assert result["rho2"] is None and result["vrf"] is None
    main = y[result["n_pilot"]:]
    assert result["estimate"] == pytest.approx(float(np.mean(main)), rel=1e-12)
    # contract: same_sample keeps ddof = 2 even when beta falls back to 0
    ddof = 2 if result["beta_source"] == "same_sample" else 1
    assert result["standard_error"] == pytest.approx(
        float(np.std(main, ddof=ddof)) / math.sqrt(main.size), rel=1e-10)


def test_constant_control_with_fixed_beta_is_finite_and_warned():
    x = np.full(10, 3.0)
    y = np.arange(10.0)
    result = control_variate_mean(y, x, 2.0, beta=0.5)
    _assert_contract(result)
    assert result["warning"] and result["rho2"] is None and result["vrf"] is None
    assert result["estimate"] == pytest.approx(4.5 - 0.5 * (3.0 - 2.0), rel=1e-12)


def test_constant_target_has_undefined_rho2():
    x = _terminal(np.random.default_rng(7), 30)
    y = np.full(30, 7.0)
    result = control_variate_mean(y, x, EX)
    _assert_contract(result)
    assert result["rho2"] is None
    assert result["estimate"] == pytest.approx(7.0, rel=1e-12)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-12)


# --- law / value ------------------------------------------------------------

def test_call_estimate_vrf_and_se_against_analytic():
    """N = 200_000 call draws, seed 2202: same_sample and pilot (f = 0.1).

    VRF tolerance: across 30 independent replicates of this size (scratch
    study, seed 7) the measured VRF had standard deviation 0.034 around the
    theoretical 6.883; 0.2 is about 6 standard deviations, and the pilot
    VRF on the main sample is at most the same-sample value.
    """
    x = _terminal(np.random.default_rng(2202), 200_000)
    y = _call(x)
    truth = _call_mean_analytic()
    vrf_theory = 1 / (1 - _call_rho2_analytic())
    plain_se = float(np.std(y, ddof=1)) / math.sqrt(y.size)

    same = control_variate_mean(y, x, EX)
    pilot = control_variate_mean(y, x, EX, pilot_fraction=0.1)
    for result in (same, pilot):
        _assert_contract(result)
        assert abs(result["estimate"] - truth) <= K_SE * result["standard_error"]
        assert result["standard_error"] < plain_se
        assert result["vrf"] == pytest.approx(vrf_theory, abs=0.2)
        assert result["rho2"] == pytest.approx(1 - 1 / vrf_theory, abs=0.005)
    assert same["beta_source"] == "same_sample" and same["bias"] == "O(1/N)"
    assert same["n"] == 200_000 and same["n_pilot"] == 0
    assert pilot["beta_source"] == "pilot" and pilot["bias"] == "none"
    assert pilot["n_pilot"] == 20_000 and pilot["n"] == 180_000
    # identity in same_sample mode, upper bound in pilot mode
    assert same["vrf"] == pytest.approx(1 / (1 - same["rho2"]), rel=1e-9)
    assert pilot["vrf"] <= 1 / (1 - pilot["rho2"]) * (1 + 1e-12)
    # the SE ratio reflects the VRF: plain_se / se ~ sqrt(vrf)
    assert plain_se / same["standard_error"] == pytest.approx(math.sqrt(same["vrf"]),
                                                              rel=1e-3)


def test_call_with_production_sampler_terminal_row():
    paths = simulate_gbm(S0, MU, SIGMA, T, steps=4, n_sims=50_000, seed=31)
    x = paths[-1]
    y = _call(x)
    truth = _call_mean_analytic()
    for kwargs in ({}, {"pilot_fraction": 0.2}):
        result = control_variate_mean(y, x, gbm_terminal_control_mean(S0, MU, T), **kwargs)
        assert abs(result["estimate"] - truth) <= K_SE * result["standard_error"]


def test_deep_put_does_not_help():
    # parecer: put K = 60 has VRF ~ 1.01; measured SD of the VRF at this N is
    # ~5e-4 (30 replicates, scratch study), so vrf < 1.1 is a wide margin.
    x = _terminal(np.random.default_rng(60), 200_000)
    result = control_variate_mean(_put(x), x, EX)
    _assert_contract(result)
    assert 1.0 <= result["vrf"] < 1.1
    assert result["vrf"] == pytest.approx(1 / (1 - _rho2_quadrature(
        lambda s: max(60.0 - s, 0.0))), abs=0.005)


# --- standard error against independent computations -----------------------

def test_same_sample_se_matches_ols_with_intercept():
    rng = np.random.default_rng(303)
    x = _terminal(rng, 500)
    y = _call(x)
    result = control_variate_mean(y, x, EX)
    design = np.column_stack([np.ones_like(x), x])
    coef, ssr, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    assert rank == 2
    n = x.size
    se_ols = math.sqrt(float(ssr[0]) / (n - 2) / n)
    assert result["beta"] == pytest.approx(coef[1], rel=1e-10)
    assert result["standard_error"] == pytest.approx(se_ols, rel=1e-9)
    # regression prediction at E X equals the control-variate estimate
    assert result["estimate"] == pytest.approx(coef[0] + coef[1] * EX, rel=1e-10)
    r = np.corrcoef(x, y)[0, 1]
    assert result["rho2"] == pytest.approx(r * r, rel=1e-10)


@pytest.mark.parametrize("fraction", [0.1, 0.37, 0.5])
def test_pilot_se_matches_direct_computation(fraction):
    rng = np.random.default_rng(404)
    x = _terminal(rng, 999)
    y = _call(x)
    result = control_variate_mean(y, x, EX, pilot_fraction=fraction)
    n_pilot = math.floor(fraction * x.size)
    beta = np.cov(x[:n_pilot], y[:n_pilot], ddof=1)
    beta_hat = beta[0, 1] / beta[0, 0]
    xm, ym = x[n_pilot:], y[n_pilot:]
    e = ym - beta_hat * (xm - EX)
    assert result["n_pilot"] == n_pilot and result["n"] == x.size - n_pilot
    assert result["beta"] == pytest.approx(beta_hat, rel=1e-10)
    assert result["estimate"] == pytest.approx(float(np.mean(e)), rel=1e-12)
    assert result["standard_error"] == pytest.approx(
        float(np.std(e, ddof=1)) / math.sqrt(xm.size), rel=1e-10)
    assert result["vrf"] == pytest.approx(float(np.var(ym, ddof=1) / np.var(e, ddof=1)),
                                          rel=1e-10)
    r = np.corrcoef(xm, ym)[0, 1]
    assert result["rho2"] == pytest.approx(r * r, rel=1e-10)


def test_pilot_beta_depends_only_on_prefix():
    rng = np.random.default_rng(405)
    x = _terminal(rng, 100)
    y = _call(x)
    first = control_variate_mean(y, x, EX, pilot_fraction=0.25)
    y2 = y.copy()
    y2[25:] = y2[25:][::-1]  # reorder only the main part
    x2 = x.copy()
    x2[25:] = x2[25:][::-1]
    second = control_variate_mean(y2, x2, EX, pilot_fraction=0.25)
    assert second["beta"] == first["beta"]


def test_fixed_beta_se_matches_direct_computation():
    rng = np.random.default_rng(505)
    x = _terminal(rng, 300)
    y = _call(x)
    beta = 0.63
    result = control_variate_mean(y, x, EX, beta=beta)
    _assert_contract(result)
    e = y - beta * (x - EX)
    assert result["beta"] == beta
    assert result["beta_source"] == "fixed" and result["bias"] == "none"
    assert result["n"] == 300 and result["n_pilot"] == 0
    assert result["estimate"] == pytest.approx(float(np.mean(e)), rel=1e-12)
    assert result["standard_error"] == pytest.approx(
        float(np.std(e, ddof=1)) / math.sqrt(e.size), rel=1e-10)
    assert result["vrf"] == pytest.approx(float(np.var(y, ddof=1) / np.var(e, ddof=1)),
                                          rel=1e-10)


def test_fixed_poor_beta_can_increase_variance():
    x = _terminal(np.random.default_rng(506), 2_000)
    result = control_variate_mean(_call(x), x, EX, beta=-1.0)
    assert result["vrf"] < 1.0  # documented: a poor beta can make vrf < 1


# --- bias: same_sample O(1/N), pilot unbiased --------------------------------

def _replicate(n: int, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    same = np.empty(reps)
    pilot = np.empty(reps)
    for i in range(reps):
        x = _terminal(rng, n)
        y = _call(x)
        same[i] = control_variate_mean(y, x, EX)["estimate"]
        pilot[i] = control_variate_mean(y, x, EX, pilot_fraction=0.5)["estimate"]
    return same, pilot


def _leading_bias_times_n() -> float:
    """First-order bias of the same-sample estimator times N.

    beta_hat - beta ~ sum (X_i - EX) e_i / (N Var X), so
    E[theta_hat] - theta ~ -Cov(beta_hat, mean X) ~ -E[(X - EX)^2 e]/(N Var X),
    e = Y - EY - beta (X - EX). Computed by quadrature (about -7.79 here).
    """
    ey = _expect(lambda s: max(s - 100.0, 0.0))
    var_x = _expect(lambda s: s * s) - EX**2
    beta = (_expect(lambda s: s * max(s - 100.0, 0.0)) - EX * ey) / var_x
    third = _expect(lambda s: (s - EX) ** 2 * (max(s - 100.0, 0.0) - ey - beta * (s - EX)))
    return -third / var_x


def test_same_sample_bias_visible_at_small_n_pilot_unbiased():
    """N = 20, 4000 replicates, seed 777 (fast).

    The leading bias is about -7.79/20 = -0.39 while the replicate SE of the
    mean of estimates is about 0.023 (residual variance 34.8 / 20 / 4000), so
    the bias is ~17 SE: asserting |bias| > 4 SE has negligible miss rate. The
    pilot estimator must be within 4 SE of the analytic mean.
    """
    reps = 4000
    same, pilot = _replicate(20, reps, seed=777)
    truth = _call_mean_analytic()
    se_same = float(np.std(same, ddof=1)) / math.sqrt(reps)
    se_pilot = float(np.std(pilot, ddof=1)) / math.sqrt(reps)
    bias_same = float(np.mean(same)) - truth
    assert bias_same < -K_SE * se_same  # sign of the leading term (negative)
    assert abs(float(np.mean(pilot)) - truth) <= K_SE * se_pilot


@pytest.mark.slow
def test_same_sample_bias_scales_as_one_over_n():
    """N in {10, 20, 40}, 40_000 replicates each, seeds 900+N.

    N * bias must stay near the first-order constant c ~ -7.79: a scratch
    study (20_000 replicates) gave N * bias = -8.8, -8.8, -8.5, i.e. O(1/N^2)
    terms of ~13 % at N = 10. The band [1.5 c, 0.75 c] allows for those
    terms plus 4 N * SE (at most 0.4 here). Pilot (f = 0.5) stays unbiased
    at every N within 4 SE.
    """
    c = _leading_bias_times_n()
    assert c == pytest.approx(-7.79, abs=0.01)
    truth = _call_mean_analytic()
    reps = 40_000
    for n in (10, 20, 40):
        same, pilot = _replicate(n, reps, seed=900 + n)
        bias = float(np.mean(same)) - truth
        se = float(np.std(same, ddof=1)) / math.sqrt(reps)
        assert abs(bias) > K_SE * se
        assert 1.5 * c - K_SE * n * se <= n * bias <= 0.75 * c + K_SE * n * se
        se_pilot = float(np.std(pilot, ddof=1)) / math.sqrt(reps)
        assert abs(float(np.mean(pilot)) - truth) <= K_SE * se_pilot


# --- coverage of the normal interval ---------------------------------------

def _coverage(reps: int, seed: int, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    truth = _call_mean_analytic()
    hits_same = hits_pilot = 0
    for _ in range(reps):
        x = _terminal(rng, n)
        y = _call(x)
        same = control_variate_mean(y, x, EX)
        pilot = control_variate_mean(y, x, EX, pilot_fraction=0.2)
        hits_same += abs(same["estimate"] - truth) <= 1.96 * same["standard_error"]
        hits_pilot += abs(pilot["estimate"] - truth) <= 1.96 * pilot["standard_error"]
    return hits_same / reps, hits_pilot / reps


def _coverage_band(reps: int) -> float:
    # 4 binomial SDs of the empirical coverage plus 0.005 for the finite-N
    # skewness of the call payoff (scratch study: 0.9465 at N = 1000).
    return K_SE * math.sqrt(0.95 * 0.05 / reps) + 0.005


def test_normal_interval_coverage_fast():
    reps = 2000  # N = 1000 per replicate, seed 1960; band ~ +-0.025
    same, pilot = _coverage(reps, seed=1960)
    band = _coverage_band(reps)
    assert abs(same - 0.95) <= band
    assert abs(pilot - 0.95) <= band


@pytest.mark.slow
def test_normal_interval_coverage_slow():
    reps = 20_000  # N = 1000 per replicate, seed 1961; band ~ +-0.011
    same, pilot = _coverage(reps, seed=1961)
    band = _coverage_band(reps)
    assert abs(same - 0.95) <= band
    assert abs(pilot - 0.95) <= band


# --- scale invariance and extreme magnitudes --------------------------------

@pytest.mark.parametrize("factor", [2.0**-40, 2.0**7, 2.0**600, 3.0, 1e-7, 7.3e150])
@pytest.mark.parametrize("kwargs", [pytest.param({}, id="same_sample"),
                                    pytest.param({"pilot_fraction": 0.3}, id="pilot")])
def test_scaling_both_rescales_estimate_and_keeps_beta(factor, kwargs):
    x = _terminal(np.random.default_rng(808), 400)
    y = _call(x)
    base = control_variate_mean(y, x, EX, **kwargs)
    scaled = control_variate_mean(y * factor, x * factor, EX * factor, **kwargs)
    _assert_contract(scaled)
    assert scaled["estimate"] == pytest.approx(base["estimate"] * factor, rel=1e-12)
    assert scaled["standard_error"] == pytest.approx(base["standard_error"] * factor,
                                                     rel=1e-10)
    assert scaled["beta"] == pytest.approx(base["beta"], rel=1e-10)
    assert scaled["rho2"] == pytest.approx(base["rho2"], rel=1e-10)
    assert scaled["vrf"] == pytest.approx(base["vrf"], rel=1e-9)


def test_scaling_control_only_rescales_beta():
    x = _terminal(np.random.default_rng(809), 400)
    y = _call(x)
    base = control_variate_mean(y, x, EX)
    for c in (2.0**10, 3.0, 1e290):
        scaled = control_variate_mean(y, x * c, EX * c)
        assert scaled["estimate"] == pytest.approx(base["estimate"], rel=1e-12)
        assert scaled["standard_error"] == pytest.approx(base["standard_error"], rel=1e-10)
        assert scaled["beta"] == pytest.approx(base["beta"] / c, rel=1e-10)


def test_huge_values_stay_finite():
    # X ~ 1e300 * S_T/100 up to ~2e300: naive squares and sums overflow
    x = _terminal(np.random.default_rng(1300), 1000)
    y = _call(x)
    c = 1e298
    for kwargs in ({}, {"pilot_fraction": 0.5}, {"beta": 0.5}):
        base = control_variate_mean(y, x, EX, **kwargs)
        with np.errstate(over="raise"):
            big = control_variate_mean(y * c, x * c, EX * c, **kwargs)
        _assert_contract(big)
        assert big["estimate"] == pytest.approx(base["estimate"] * c, rel=1e-10)
        assert big["standard_error"] == pytest.approx(base["standard_error"] * c, rel=1e-9)


def test_beta_beyond_finite_precision_raises():
    x = np.linspace(1e300, 1.5e300, 10)
    y = np.linspace(1e-300, 2e-300, 10)
    with pytest.raises(ValueError):
        control_variate_mean(y, x, 1.2e300, beta=1e300)


# --- input validation -------------------------------------------------------

GOOD = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0])


@pytest.mark.parametrize("values, control", [
    pytest.param(GOOD, GOOD[:-1], id="length"),
    pytest.param(GOOD.reshape(2, 3), GOOD.reshape(2, 3), id="2d"),
    pytest.param(np.array(1.0), np.array(1.0), id="0d"),
    pytest.param(np.r_[GOOD[:-1], np.nan], GOOD, id="nan-values"),
    pytest.param(GOOD, np.r_[GOOD[:-1], np.inf], id="inf-control"),
    pytest.param(GOOD, np.r_[GOOD[:-1], -np.inf], id="minus-inf-control"),
    pytest.param(GOOD > 2, GOOD, id="bool-values"),
    pytest.param(GOOD, GOOD > 2, id="bool-control"),
    pytest.param(GOOD + 0j, GOOD, id="complex-values"),
    pytest.param(GOOD, GOOD + 1j, id="complex-control"),
    pytest.param(["a"] * 6, GOOD, id="strings"),
    pytest.param(np.array([], dtype=float), np.array([], dtype=float), id="empty"),
])
def test_invalid_arrays_raise(values, control):
    with pytest.raises(ValueError):
        control_variate_mean(values, control, 1.0)


@pytest.mark.parametrize("mean", [math.nan, math.inf, -math.inf, True, 1 + 0j, "1",
                                  None, np.array([1.0, 2.0])])
def test_invalid_control_mean_raises(mean):
    with pytest.raises(ValueError):
        control_variate_mean(GOOD, GOOD, mean)


def test_beta_and_pilot_fraction_are_exclusive():
    with pytest.raises(ValueError):
        control_variate_mean(GOOD, GOOD, 1.0, beta=1.0, pilot_fraction=0.5)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, True, "0.5"])
def test_invalid_pilot_fraction_raises(fraction):
    with pytest.raises(ValueError):
        control_variate_mean(GOOD, GOOD, 1.0, pilot_fraction=fraction)


@pytest.mark.parametrize("beta", [math.nan, math.inf, True, "1", 1j])
def test_invalid_fixed_beta_raises(beta):
    with pytest.raises(ValueError):
        control_variate_mean(GOOD, GOOD, 1.0, beta=beta)


@pytest.mark.parametrize("n, kwargs", [
    pytest.param(2, {}, id="same_sample-N2"),
    pytest.param(1, {"beta": 1.0}, id="fixed-N1"),
    pytest.param(5, {"pilot_fraction": 0.3}, id="pilot-1-of-5"),
    pytest.param(5, {"pilot_fraction": 0.8}, id="main-1-of-5"),
    pytest.param(3, {"pilot_fraction": 0.5}, id="pilot-1-of-3"),
])
def test_too_small_samples_raise(n, kwargs):
    data = np.arange(1.0, n + 1.0)
    with pytest.raises(ValueError):
        control_variate_mean(data, data**2, 2.0, **kwargs)


@pytest.mark.parametrize("n, kwargs, n_pilot", [
    pytest.param(3, {}, 0, id="same_sample-N3"),
    pytest.param(2, {"beta": 0.0}, 0, id="fixed-N2"),
    pytest.param(4, {"pilot_fraction": 0.5}, 2, id="pilot-2-2"),
    pytest.param(5, {"pilot_fraction": 0.7}, 3, id="pilot-floor"),
])
def test_minimum_sizes_accepted(n, kwargs, n_pilot):
    data = np.arange(1.0, n + 1.0)
    result = control_variate_mean(data, data**2 + np.r_[0.5, np.zeros(n - 1)], 2.0, **kwargs)
    _assert_contract(result)
    assert result["n_pilot"] == n_pilot and result["n"] == n - n_pilot


def test_integer_inputs_and_lists_accepted():
    a = control_variate_mean([1, 2, 4, 3, 5], [2, 1, 3, 5, 4], 3)
    b = control_variate_mean(np.array([1.0, 2, 4, 3, 5]), np.array([2.0, 1, 3, 5, 4]), 3.0)
    assert a == b


def test_inputs_not_modified():
    x = _terminal(np.random.default_rng(9), 50)
    y = _call(x)
    x0, y0 = x.copy(), y.copy()
    control_variate_mean(y, x, EX, pilot_fraction=0.4)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


# --- gbm_terminal_control_mean ----------------------------------------------

@pytest.mark.parametrize("s0, mu, horizon", [
    (100.0, 0.05, 1.0), (1.0, -0.3, 2.5), (1e-3, 0.0, 10.0), (5e4, 0.2, 0.0), (7, 1, 3),
])
def test_terminal_control_mean_matches_formula(s0, mu, horizon):
    assert gbm_terminal_control_mean(s0, mu, horizon) == pytest.approx(
        s0 * math.exp(mu * horizon), rel=1e-14)


def test_terminal_control_mean_extreme_but_finite():
    value = gbm_terminal_control_mean(1e300, 1.0, 10.0)  # exp(log 1e300 + 10) ~ 2.2e304
    assert math.isfinite(value)
    assert value == pytest.approx(math.exp(math.log(1e300) + 10.0), rel=1e-12)


@pytest.mark.parametrize("s0, mu, horizon", [
    pytest.param(1e300, 100.0, 10.0, id="overflow"),
    pytest.param(1e-300, -100.0, 10.0, id="underflow"),
    pytest.param(1.0, 1e308, 1e308, id="inf-exponent"),
    pytest.param(0.0, 0.05, 1.0, id="s0-zero"),
    pytest.param(-1.0, 0.05, 1.0, id="s0-negative"),
    pytest.param(math.nan, 0.05, 1.0, id="s0-nan"),
    pytest.param(True, 0.05, 1.0, id="s0-bool"),
    pytest.param(100.0, math.inf, 1.0, id="mu-inf"),
    pytest.param(100.0, True, 1.0, id="mu-bool"),
    pytest.param(100.0, 0.05, -1.0, id="horizon-negative"),
    pytest.param(100.0, 0.05, math.nan, id="horizon-nan"),
    pytest.param(100.0, 0.05, "1", id="horizon-str"),
])
def test_terminal_control_mean_invalid_raises(s0, mu, horizon):
    with pytest.raises(ValueError):
        gbm_terminal_control_mean(s0, mu, horizon)
