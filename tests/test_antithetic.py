"""Tests for antithetic variates in the normal GBM sampler (E2.1, offline, seeded).

Model (constant parameters, exact lognormal increments)::

    S(T) = S0 * exp((mu - sigma^2/2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
    E[S_T]   = S0 * exp(mu*T)
    Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)

Antithetic pair (Z, -Z), written here from the plan (E2.1) and independent of
``simulation.variance_reduction``::

    Cov(S+, S-) = S0^2 * exp(2*(mu - sigma^2/2)*T) - (E S_T)^2
    Var_pair    = Var(Y) = (Var + Cov) / 2,      Y = (S+ + S-) / 2
    VRF         = Var_plain / (2 * Var_pair)

Units: prices in currency, T in years, mu in 1/year, sigma in 1/sqrt(year).

Tolerances follow docs/plano-expansao.md section 2.5: Monte Carlo means are
checked against theory at 4 standard errors (false alarm about 6e-5 per
assertion under asymptotic normality); other tolerances are justified inline.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from scipy import integrate, stats

from simulation.gbm import GbmSimulator, simulate_gbm, simulate_terminal_batched
from simulation.variance_reduction import (
    antithetic_mean,
    antithetic_pair_covariance,
    antithetic_terminal_vrf,
)

S0, MU, T = 100.0, 0.05, 1.0
K_SE = 4.0


def _expected_terminal(s0: float = S0, mu: float = MU, horizon: float = T) -> float:
    return s0 * math.exp(mu * horizon)


def _plain_variance(s0: float, mu: float, sigma: float, horizon: float) -> float:
    return s0**2 * math.exp(2 * mu * horizon) * (math.exp(sigma**2 * horizon) - 1)


def _pair_covariance_direct(s0: float, mu: float, sigma: float, horizon: float) -> float:
    """Plan formula: Cov(S+, S-) = S0^2 e^{2(mu - sigma^2/2)T} - (E S_T)^2."""
    expected = s0 * math.exp(mu * horizon)
    return s0**2 * math.exp(2 * (mu - sigma**2 / 2) * horizon) - expected**2


def _vrf_direct(s0: float, mu: float, sigma: float, horizon: float) -> float:
    var = _plain_variance(s0, mu, sigma, horizon)
    var_pair = (var + _pair_covariance_direct(s0, mu, sigma, horizon)) / 2
    return var / (2 * var_pair)


def _empirical_vrf(values: np.ndarray) -> float:
    """Var of one draw (the antithetic sample is a valid sample of the law)
    over twice the variance of the pair average Y."""
    m = values.size // 2
    pair = 0.5 * (values[:m] + values[m:])
    return float(np.var(values, ddof=1) / (2 * np.var(pair, ddof=1)))


def _antithetic_terminal(sigma: float, n_sims: int, seed: int, **kwargs) -> np.ndarray:
    return simulate_terminal_batched(S0, MU, sigma, T, n_sims, seed,
                                     variance_reduction="antithetic", **kwargs)


# ---------------------------------------------------------------------------
# (a) Unbiased terminal mean with the pair-based standard error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.2, 0.5, 1.0])
def test_terminal_mean_within_4_se_of_analytic(sigma):
    out = antithetic_mean(_antithetic_terminal(sigma, 2 * 20_000, seed=101))
    assert out["pairs"] == 20_000
    assert out["standard_error"] > 0
    assert abs(out["estimate"] - _expected_terminal()) <= K_SE * out["standard_error"]


def test_path_terminal_row_mean_within_4_se_of_analytic():
    paths = simulate_gbm(S0, MU, 0.3, T, 12, 2 * 10_000, seed=202,
                         variance_reduction="antithetic")
    out = antithetic_mean(paths[-1])
    assert abs(out["estimate"] - _expected_terminal()) <= K_SE * out["standard_error"]


def test_antithetic_originals_follow_exact_terminal_law():
    # The first half is an i.i.d. sample of the exact law; KS with a fixed
    # seed and minimum p-value 1e-3 (plan section 2.5).
    sigma, m = 0.4, 5_000
    values = _antithetic_terminal(sigma, 2 * m, seed=303)
    law = stats.lognorm(s=sigma * math.sqrt(T),
                        scale=S0 * math.exp((MU - sigma**2 / 2) * T))
    assert stats.kstest(values[:m], law.cdf).pvalue > 1e-3
    assert stats.kstest(values[m:], law.cdf).pvalue > 1e-3


# ---------------------------------------------------------------------------
# (b) Variance reduction factor: theory and measurement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma,expected", [(0.2, 25.50), (0.25, 16.51),
                                            (0.5, 4.521), (1.0, 1.582)])
def test_theoretical_vrf_matches_direct_formula_and_plan_table(sigma, expected):
    direct = _vrf_direct(S0, MU, sigma, T)
    assert direct == pytest.approx(expected, abs=0.006)
    # Cancellation in the direct covariance costs at most ~2 digits here.
    assert antithetic_terminal_vrf(sigma, T) == pytest.approx(direct, rel=1e-10)
    assert antithetic_terminal_vrf(sigma, T) > 1


@pytest.mark.parametrize("s0,mu,sigma,horizon", [
    (100.0, 0.05, 0.2, 1.0), (50.0, -0.1, 0.5, 2.0), (1.0, 0.0, 1.0, 0.25),
    (250.0, 0.3, 0.05, 3.0),
])
def test_pair_covariance_matches_direct_formula(s0, mu, sigma, horizon):
    direct = _pair_covariance_direct(s0, mu, sigma, horizon)
    got = antithetic_pair_covariance(s0, mu, sigma, horizon)
    assert got < 0
    # The direct form subtracts two O(S0^2) terms; relative cancellation is
    # at most 1/v (v >= 0.0075 here), so 1e-10 is far above its rounding.
    assert got == pytest.approx(direct, rel=1e-10)
    # VRF independent of s0 and mu.
    assert antithetic_terminal_vrf(sigma, horizon) == pytest.approx(
        _vrf_direct(s0, mu, sigma, horizon), rel=1e-10)


def test_degenerate_volatility_theory():
    assert antithetic_pair_covariance(S0, MU, 0.0, T) == 0.0
    assert antithetic_terminal_vrf(0.0, T) == math.inf
    # Tiny but positive v: finite, huge, never nan.
    vrf = antithetic_terminal_vrf(1e-9, 1e-9)
    assert math.isfinite(vrf) or vrf == math.inf
    assert vrf > 1e20


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, True, "0.2", None])
def test_theory_helpers_reject_invalid_inputs(bad):
    with pytest.raises(ValueError):
        antithetic_terminal_vrf(bad, T)
    with pytest.raises(ValueError):
        antithetic_terminal_vrf(0.2, bad)
    with pytest.raises(ValueError):
        antithetic_pair_covariance(S0, MU, bad, T)


@pytest.mark.parametrize("bad_s0", [0.0, -1.0, math.nan, True])
def test_pair_covariance_rejects_invalid_s0(bad_s0):
    with pytest.raises(ValueError):
        antithetic_pair_covariance(bad_s0, MU, 0.2, T)


@pytest.mark.parametrize("sigma,seed", [(0.2, 404), (0.5, 405)])
def test_measured_vrf_within_20_percent_of_theory(sigma, seed):
    # sigma <= 0.5 and m = 2^14: the ratio measured/theory had 0.1%..99.9%
    # quantiles [0.94, 1.05] over 400 seeds (pre-study), well inside +-20%.
    # sigma = 1 with m = 4096 fails ~18% of seeds (heavy lognormal tail).
    m = 2**14
    values = _antithetic_terminal(sigma, 2 * m, seed=seed)
    measured = _empirical_vrf(values)
    theory = _vrf_direct(S0, MU, sigma, T)
    assert measured > 1
    assert measured == pytest.approx(theory, rel=0.20)


def test_plain_sampling_has_no_variance_reduction():
    # Control: under "none" the same statistic estimates VRF = 1.
    values = simulate_terminal_batched(S0, MU, 0.5, T, 2**15, 406)
    assert _empirical_vrf(values) == pytest.approx(1.0, rel=0.10)


def test_pair_se_is_smaller_than_naive_iid_se_for_monotone_payoff():
    sigma, m = 0.3, 10_000
    anti = antithetic_mean(_antithetic_terminal(sigma, 2 * m, seed=505))
    plain = simulate_terminal_batched(S0, MU, sigma, T, 2 * m, 506)
    naive_se = float(np.std(plain, ddof=1) / math.sqrt(2 * m))
    # Theory: SE ratio sqrt(VRF) ~ 3.5 for sigma = 0.3; demand only a factor 2.
    assert anti["standard_error"] * 2 < naive_se


# ---------------------------------------------------------------------------
# (g) A case where antithetics hurt: the straddle |S_T - K| at the median
# ---------------------------------------------------------------------------

def _straddle_vrf_quadrature(sigma: float) -> float:
    """VRF of f(Z) = K |e^{bZ} - 1| (K = median, b = sigma sqrt T) by quadrature."""
    b = sigma * math.sqrt(T)
    phi = stats.norm.pdf

    def f(z: float) -> float:
        return abs(math.expm1(b * z))

    def expect(g) -> float:
        # Split at the kink z = 0 of |e^{bz} - 1|; |z| <= 12 drops normal
        # mass below 1e-32 and avoids overflow of e^{2bz} in quadrature.
        return (integrate.quad(lambda z: g(z) * phi(z), -12.0, 0.0)[0]
                + integrate.quad(lambda z: g(z) * phi(z), 0.0, 12.0)[0])

    m1 = expect(f)
    m2 = expect(lambda z: f(z) ** 2)
    cross = expect(lambda z: f(z) * f(-z))
    var, cov = m2 - m1**2, cross - m1**2
    return var / (var + cov)


def test_antithetic_hurts_even_straddle():
    sigma, m = 0.2, 2**14
    theory = _straddle_vrf_quadrature(sigma)
    assert 0.5 < theory < 0.6  # limit 0.5 as the payoff becomes even in Z
    median = S0 * math.exp((MU - sigma**2 / 2) * T)
    payoff = np.abs(_antithetic_terminal(sigma, 2 * m, seed=606) - median)
    measured = _empirical_vrf(payoff)
    assert measured < 1
    # Over 400 seeds the measured VRF stayed in [0.538, 0.542]; +-10% is
    # many sampling SDs wide yet still excludes VRF >= 1.
    assert measured == pytest.approx(theory, rel=0.10)


# ---------------------------------------------------------------------------
# (c) Matrix contract and pair ordering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("steps,n_sims", [(1, 2), (5, 8), (37, 300)])
def test_paths_shape_row_zero_and_positive(steps, n_sims):
    paths = simulate_gbm(S0, MU, 0.3, T, steps, n_sims, seed=7,
                         variance_reduction="antithetic")
    assert paths.shape == (steps + 1, n_sims)
    assert paths.dtype == np.float64
    np.testing.assert_array_equal(paths[0], np.full(n_sims, S0))
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)


@pytest.mark.parametrize("sigma,steps,horizon", [(0.3, 50, 1.0), (1.2, 7, 3.0),
                                                 (0.2, 3, 1e-6)])
def test_paths_mirror_columns_are_negated_shocks(sigma, steps, horizon):
    n_sims = 400
    m = n_sims // 2
    paths = simulate_gbm(S0, MU, sigma, horizon, steps, n_sims, seed=8,
                         variance_reduction="antithetic")
    k = np.arange(steps + 1)[:, None]
    drift = (MU - sigma**2 / 2) * horizon / steps
    centred = np.log(paths / S0) - k * drift
    scale = float(np.max(np.abs(centred)))
    # Drift is added before the cumulative sum, so the mirror is not
    # bitwise -Z; exp/log rounding leaves O(1e-15) absolute error.
    np.testing.assert_allclose(centred[:, m:], -centred[:, :m],
                               rtol=1e-12, atol=1e-12 * max(scale, 1e-300))
    # Mirrors differ from originals (not a duplicated half).
    assert not np.allclose(centred[1:, m:], centred[1:, :m])


def test_terminal_mirror_order():
    sigma, m = 0.4, 1_000
    values = _antithetic_terminal(sigma, 2 * m, seed=9)
    centred = np.log(values / S0) - (MU - sigma**2 / 2) * T
    np.testing.assert_allclose(centred[m:], -centred[:m], rtol=1e-12,
                               atol=1e-12 * float(np.max(np.abs(centred))))


# ---------------------------------------------------------------------------
# (d) Rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_sims", [1, 3, 101])
def test_odd_n_sims_rejected_everywhere(n_sims):
    sim = GbmSimulator(S0, T, 4, n_sims, seed=1, mu=MU, sigma=0.2)
    with pytest.raises(ValueError, match="even"):
        sim.generate_paths(variance_reduction="antithetic")
    with pytest.raises(ValueError, match="even"):
        sim.generate_terminal(variance_reduction="antithetic")
    with pytest.raises(ValueError, match="even"):
        simulate_gbm(S0, MU, 0.2, T, 4, n_sims, seed=1, variance_reduction="antithetic")
    with pytest.raises(ValueError, match="even"):
        simulate_terminal_batched(S0, MU, 0.2, T, n_sims, seed=1,
                                  variance_reduction="antithetic")


def test_t_student_with_antithetic_rejected():
    sim = GbmSimulator(S0, T, 4, 6, seed=1, mu=MU, sigma=0.2, df=5.0)
    with pytest.raises(ValueError):
        sim.generate_paths("t_student", variance_reduction="antithetic")
    # The t scenario itself still works without antithetics.
    assert sim.generate_paths("t_student").shape == (5, 6)


@pytest.mark.parametrize("bad", ["control", "qmc", "Antithetic", "", None, 1, True])
def test_invalid_variance_reduction_rejected(bad):
    sim = GbmSimulator(S0, T, 4, 6, seed=1, mu=MU, sigma=0.2)
    with pytest.raises(ValueError):
        sim.generate_paths(variance_reduction=bad)
    with pytest.raises(ValueError):
        sim.generate_terminal(variance_reduction=bad)
    with pytest.raises(ValueError):
        simulate_gbm(S0, MU, 0.2, T, 4, 6, seed=1, variance_reduction=bad)
    with pytest.raises(ValueError):
        simulate_terminal_batched(S0, MU, 0.2, T, 6, seed=1, variance_reduction=bad)


def test_variance_reduction_is_keyword_only():
    sim = GbmSimulator(S0, T, 4, 6, seed=1, mu=MU, sigma=0.2)
    with pytest.raises(TypeError):
        sim.generate_paths("normal", "antithetic")


# ---------------------------------------------------------------------------
# (e) Default "none" unchanged and reproducibility
# ---------------------------------------------------------------------------

def test_default_equals_explicit_none_and_is_reproducible():
    args = (S0, MU, 0.25, T, 6, 9)
    default = simulate_gbm(*args, seed=42)
    np.testing.assert_array_equal(default, simulate_gbm(*args, seed=42, variance_reduction="none"))
    np.testing.assert_array_equal(default, simulate_gbm(*args, seed=42))
    term = simulate_terminal_batched(S0, MU, 0.25, T, 9, 42)
    np.testing.assert_array_equal(
        term, simulate_terminal_batched(S0, MU, 0.25, T, 9, 42, variance_reduction="none"))


def test_none_mode_matches_independent_reconstruction():
    # One C-order (steps, n_sims) normal draw from default_rng(seed), as
    # documented; reconstructed here with the closed-form GBM on the grid.
    sigma, steps, n_sims, seed = 0.25, 6, 9, 42
    z = np.random.default_rng(seed).standard_normal((steps, n_sims))
    dt = T / steps
    logs = np.cumsum((MU - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * z, axis=0)
    expected = np.vstack([np.full(n_sims, S0), S0 * np.exp(logs)])
    np.testing.assert_allclose(simulate_gbm(S0, MU, sigma, T, steps, n_sims, seed=seed),
                               expected, rtol=1e-12)


def test_antithetic_is_reproducible_for_same_seed():
    first = simulate_gbm(S0, MU, 0.3, T, 5, 10, seed=11, variance_reduction="antithetic")
    second = simulate_gbm(S0, MU, 0.3, T, 5, 10, seed=11, variance_reduction="antithetic")
    np.testing.assert_array_equal(first, second)
    t1 = _antithetic_terminal(0.3, 10, seed=11)
    np.testing.assert_array_equal(t1, _antithetic_terminal(0.3, 10, seed=11))


def test_antithetic_originals_use_documented_shock_draw():
    # Documented: the (steps, m) shocks equal one C-order draw from the
    # instance RNG; reconstructed independently with rtol (not bitwise).
    sigma, steps, m, seed = 0.3, 4, 5, 13
    z = np.random.default_rng(seed).standard_normal((steps, m))
    dt = T / steps
    increments = (MU - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * np.hstack([z, -z])
    expected = np.vstack([np.full(2 * m, S0), S0 * np.exp(np.cumsum(increments, axis=0))])
    paths = simulate_gbm(S0, MU, sigma, T, steps, 2 * m, seed=seed,
                         variance_reduction="antithetic")
    np.testing.assert_allclose(paths, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# (f) antithetic_mean
# ---------------------------------------------------------------------------

def test_antithetic_mean_matches_direct_pair_computation():
    values = np.random.default_rng(21).normal(size=2 * 37)
    m = 37
    y = (values[:m] + values[m:]) / 2
    out = antithetic_mean(values)
    assert out["pairs"] == m
    assert out["estimate"] == pytest.approx(float(np.mean(y)), rel=1e-12, abs=1e-15)
    assert out["standard_error"] == pytest.approx(
        float(np.std(y, ddof=1)) / math.sqrt(m), rel=1e-12)


def test_antithetic_mean_hand_example():
    # pairs (1, 3) -> 2 and (2, 6) -> 4: mean 3, std(ddof=1) sqrt(2), SE 1.
    out = antithetic_mean([1.0, 2.0, 3.0, 6.0])
    assert out == {"estimate": 3.0, "standard_error": pytest.approx(1.0), "pairs": 2}


def test_antithetic_mean_single_pair_has_no_se():
    out = antithetic_mean(np.array([1.0, 5.0]))
    assert out["estimate"] == 3.0
    assert out["standard_error"] is None
    assert out["pairs"] == 1


def test_antithetic_mean_integer_input():
    out = antithetic_mean(np.array([1, 2, 3, 6]))
    assert out["estimate"] == 3.0


def test_antithetic_mean_near_overflow_stays_finite():
    big = np.finfo(np.float64).max
    out = antithetic_mean(np.array([big, big * 0.5, big, big * 0.5]))
    assert math.isfinite(out["estimate"]) and out["estimate"] == pytest.approx(0.75 * big)
    assert out["standard_error"] is not None and math.isfinite(out["standard_error"])


@pytest.mark.parametrize("bad", [
    [], [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]],
    [1.0, math.nan], [1.0, math.inf], [True, False], ["a", "b"],
    np.array([1 + 1j, 2 + 0j]),
])
def test_antithetic_mean_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        antithetic_mean(bad)


# ---------------------------------------------------------------------------
# (h) Degenerate configurations and memory
# ---------------------------------------------------------------------------

def test_zero_volatility_is_deterministic_with_zero_se():
    paths = simulate_gbm(S0, MU, 0.0, T, 4, 6, seed=3, variance_reduction="antithetic")
    expected = S0 * np.exp(MU * np.arange(5) * T / 4)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape),
                               rtol=1e-14)
    terminal = _antithetic_terminal(0.0, 6, seed=3)
    np.testing.assert_allclose(terminal, _expected_terminal(), rtol=1e-14)
    out = antithetic_mean(terminal)
    assert out["standard_error"] == 0.0
    assert out["estimate"] == pytest.approx(_expected_terminal(), rel=1e-14)


def test_smallest_case_two_sims_one_step():
    paths = simulate_gbm(S0, MU, 0.2, T, 1, 2, seed=4, variance_reduction="antithetic")
    assert paths.shape == (2, 2)
    drift = (MU - 0.02) * T
    up, down = np.log(paths[1] / S0) - drift
    assert up == pytest.approx(-down, rel=1e-12)
    terminal = _antithetic_terminal(0.2, 2, seed=4)
    assert terminal.shape == (2,)
    assert antithetic_mean(terminal)["standard_error"] is None


def test_tiny_horizon_stays_near_s0():
    terminal = simulate_terminal_batched(S0, MU, 0.2, 1e-12, 1_000, 5,
                                         variance_reduction="antithetic")
    assert np.all(np.isfinite(terminal)) and np.all(terminal > 0)
    np.testing.assert_allclose(terminal, S0, rtol=1e-5)


def test_antithetic_terminal_respects_memory_budget_check():
    with pytest.raises(ValueError, match="bytes"):
        simulate_terminal_batched(S0, MU, 0.2, T, 1_000, 1, max_working_bytes=8 * 999,
                                  variance_reduction="antithetic")


def test_no_extra_matrix_allocated():
    # The output is the only full-size array: peak traced memory stays
    # within 10% of it (measured ~1.016 for paths, ~1.0005 for terminal).
    sim = GbmSimulator(S0, T, 100, 2_000, seed=6, mu=MU, sigma=0.2)
    tracemalloc.start()
    try:
        paths = sim.generate_paths(variance_reduction="antithetic")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.10 * paths.nbytes

    sim = GbmSimulator(S0, T, 1, 200_000, seed=6, mu=MU, sigma=0.2)
    tracemalloc.start()
    try:
        terminal = sim.generate_terminal(variance_reduction="antithetic", batch_size=1_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.10 * terminal.nbytes


# ---------------------------------------------------------------------------
# Empirical coverage of the pair-based normal interval
# ---------------------------------------------------------------------------

def _pair_interval_coverage(n_reps: int, m: int, seed: int, sigma: float = 0.2) -> float:
    """Replicate r uses pairs r*m..(r+1)*m-1 of one antithetic stream:
    disjoint pairs, hence independent replicates."""
    values = _antithetic_terminal(sigma, 2 * n_reps * m, seed=seed)
    half = n_reps * m
    originals = values[:half].reshape(n_reps, m)
    mirrors = values[half:].reshape(n_reps, m)
    target = _expected_terminal()
    z = stats.norm.isf(0.025)
    hits = 0
    for orig, mirr in zip(originals, mirrors):
        out = antithetic_mean(np.concatenate([orig, mirr]))
        hits += abs(out["estimate"] - target) <= z * out["standard_error"]
    return hits / n_reps


def test_pair_interval_coverage_quick():
    # R = 500 replicates: binomial SD of the coverage at 0.95 is 0.0097;
    # 4 SD gives [0.911, 0.989].
    coverage = _pair_interval_coverage(500, 512, seed=707)
    assert 0.911 <= coverage <= 0.989


@pytest.mark.slow
def test_pair_interval_coverage_full():
    # R = 4000: SD 0.00345, 4 SD around the nominal 0.95 gives [0.936, 0.964].
    # The normal interval with m = 512 skewed pairs under-covers slightly:
    # 15 other seeds x 4000 replicates averaged 0.945 (SE ~0.001), all in band.
    coverage = _pair_interval_coverage(4000, 512, seed=708)
    assert 0.936 <= coverage <= 0.964
