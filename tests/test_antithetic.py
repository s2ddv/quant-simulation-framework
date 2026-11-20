"""Offline checks for antithetic GBM sampling and pair-based estimators.

Reference (constant parameters, T in years):
    E[S(T)]   = s0*exp(mu*T)
    Var[S(T)] = s0**2*exp(2*mu*T)*(exp(sigma**2*T) - 1)
    Cov[S(Z), S(-Z)] = s0**2*exp(2*mu*T)*(exp(-sigma**2*T) - 1)
so, for the same number of draws, the variance of the antithetic mean
estimator relative to plain Monte Carlo is 1 + Cov/Var = 1 - exp(-sigma**2*T).

Memory: the largest array below is 200_000 float64 terminal prices (1.6 MB);
the path test uses (65, 4000) plus a (64, 2000) temporary (about 3.1 MB).
"""

from statistics import NormalDist

import numpy as np
import pytest

from simulation.gbm import GbmSimulator, simulate_gbm, simulate_terminal_batched
from simulation.variance_reduction import (
    antithetic_mean_estimate,
    antithetic_pair_means,
    plain_mean_estimate,
)

S0, MU, SIGMA, T = 100.0, 0.08, 0.2, 1.0


def analytic_mean(t=T):
    return S0 * np.exp(MU * t)


def analytic_var(t=T):
    return S0**2 * np.exp(2 * MU * t) * np.expm1(SIGMA**2 * t)


def test_default_paths_and_terminal_are_unchanged_by_the_option():
    np.testing.assert_array_equal(
        simulate_gbm(S0, MU, SIGMA, T, 8, 11, seed=3),
        simulate_gbm(S0, MU, SIGMA, T, 8, 11, seed=3, antithetic=False),
    )
    np.testing.assert_array_equal(
        simulate_terminal_batched(S0, MU, SIGMA, T, 11, seed=3, batch_size=4),
        simulate_terminal_batched(S0, MU, SIGMA, T, 11, seed=3, batch_size=4,
                                  antithetic=False),
    )


def test_paths_shape_row_zero_and_mirrored_log_returns():
    steps, n = 6, 10
    paths = simulate_gbm(S0, MU, SIGMA, T, steps, n, seed=5, antithetic=True)
    assert paths.shape == (steps + 1, n)
    assert np.all(paths[0] == S0)
    assert np.all(paths > 0)
    # log(S_i/s0) + log(S_{i+m}/s0) = 2*(mu - sigma^2/2)*t at every grid time.
    m = n // 2
    times = np.arange(steps + 1) * T / steps
    pair_sum = np.log(paths[:, :m] / S0) + np.log(paths[:, m:] / S0)
    expected = np.broadcast_to((2 * (MU - SIGMA**2 / 2) * times)[:, None], pair_sum.shape)
    np.testing.assert_allclose(pair_sum, expected, atol=1e-13)
    # Reproducible: the first half uses one (steps, m) draw from the seed.
    z = np.random.default_rng(5).standard_normal((steps, m))
    dt = T / steps
    log_paths = np.cumsum((MU - SIGMA**2 / 2) * dt + SIGMA * np.sqrt(dt) * z, axis=0)
    np.testing.assert_allclose(paths[1:, :m], S0 * np.exp(log_paths), rtol=1e-14)
    np.testing.assert_allclose(paths[1:, m:], S0 * np.exp(
        np.cumsum((MU - SIGMA**2 / 2) * dt - SIGMA * np.sqrt(dt) * z, axis=0)),
        rtol=1e-14)


@pytest.mark.parametrize("batch_size", [1, 3, 4, 100])
def test_terminal_pairs_are_seeded_and_batch_independent(batch_size):
    n, m = 8, 4
    terminal = simulate_terminal_batched(S0, MU, SIGMA, T, n, seed=21,
                                         batch_size=batch_size, antithetic=True)
    z = np.random.default_rng(21).standard_normal(m)
    drift = (MU - SIGMA**2 / 2) * T
    np.testing.assert_allclose(terminal[:m], S0 * np.exp(drift + SIGMA * np.sqrt(T) * z),
                               rtol=2e-15)
    np.testing.assert_allclose(terminal[m:], S0 * np.exp(drift - SIGMA * np.sqrt(T) * z),
                               rtol=2e-15)


def test_invalid_antithetic_requests_raise():
    with pytest.raises(ValueError, match="even n_sims"):
        simulate_gbm(S0, MU, SIGMA, T, 4, 7, seed=1, antithetic=True)
    with pytest.raises(ValueError, match="even n_sims"):
        simulate_terminal_batched(S0, MU, SIGMA, T, 7, seed=1, antithetic=True)
    model = GbmSimulator(S0, T, 4, 6, seed=1, mu=MU, sigma=SIGMA)
    model.calibrated_df = 5.0
    with pytest.raises(ValueError, match="normal GBM"):
        model.generate_paths("t_student", antithetic=True)
    with pytest.raises(ValueError, match="bool"):
        model.generate_paths(antithetic=1)
    with pytest.raises(ValueError, match="bool"):
        model.generate_terminal(antithetic="yes")
    with pytest.raises(ValueError):
        antithetic_pair_means(np.ones(5))
    with pytest.raises(ValueError):
        antithetic_mean_estimate(np.ones(2), confidence=1.0)


def test_zero_volatility_antithetic_is_deterministic():
    paths = simulate_gbm(S0, MU, 0.0, T, 4, 4, seed=0, antithetic=True)
    expected = S0 * np.exp(MU * np.arange(5) / 4)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape))


def test_terminal_moments_and_quantiles_match_analytic_law():
    n = 200_000
    terminal = simulate_terminal_batched(S0, MU, SIGMA, T, n, seed=2024,
                                         batch_size=7_919, antithetic=True)
    estimate = antithetic_mean_estimate(terminal)
    assert estimate.n_independent == n // 2
    # Mean: within 4 pair-based standard errors (fixed seed).
    assert abs(estimate.mean - analytic_mean()) < 4 * estimate.standard_error
    # Each element is marginally exact GBM; variance/quantiles are marginal.
    # Relative SE of a lognormal sample variance here is below 1%; allow 3%.
    assert np.var(terminal, ddof=1) == pytest.approx(analytic_var(), rel=0.03)
    probs = np.array([0.05, 0.5, 0.95])
    z = np.array([NormalDist().inv_cdf(p) for p in probs])
    expected_q = S0 * np.exp((MU - SIGMA**2 / 2) * T + SIGMA * np.sqrt(T) * z)
    np.testing.assert_allclose(np.quantile(terminal, probs), expected_q, rtol=0.01)


def test_path_rows_match_analytic_mean_with_pair_standard_errors():
    steps, n = 64, 4_000
    paths = simulate_gbm(S0, MU, SIGMA, T, steps, n, seed=77, antithetic=True)
    times = np.arange(steps + 1) * T / steps
    for k in (1, steps // 2, steps):
        estimate = antithetic_mean_estimate(paths[k])
        assert abs(estimate.mean - analytic_mean(times[k])) < 4 * estimate.standard_error


def test_antithetic_reduces_replicate_variance_of_the_mean():
    """Compare the spread of the mean estimator across independent replicates.

    Same total draws per replicate (n). Analytic variance ratio for
    sigma=0.2, T=1 is 1 - exp(-0.04) ~ 0.039. With 300 replicates the
    ratio of sample variances has relative SE of roughly 12%, so the
    bounds below are wide and do not assume monotone error decay.
    """
    n, replicates = 2_000, 300
    seeds = np.random.SeedSequence(31415).generate_state(2 * replicates)
    plain_means, anti_means, pair_ses, naive_ses, covered = [], [], [], [], 0
    for r in range(replicates):
        plain = simulate_terminal_batched(S0, MU, SIGMA, T, n, seed=int(seeds[2 * r]))
        anti = simulate_terminal_batched(S0, MU, SIGMA, T, n, seed=int(seeds[2 * r + 1]),
                                         antithetic=True)
        plain_means.append(plain.mean())
        est = antithetic_mean_estimate(anti)
        anti_means.append(est.mean)
        pair_ses.append(est.standard_error)
        naive_ses.append(plain_mean_estimate(anti).standard_error)
        covered += est.ci_low <= analytic_mean() <= est.ci_high

    var_plain = np.var(plain_means, ddof=1)
    var_anti = np.var(anti_means, ddof=1)
    analytic_ratio = -np.expm1(-SIGMA**2 * T)
    ratio = var_anti / var_plain
    assert ratio < 0.1
    assert 0.5 * analytic_ratio < ratio < 2.0 * analytic_ratio

    # Analytic SE of each estimator for n draws.
    se_plain = np.sqrt(analytic_var() / n)
    se_anti = se_plain * np.sqrt(analytic_ratio)
    assert np.sqrt(var_plain) == pytest.approx(se_plain, rel=0.2)
    assert np.sqrt(var_anti) == pytest.approx(se_anti, rel=0.2)
    # Pair-based SE tracks the true spread; the i.i.d. formula overstates it here.
    assert np.mean(pair_ses) == pytest.approx(se_anti, rel=0.1)
    assert np.mean(naive_ses) > 3 * np.mean(pair_ses)
    # Pair-based 95% CI coverage (binomial SD ~1.3% at 300 replicates).
    assert 0.90 <= covered / replicates <= 0.99
