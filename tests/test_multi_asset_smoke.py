"""Fast smoke tests of the multi-asset GBM, calibration and CVaR decomposition."""

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from calibration.estimation import estimate_gbm_params
from calibration.multi_asset import (
    align_prices,
    estimate_multi_asset_gbm,
    ledoit_wolf_constant_correlation,
)
from risk.decomposition import (
    cvar_contribution_bootstrap,
    cvar_contributions,
    portfolio_cvar_contributions,
)
from risk.statistics import risk_summary
from simulation.gbm import GbmSimulator
from simulation.multi_asset import (
    MultiAssetGbmSimulator,
    correlation_factor,
    estimate_path_bytes,
    nearest_correlation,
    portfolio_paths,
    validate_correlation,
)


def _sim(n_sims=2000, steps=12, seed=7, rho=0.5, sigma=(0.2, 0.3)):
    return MultiAssetGbmSimulator(
        np.array([100.0, 50.0]), 1.0, steps, n_sims, seed,
        mu=np.array([0.05, 0.1]), sigma=np.array(sigma),
        correlation=np.array([[1.0, rho], [rho, 1.0]]))


def test_shape_row_zero_and_single_asset_matches_gbm():
    paths = _sim().generate_paths()
    assert paths.shape == (13, 2000, 2)
    assert np.all(paths[0] == np.array([100.0, 50.0]))
    one = MultiAssetGbmSimulator(np.array([100.0]), 1.0, 20, 500, 3, mu=np.array([0.08]),
                                 sigma=np.array([0.2]), correlation=[[1.0]])
    reference = GbmSimulator(100.0, 1.0, 20, 500, 3, mu=0.08, sigma=0.2).generate_paths()
    np.testing.assert_allclose(one.generate_paths()[:, :, 0], reference, rtol=1e-14)


def test_zero_volatility_is_deterministic_and_perfect_correlation_is_finite():
    paths = _sim(sigma=(0.0, 0.0)).generate_paths()
    times = np.arange(13) / 12
    np.testing.assert_allclose(paths[:, 0, 0], 100 * np.exp(0.05 * times), rtol=1e-14)
    assert np.all(paths == paths[:, :1, :])
    for rho in (1.0, -1.0):
        sim = _sim(rho=rho)
        assert sim.factorization == "eigh_clipped"
        logs = np.log(sim.generate_paths()[-1] / sim.s0)
        assert np.isfinite(logs).all()
        z = (logs - logs.mean(axis=0)) / logs.std(axis=0)
        np.testing.assert_allclose(z[:, 0], rho * z[:, 1], atol=1e-9)


def test_invalid_correlations_are_rejected_and_higham_repairs_explicitly():
    for bad in ([[1, 0.5], [0.4, 1]], [[1.1, 0], [0, 1]], [[1, 1.2], [1.2, 1]],
                [[True, False], [False, True]],
                [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]):
        with pytest.raises(ValueError):
            validate_correlation(bad)
    repaired = nearest_correlation([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    assert repaired["converged"]
    validate_correlation(repaired["correlation"])
    assert repaired["frobenius_distance"] > 0
    with pytest.raises(ValueError):
        nearest_correlation([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]], max_iterations=1)
    factor, method = correlation_factor([[1, 0.3], [0.3, 1]])
    assert method == "cholesky"
    np.testing.assert_allclose(factor @ factor.T, [[1, 0.3], [0.3, 1]], atol=1e-15)


def test_terminal_moments_and_batch_independence():
    n = 40_000
    sim = _sim(n_sims=n, seed=11, rho=-0.4)
    terminal = sim.generate_terminal(batch_size=3001)
    again = _sim(n_sims=n, seed=11, rho=-0.4).generate_terminal()
    np.testing.assert_array_equal(terminal, again)
    expected = sim.s0 * np.exp(sim.mu)
    se = terminal.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(terminal.mean(axis=0) - expected) < 4 * se)
    logs = np.log(terminal)
    r = np.corrcoef(logs, rowvar=False)[0, 1]
    assert abs(np.arctanh(r) - np.arctanh(-0.4)) < 4 / np.sqrt(n - 3)


def test_memory_estimate_bounds_measured_peak_and_rejects_before_allocation():
    sim = _sim(n_sims=3000, steps=50)
    tracemalloc.start()
    sim.generate_paths()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak <= sim.path_bytes() == estimate_path_bytes(50, 3000, 2, 2)
    huge = _sim(n_sims=10**7, steps=10)
    state = huge.rng.bit_generator.state
    with pytest.raises(ValueError, match="bytes"):
        huge.generate_paths()
    assert huge.rng.bit_generator.state == state


def test_portfolio_risk_and_contributions_match_risk_summary():
    paths = _sim(n_sims=5000, seed=5).generate_paths()
    values = portfolio_paths(paths, np.array([0.6, 0.4]), "initial_weights", initial_value=1000.0)
    assert values.shape == (13, 5000)
    assert np.all(values[0] == values[0, 0])
    assert values[0, 0] == pytest.approx(1000.0, rel=1e-14)
    summary = risk_summary(values)
    result = portfolio_cvar_contributions(paths, np.array([0.6, 0.4]), "initial_weights",
                                          initial_value=1000.0)
    assert result["cvar"] == summary["cvar"] and result["var"] == summary["var"]
    assert result["sum_identity_error"] <= 1e-12 * result["sum_identity_scale"]
    with pytest.raises(ValueError, match="positive"):
        portfolio_paths(paths, np.array([1.0, -2.0]))  # V0 = 100 - 100 = 0


def test_contribution_bootstrap_is_reproducible_and_batch_independent():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal((2000, 2))
    point = cvar_contributions(losses, 0.9)
    assert sum(point["contributions"]) == pytest.approx(point["cvar"], abs=1e-12)
    first = cvar_contribution_bootstrap(losses, 0.9, n_bootstrap=50, seed=1, batch_size=7)
    second = cvar_contribution_bootstrap(losses, 0.9, n_bootstrap=50, seed=1)
    assert first["contribution_standard_errors"] == second["contribution_standard_errors"]
    assert all(se > 0 for se in first["contribution_standard_errors"])
    # Two i.i.d. identical components: equal contributions within the SE.
    c, se = point["contributions"], first["contribution_standard_errors"]
    assert abs(c[0] - c[1]) < 4 * np.hypot(*se)


def _dated(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_calibration_alignment_and_estimates():
    paths = _sim(n_sims=1, steps=500, seed=2).generate_paths()[:, 0, :]
    a, b = _dated(paths[:, 0]), _dated(paths[:, 1])
    result = estimate_multi_asset_gbm({"A": a, "B": b.iloc[3:]})
    assert result["alignment"]["dropped_dates"] == {"A": 3, "B": 0}
    mu_a, sigma_a = estimate_gbm_params(a.iloc[3:])
    assert result["sigma"][0] == pytest.approx(sigma_a, rel=1e-12)
    assert result["mu"][0] == pytest.approx(mu_a, rel=1e-12)
    validate_correlation(result["correlation"])
    gapped = b.drop(b.index[10])
    frame, report = align_prices({"A": a, "B": gapped}, forward_fill=True)
    assert report["filled_values"] == {"A": 0, "B": 1} and len(frame) == len(a)
    with pytest.raises(ValueError, match="zero volatility"):
        estimate_multi_asset_gbm({"A": a, "C": _dated(np.full(len(a), 5.0))})


def _ledoit_wolf_literal(y):
    t, n = y.shape
    y = y - y.mean(axis=0)
    s = y.T @ y / t
    r = s / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    rbar = (r.sum() - n) / (n * (n - 1))
    f = rbar * np.sqrt(np.outer(np.diag(s), np.diag(s)))
    np.fill_diagonal(f, np.diag(s))
    pi = sum(np.mean((y[:, i] * y[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))
    rho = sum(np.mean((y[:, i] * y[:, i] - s[i, i]) ** 2) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                th_ii = np.mean((y[:, i] ** 2 - s[i, i]) * (y[:, i] * y[:, j] - s[i, j]))
                th_jj = np.mean((y[:, j] ** 2 - s[j, j]) * (y[:, i] * y[:, j] - s[i, j]))
                rho += rbar / 2 * (np.sqrt(s[j, j] / s[i, i]) * th_ii
                                   + np.sqrt(s[i, i] / s[j, j]) * th_jj)
    gamma = np.sum((f - s) ** 2)
    return max(0.0, min((pi - rho) / gamma / t, 1.0))


def test_ledoit_wolf_matches_literal_formula():
    rng = np.random.default_rng(4)
    y = rng.standard_normal((60, 4)) @ np.linalg.cholesky(
        np.array([[1, .5, .2, 0], [.5, 1, .3, .1], [.2, .3, 1, .4], [0, .1, .4, 1]])).T
    result = ledoit_wolf_constant_correlation(y)
    assert result["shrinkage"] == pytest.approx(_ledoit_wolf_literal(y), rel=1e-10)
    assert 0 <= result["shrinkage"] <= 1
    two = ledoit_wolf_constant_correlation(y[:, :2])
    assert two["shrinkage"] == 0.0 and two["note"]
