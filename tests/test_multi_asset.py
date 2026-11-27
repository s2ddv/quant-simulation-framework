"""Independent statistical and contract tests of the correlated multi-asset GBM (E6.1, E6.3).

Reference law (constant parameters, exact solution; Glasserman 2003, Sec. 3.2.3)::

    ln S_i(T) - ln S_i0 ~ N((mu_i - sigma_i^2/2) T, sigma_i^2 T)
    Cov(ln S_i(T), ln S_j(T)) = rho_ij sigma_i sigma_j T
    E[S_i(T)] = S_i0 exp(mu_i T)
    Cov(S_i(T), S_j(T)) = S_i0 S_j0 exp((mu_i + mu_j) T) (exp(rho_ij sigma_i sigma_j T) - 1)

Every formula is re-derived here (no call into the implementation for the
reference values). Tolerances follow docs/plano-expansao.md Sec. 2.5:
``k = 4`` standard errors per assertion (two-sided false alarm ~6.3e-5 under
asymptotic normality), Fisher-z for correlations, KS with p > 1e-3. Seeds are
fixed, so each assertion is deterministic for a given NumPy build; the
tolerance argument says the chosen seeds are not cherry-picked outliers.
"""

import tracemalloc

import numpy as np
import pytest
from scipy import stats

from risk.statistics import risk_summary
from simulation.gbm import GbmSimulator
from simulation.multi_asset import (
    MultiAssetGbmSimulator,
    correlation_factor,
    estimate_path_bytes,
    nearest_correlation,
    portfolio_paths,
    portfolio_quantities,
    validate_correlation,
)

K_SE = 4.0
S0 = np.array([100.0, 50.0, 20.0])
MU = np.array([0.05, 0.10, -0.02])
SIGMA = np.array([0.20, 0.35, 0.50])
HORIZON = 1.5  # non-unit horizon so that a missing T or sqrt(dt) is visible
# Positive and negative correlations; eigenvalues ~ (0.36, 0.82, 1.82): PD.
R3 = np.array([[1.0, 0.6, -0.4],
               [0.6, 1.0, -0.2],
               [-0.4, -0.2, 1.0]])


def _sim(n_sims, steps=4, seed=0, *, s0=S0, mu=MU, sigma=SIGMA, correlation=R3,
         horizon=HORIZON):
    return MultiAssetGbmSimulator(s0, horizon, steps, n_sims, seed, mu=mu, sigma=sigma,
                                  correlation=correlation)


def _fisher_ok(logs, correlation):
    """|atanh(r_ij) - atanh(rho_ij)| <= 4/sqrt(N - 3) for every i < j.

    Under bivariate normality atanh(r) is ~N(atanh(rho), 1/(N-3)), so 4/sqrt(N-3)
    is a 4-sigma band. Returns the list of (i, j, r, rho, z-score).
    """
    n = logs.shape[0]
    r = np.corrcoef(logs, rowvar=False)
    rows = []
    for i in range(r.shape[0]):
        for j in range(i + 1, r.shape[0]):
            if abs(correlation[i, j]) >= 1:
                continue  # Fisher-z undefined at |rho| = 1; tested separately
            zscore = (np.arctanh(r[i, j]) - np.arctanh(correlation[i, j])) * np.sqrt(n - 3)
            rows.append((i, j, float(r[i, j]), float(correlation[i, j]), float(zscore)))
    return rows


def _analytic_cov(s0, mu, sigma, correlation, horizon):
    growth = np.outer(s0 * np.exp(mu * horizon), s0 * np.exp(mu * horizon))
    return growth * np.expm1(correlation * np.outer(sigma, sigma) * horizon)


def _cov_with_se(terminal):
    """Sample covariance and its plug-in SE std(d_i d_j, ddof=1)/sqrt(N), d = S - mean."""
    n = terminal.shape[0]
    d = terminal - terminal.mean(axis=0)
    k = terminal.shape[1]
    est = np.empty((k, k))
    se = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            prod = d[:, i] * d[:, j]
            est[i, j] = prod.sum() / (n - 1)
            se[i, j] = prod.std(ddof=1) / np.sqrt(n)
    return est, se


# --------------------------------------------------------------------------- E6.1 law


@pytest.mark.parametrize("method", ["paths", "terminal"])
def test_terminal_log_return_correlation_fisher_z(method):
    # N = 100_000: Fisher band half-width 4/sqrt(N-3) = 0.0126 in z units.
    n = 100_000
    sim = _sim(n, steps=4, seed=20261127)
    terminal = sim.generate_paths()[-1] if method == "paths" else sim.generate_terminal()
    logs = np.log(terminal) - np.log(S0)
    for i, j, r, rho, z in _fisher_ok(logs, R3):
        assert abs(z) <= K_SE, (i, j, r, rho, z)


@pytest.mark.slow
def test_terminal_log_return_correlation_fisher_z_large():
    n = 1_000_000  # band 0.004 in z units; terminal sampler only (24 MB)
    logs = np.log(_sim(n, seed=99).generate_terminal()) - np.log(S0)
    for i, j, r, rho, z in _fisher_ok(logs, R3):
        assert abs(z) <= K_SE, (i, j, r, rho, z)


@pytest.mark.parametrize("method", ["paths", "terminal"])
def test_terminal_means_variances_covariances_match_analytic(method):
    n = 200_000
    sim = _sim(n, steps=3, seed=314)
    terminal = sim.generate_paths()[-1] if method == "paths" else sim.generate_terminal()
    mean_se = terminal.std(axis=0, ddof=1) / np.sqrt(n)
    expected_mean = S0 * np.exp(MU * HORIZON)
    assert np.all(np.abs(terminal.mean(axis=0) - expected_mean) <= K_SE * mean_se)
    est, se = _cov_with_se(terminal)
    expected = _analytic_cov(S0, MU, SIGMA, R3, HORIZON)
    # Lognormal products are heavy tailed; the plug-in SE is still consistent and
    # N = 2e5 keeps its relative error small (fourth moments are finite here).
    z = (est - expected) / se
    assert np.all(np.abs(z) <= K_SE), z
    # Log-space covariance rho_ij sigma_i sigma_j T: SE of a normal covariance
    # sqrt((s_ii s_jj + s_ij^2)/N).
    logs = np.log(terminal / S0)
    log_cov = np.cov(logs, rowvar=False)
    ref = R3 * np.outer(SIGMA, SIGMA) * HORIZON
    ref_se = np.sqrt((np.outer(np.diag(ref), np.diag(ref)) + ref**2) / n)
    assert np.all(np.abs(log_cov - ref) <= K_SE * ref_se)
    # Log means (mu - sigma^2/2) T: SE sigma sqrt(T/N).
    log_mean_se = SIGMA * np.sqrt(HORIZON / n)
    assert np.all(np.abs(logs.mean(axis=0) - (MU - SIGMA**2 / 2) * HORIZON)
                  <= K_SE * log_mean_se)


def test_single_asset_reproduces_gbm_law_with_independent_seeds():
    n = 50_000
    s0, mu, sigma, horizon, steps = 80.0, 0.07, 0.3, 2.0, 8
    multi = MultiAssetGbmSimulator([s0], horizon, steps, n, 101, mu=[mu], sigma=[sigma],
                                   correlation=[[1.0]]).generate_paths()
    single = GbmSimulator(s0, horizon, steps, n, 202, mu=mu, sigma=sigma).generate_paths("normal")
    assert multi.shape == (steps + 1, n, 1)
    a, b = multi[-1, :, 0], single[-1]
    mean = s0 * np.exp(mu * horizon)
    var = s0**2 * np.exp(2 * mu * horizon) * np.expm1(sigma**2 * horizon)
    for sample in (a, b):
        assert abs(sample.mean() - mean) <= K_SE * sample.std(ddof=1) / np.sqrt(n)
        d2 = (sample - sample.mean()) ** 2
        assert abs(sample.var(ddof=1) - var) <= K_SE * d2.std(ddof=1) / np.sqrt(n)
    assert stats.ks_2samp(a, b).pvalue > 1e-3
    # Against the analytic lognormal law, at an interior row too (t = 1 year).
    for row, t in ((-1, horizon), (4, 1.0)):
        logs = np.log(multi[row, :, 0] / s0)
        ref = stats.norm(loc=(mu - sigma**2 / 2) * t, scale=sigma * np.sqrt(t))
        assert stats.kstest(logs, ref.cdf).pvalue > 1e-3


def test_generate_terminal_law_equals_generate_paths_last_row():
    n = 100_000
    a = _sim(n, steps=6, seed=1).generate_paths()[-1]
    b = _sim(n, steps=6, seed=2).generate_terminal()
    se = np.sqrt(a.var(axis=0, ddof=1) / n + b.var(axis=0, ddof=1) / n)
    assert np.all(np.abs(a.mean(axis=0) - b.mean(axis=0)) <= K_SE * se)
    for i in range(3):
        assert stats.ks_2samp(np.log(a[:, i]), np.log(b[:, i])).pvalue > 1e-3


def test_increments_are_independent_across_steps_and_scale_with_dt():
    # Exact GBM: log increments over disjoint steps are i.i.d. N((mu-s^2/2)dt, s^2 dt)
    # with cross-asset correlation R and zero correlation across steps.
    n, steps = 40_000, 5
    paths = _sim(n, steps=steps, seed=8).generate_paths()
    dt = HORIZON / steps
    inc = np.diff(np.log(paths), axis=0)  # (steps, n, 3)
    var_se = SIGMA**2 * dt * np.sqrt(2 / (n - 1))
    assert np.all(np.abs(inc.var(axis=1, ddof=1) - SIGMA**2 * dt) <= K_SE * var_se)
    across = np.corrcoef(inc[0, :, 1], inc[3, :, 1])[0, 1]
    assert abs(np.arctanh(across)) <= K_SE / np.sqrt(n - 3)
    for row in _fisher_ok(inc[2], R3):
        assert abs(row[-1]) <= K_SE, row


# --------------------------------------------------------------------------- rank deficiency


@pytest.mark.parametrize("rho", [1.0, -1.0])
def test_perfect_correlation_rank_one(rho):
    corr = np.array([[1.0, rho], [rho, 1.0]])
    factor, method = correlation_factor(corr)
    assert method == "eigh_clipped"
    np.testing.assert_allclose(factor @ factor.T, corr, atol=1e-15)
    sim = _sim(5000, steps=7, seed=3, s0=S0[:2], mu=MU[:2], sigma=SIGMA[:2],
               correlation=corr)
    assert sim.factorization == "eigh_clipped"
    for terminal in (sim.generate_paths()[-1], sim.generate_terminal()):
        logs = np.log(terminal / S0[:2])
        assert np.isfinite(logs).all()
        z = (logs - (MU[:2] - SIGMA[:2] ** 2 / 2) * HORIZON) / (SIGMA[:2] * np.sqrt(HORIZON))
        # Exact linear dependence up to rounding (|z| <= ~5, so 1e-12 is ~1e3 ulps).
        np.testing.assert_allclose(z[:, 0], rho * z[:, 1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("corr", [
    [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    # Three unit vectors at 120 degrees in a plane: rank 2, entries exact in binary.
    [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]],
    [[1.0, 0.5, 0.5], [0.5, 1.0, -0.5], [0.5, -0.5, 1.0]],
])
def test_rank_two_three_by_three(corr):
    corr = np.array(corr)
    eigenvalues, vectors = np.linalg.eigh(corr)
    assert abs(eigenvalues[0]) < 1e-15 and eigenvalues[1] > 0.5
    factor, method = correlation_factor(corr)
    assert method == "eigh_clipped"
    np.testing.assert_allclose(factor @ factor.T, corr, atol=1e-14)
    sim = _sim(20_000, steps=5, seed=4, correlation=corr)
    logs = np.log(sim.generate_paths()[-1] / S0)
    assert np.isfinite(logs).all()
    z = (logs - (MU - SIGMA**2 / 2) * HORIZON) / (SIGMA * np.sqrt(HORIZON))
    # The null vector w of R gives w.z = 0 exactly in law.
    assert np.max(np.abs(z @ vectors[:, 0])) <= 1e-12
    # Pairs with |rho| < 1 still match rho within Fisher-z (_fisher_ok skips |rho| = 1).
    for row in _fisher_ok(logs, corr):
        assert abs(row[-1]) <= K_SE, row


def test_eigh_clipped_handles_slightly_negative_eigenvalue_as_documented():
    # R with smallest eigenvalue ~ -5e-13 (inside the accepted [-1e-12, 0) band):
    # documented fallback = clip to 0, L = V sqrt(lam+), rows renormalized so that
    # (L L^T)_ii = 1. Reference rebuilt here; L L^T is basis/sign invariant.
    base = np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]])
    w = np.ones(3) / np.sqrt(3)  # null vector of base
    eps = 5e-13
    shifted = base - eps * np.outer(w, w)
    d = np.sqrt(np.diag(shifted))
    corr = shifted / np.outer(d, d)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    lam, vec = np.linalg.eigh(corr)
    assert -1e-12 < lam[0] < -1e-13
    validate_correlation(corr)
    factor, method = correlation_factor(corr)
    assert method == "eigh_clipped"
    ref = vec * np.sqrt(np.maximum(lam, 0.0))
    ref /= np.linalg.norm(ref, axis=1)[:, None]
    np.testing.assert_allclose(factor @ factor.T, ref @ ref.T, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.diag(factor @ factor.T), 1.0, rtol=0, atol=2e-15)
    # Same construction with eps = 3e-12 falls outside the band: rejected.
    beyond = base - 3e-12 * np.outer(w, w)
    beyond /= np.sqrt(np.outer(np.diag(beyond), np.diag(beyond)))
    np.fill_diagonal(beyond, 1.0)
    with pytest.raises(ValueError, match="semidefinite"):
        validate_correlation((beyond + beyond.T) / 2)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("bad, message", [
    ([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]], "semidefinite"),
    ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
    ([[1.0, 0.0], [0.0, 0.9]], "diagonal"),
    ([[1.0, 1.2], [1.2, 1.0]], r"\|R_ij\| <= 1"),
    ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
    ([[1.0, np.inf], [np.inf, 1.0]], "finite"),
    ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square"),
    ([1.0, 1.0], "square"),
    (np.empty((0, 0)), "square"),
    ([[True, False], [False, True]], "real"),
    ([[1 + 0j, 0], [0, 1]], "real"),
    ([["1", "0"], ["0", "1"]], "real"),
])
def test_invalid_correlations_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_correlation(bad)
    with pytest.raises(ValueError):
        correlation_factor(bad)


def test_validate_correlation_tolerance_boundaries_and_copy():
    tiny = 5e-13
    ok = np.array([[1.0, 0.3 + tiny], [0.3, 1.0]])
    out = validate_correlation(ok)
    assert out.dtype == np.float64 and out is not ok
    with pytest.raises(ValueError, match="symmetric"):
        validate_correlation(np.array([[1.0, 0.3 + 3e-12], [0.3, 1.0]]))
    validate_correlation([[1]])  # integer 1x1 accepted
    for tol in (True, -1.0, np.nan, "x"):
        with pytest.raises(ValueError, match="tol"):
            validate_correlation(np.eye(2), tol=tol)


def test_nearest_correlation_repairs_explicitly_and_beats_naive_clip():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    before = bad.copy()
    out = nearest_correlation(bad)
    np.testing.assert_array_equal(bad, before)  # input untouched
    fixed = out["correlation"]
    validate_correlation(fixed)
    assert out["converged"] and out["iterations"] >= 1
    assert out["frobenius_distance"] == pytest.approx(np.linalg.norm(bad - fixed), rel=1e-12)
    # Naive repair: clip eigenvalues at 0 then rescale to unit diagonal.
    w, v = np.linalg.eigh(bad)
    clipped = (v * np.maximum(w, 0)) @ v.T
    d = np.sqrt(np.diag(clipped))
    naive = clipped / np.outer(d, d)
    assert np.linalg.norm(bad - fixed) < np.linalg.norm(bad - naive)
    # Local optimality check: no feasible random perturbation is closer.
    rng = np.random.default_rng(0)
    for _ in range(200):
        e = rng.standard_normal((3, 3)) * 1e-3
        e = (e + e.T) / 2
        np.fill_diagonal(e, 0)
        cand = fixed + e
        if np.linalg.eigvalsh(cand)[0] >= 0:
            assert np.linalg.norm(bad - cand) >= np.linalg.norm(bad - fixed) - 1e-9
    # A valid correlation matrix is (numerically) a fixed point.
    same = nearest_correlation(R3)["correlation"]
    np.testing.assert_allclose(same, R3, atol=1e-8)
    # Known optimum (Higham 2002, Sec. 4 example): A = [[1,1,0],[1,1,1],[0,1,1]].
    higham = nearest_correlation(np.array([[1.0, 1, 0], [1, 1, 1], [0, 1, 1]]), tol=1e-12)
    expected = np.array([[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]])
    np.testing.assert_allclose(higham["correlation"], expected, atol=1e-4)


@pytest.mark.parametrize("kwargs, message", [
    ({"tol": 0.0}, "tol"), ({"tol": True}, "tol"), ({"max_iterations": 0}, "max_iterations"),
    ({"max_iterations": True}, "max_iterations"), ({"max_iterations": 2.0}, "max_iterations"),
])
def test_nearest_correlation_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        nearest_correlation(np.eye(2), **kwargs)


@pytest.mark.parametrize("matrix", [
    [[1.0, 0.5], [0.4, 1.0]], [[1.0, np.nan], [np.nan, 1.0]], [[True]], [1.0, 2.0],
    np.empty((0, 0)),
])
def test_nearest_correlation_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        nearest_correlation(matrix)


def test_nearest_correlation_never_returns_unconverged():
    with pytest.raises(ValueError, match="converge"):
        nearest_correlation([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]], max_iterations=1)


# --------------------------------------------------------------------------- contracts


def test_shape_dtype_row_zero_and_determinism():
    a = _sim(300, steps=9, seed=5).generate_paths()
    b = _sim(300, steps=9, seed=5).generate_paths()
    assert a.shape == (10, 300, 3) and a.dtype == np.float64
    assert np.all(a[0] == S0)  # exact, every simulation
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, _sim(300, steps=9, seed=6).generate_paths())
    ta = _sim(300, seed=5).generate_terminal()
    assert ta.shape == (300, 3) and ta.dtype == np.float64
    np.testing.assert_array_equal(ta, _sim(300, seed=5).generate_terminal())
    # Successive calls advance the single stream.
    sim = _sim(50, seed=5)
    assert not np.array_equal(sim.generate_paths(), sim.generate_paths())


def test_zero_volatility_is_deterministic():
    sim = _sim(4, steps=6, seed=0, sigma=np.zeros(3))
    paths = sim.generate_paths()
    t = np.linspace(0, HORIZON, 7)
    np.testing.assert_allclose(paths, np.broadcast_to(
        (S0 * np.exp(np.outer(t, MU)))[:, None, :], paths.shape), rtol=1e-14)
    np.testing.assert_allclose(sim.generate_terminal(), np.broadcast_to(
        S0 * np.exp(MU * HORIZON), (4, 3)), rtol=1e-14)
    # Mixed: only asset 0 is deterministic.
    mixed = _sim(1000, seed=1, sigma=np.array([0.0, 0.3, 0.4])).generate_paths()
    np.testing.assert_allclose(mixed[-1, :, 0], S0[0] * np.exp(MU[0] * HORIZON), rtol=1e-14)
    assert mixed[-1, :, 1].std() > 0


def test_single_simulation_and_single_step():
    one = _sim(1, steps=1, seed=2).generate_paths()
    assert one.shape == (2, 1, 3) and np.all(one[0] == S0) and np.isfinite(one).all()
    assert _sim(1, seed=2).generate_terminal().shape == (1, 3)
    # steps=1: the single row-1 draw has the terminal law.
    n = 50_000
    last = _sim(n, steps=1, seed=9).generate_paths()[-1]
    se = last.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(last.mean(axis=0) - S0 * np.exp(MU * HORIZON)) <= K_SE * se)


def test_tiny_horizon_and_extreme_but_finite_parameters():
    sim = _sim(100, steps=3, seed=1, horizon=1e-10)
    paths = sim.generate_paths()
    assert np.isfinite(paths).all()
    np.testing.assert_allclose(paths[-1], np.broadcast_to(S0, (100, 3)), rtol=1e-3)
    # sigma = 5, T = 10: ln S - ln s0 ~ N(-125, 15.8^2), still finite and positive.
    big = _sim(100, steps=2, seed=1, sigma=np.array([5.0, 5.0, 5.0]), horizon=10.0)
    assert np.isfinite(big.generate_paths()).all() and np.all(big.generate_terminal() > 0)
    # s0 = 1e-300 with drift -12.5 * 40 = -500: prices underflow to 0 -> ValueError.
    with pytest.raises(ValueError, match="precision"):
        _sim(100, steps=2, seed=1, s0=np.array([1e-300, 1e-300, 1e-300]),
             sigma=np.array([5.0, 5.0, 5.0]), horizon=40.0).generate_paths()


def test_generate_terminal_independent_of_batch_size():
    sim_ref = _sim(10_001, seed=12).generate_terminal()
    for batch in (2, 7, 4096, 10_000, 10_001, 10**6):
        np.testing.assert_array_equal(_sim(10_001, seed=12).generate_terminal(batch), sim_ref)
    # batch_size=1: same draws; equal to rounding (see the next test).
    np.testing.assert_allclose(_sim(10_001, seed=12).generate_terminal(1), sim_ref,
                               rtol=4e-15, atol=0)


def test_generate_terminal_batch_size_changes_only_blas_rounding():
    """Documented contract: the shocks Z never depend on batch_size; the values
    agree up to the rounding of the BLAS product Z L^T (a 1-row block may take a
    different matmul kernel, e.g. OpenBLAS 0.3.34: ~6.7e-16 relative), not bitwise.

    Tolerance per element (k = 3 assets, eps = 2^-52): two evaluations of
    x_i = sum_j Z_j L_ij in any order each err by <= (k - 1) eps sum_j |Z_j L_ij|
    (Higham 2002, ch. 4), so ln S differs by <= 2 (k - 1) eps sigma_i sqrt(T)
    sum_j |Z_j L_ij|; the shared drift add, exp and the s0 multiply add <= 4 eps
    relative. Taking the absolute values makes the bound independent of cancellation.
    """
    n, seed, eps, k = 10_001, 12, np.finfo(float).eps, 3
    one, default = _sim(n, seed=seed), _sim(n, seed=seed)
    values_one, values_default = one.generate_terminal(1), default.generate_terminal()
    # Same draws: the generators consumed exactly the same stream...
    assert one.rng.bit_generator.state == default.rng.bit_generator.state
    # ...which is one C-order (n, k) standard-normal draw from default_rng(seed).
    z = np.random.default_rng(seed).standard_normal((n, k))
    factor = one.factor
    log_scale = SIGMA * np.sqrt(HORIZON) * (np.abs(z) @ np.abs(factor).T)
    bound = 2 * (k - 1) * eps * log_scale + 4 * eps
    reference = S0 * np.exp((MU - SIGMA**2 / 2) * HORIZON + SIGMA * np.sqrt(HORIZON) * (z @ factor.T))
    for values in (values_one, values_default):
        relative = np.abs(values / reference - 1)
        assert np.all(relative <= bound + 4 * eps), float(np.max(relative / (bound + 4 * eps)))
    relative = np.abs(values_one / values_default - 1)
    assert np.all(relative <= 2 * bound), float(np.max(relative / (2 * bound)))


def test_generate_terminal_rejects_bad_batch_size():
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="batch_size"):
            _sim(10, seed=1).generate_terminal(bad)


def test_memory_budget_rejects_before_allocating_and_boundary_is_exact():
    sim = _sim(4000, steps=20, seed=3)
    required = sim.path_bytes()
    assert required == estimate_path_bytes(20, 4000, 3)
    assert required >= 8 * 21 * 4000 * 3 + 2 * 8 * 4000 * 3
    sim.MAX_WORKING_BYTES = required  # exactly at the limit: accepted
    sim.generate_paths()
    sim.MAX_WORKING_BYTES = required - 1  # one byte short: rejected, no draw
    state = sim.rng.bit_generator.state
    with pytest.raises(ValueError, match="bytes"):
        sim.generate_paths()
    assert sim.rng.bit_generator.state == state
    sim.MAX_WORKING_BYTES = sim.terminal_bytes(4000) - 1
    with pytest.raises(ValueError, match="bytes"):
        sim.generate_terminal()
    assert sim.rng.bit_generator.state == state
    # Default budget (256 MiB): a 10^7 x 3 x 11 request is rejected without allocation.
    huge = _sim(10**7, steps=10, seed=1)
    tracemalloc.start()
    with pytest.raises(ValueError, match="bytes"):
        huge.generate_paths()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 1024**2


@pytest.mark.parametrize("n_sims, steps, n_assets", [(3000, 50, 2), (20_000, 10, 3),
                                                     (50_000, 2, 5)])
def test_traced_peak_within_estimate(n_sims, steps, n_assets):
    # tracemalloc counts NumPy data buffers; the estimate adds the output, two
    # per-step blocks, two n_assets^2 factors and two ufunc buffers
    # (2 * 8 * getbufsize() = 128 KiB). Measured slack is reported, not asserted.
    corr = np.eye(n_assets) * 0.7 + 0.3
    sim = _sim(n_sims, steps=steps, seed=1, s0=np.full(n_assets, 10.0),
               mu=np.zeros(n_assets), sigma=np.full(n_assets, 0.2), correlation=corr)
    tracemalloc.start()
    out = sim.generate_paths()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak <= sim.path_bytes()
    assert peak >= out.nbytes
    tracemalloc.start()
    sim.generate_terminal()
    peak_t = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak_t <= sim.terminal_bytes(min(n_sims, 65536))


@pytest.mark.parametrize("kwargs, message", [
    ({"s0": [True, True, True]}, "real"),
    ({"s0": [100.0, -1.0, 2.0]}, "positive"),
    ({"s0": [100.0, np.nan, 2.0]}, "finite"),
    ({"s0": 100.0}, "1-D"),
    ({"s0": [[100.0, 50.0, 20.0]]}, "1-D"),
    ({"mu": [0.1, 0.1]}, "same length"),
    ({"mu": [0.1, np.inf, 0.1]}, "finite"),
    ({"sigma": [0.2, -0.1, 0.2]}, "nonnegative"),
    ({"sigma": [0.2, 0.2, np.nan]}, "finite"),
    ({"correlation": np.eye(2)}, "does not match"),
    ({"correlation": [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}, "R_ij"),
    ({"horizon": True}, "horizon"), ({"horizon": 0.0}, "horizon"),
    ({"horizon": np.inf}, "horizon"), ({"horizon": "1"}, "horizon"),
    ({"steps": 0}, "steps"), ({"steps": True}, "steps"), ({"steps": 1.0}, "steps"),
    ({"n_sims": 0}, "n_sims"), ({"n_sims": True}, "n_sims"),
    ({"seed": -1}, "seed"), ({"seed": True}, "seed"), ({"seed": 1.5}, "seed"),
])
def test_simulator_rejects_invalid_inputs(kwargs, message):
    args = {"s0": S0, "horizon": 1.0, "steps": 2, "n_sims": 3, "seed": 0,
            "mu": MU, "sigma": SIGMA, "correlation": R3}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MultiAssetGbmSimulator(args["s0"], args["horizon"], args["steps"], args["n_sims"],
                               args["seed"], mu=args["mu"], sigma=args["sigma"],
                               correlation=args["correlation"])


def test_dt_underflow_rejected():
    with pytest.raises(ValueError, match="horizon / steps"):
        _sim(2, steps=10**6, horizon=5e-324)


# --------------------------------------------------------------------------- E6.3 portfolio


def test_single_asset_unit_quantity_is_the_asset_path():
    paths = _sim(500, steps=5, seed=1, s0=S0[:1], mu=MU[:1], sigma=SIGMA[:1],
                 correlation=[[1.0]]).generate_paths()
    values = portfolio_paths(paths, [1.0])
    assert values.shape == (6, 500)
    np.testing.assert_array_equal(values, paths[:, :, 0])
    np.testing.assert_array_equal(portfolio_paths(paths, [1.0], "initial_weights",
                                                  initial_value=S0[0]), paths[:, :, 0])


def test_quantities_and_initial_weights_are_consistent():
    paths = _sim(2000, steps=5, seed=2).generate_paths()
    w = np.array([0.5, 0.8, -0.3])  # a short position, V0 > 0
    v0 = 1234.5
    q = w * v0 / S0
    by_weights = portfolio_paths(paths, w, "initial_weights", initial_value=v0)
    by_quantities = portfolio_paths(paths, q, "quantities")
    np.testing.assert_allclose(by_weights, by_quantities, rtol=1e-13)
    np.testing.assert_allclose(portfolio_quantities(S0, w, "initial_weights",
                                                    initial_value=v0), q, rtol=1e-15)
    assert np.all(by_weights[0] == by_weights[0, 0])
    assert by_weights[0, 0] == pytest.approx(v0, rel=1e-14)
    # Independent reference: explicit loop over assets.
    manual = sum(q[i] * paths[:, :, i] for i in range(3))
    np.testing.assert_allclose(by_quantities[1:], manual[1:], rtol=1e-12)
    summary = risk_summary(by_weights, 0.95)
    losses = by_weights[0, 0] - by_weights[-1]
    var = np.percentile(losses, 95)
    assert summary["var"] == pytest.approx(var, rel=1e-12)
    assert summary["cvar"] == pytest.approx(losses[losses >= var].mean(), rel=1e-12)
    # Buy-and-hold expectation: E[V_T] = sum q_i S_i0 exp(mu_i T), at 4 SE.
    expected = float(np.sum(q * S0 * np.exp(MU * HORIZON)))
    se = by_quantities[-1].std(ddof=1) / np.sqrt(2000)
    assert abs(by_quantities[-1].mean() - expected) <= K_SE * se


def test_zero_initial_value_and_bad_weights_rejected():
    s0 = np.array([50.0, 50.0])
    paths = _sim(10, seed=1, s0=s0, mu=MU[:2], sigma=SIGMA[:2],
                 correlation=np.eye(2)).generate_paths()
    with pytest.raises(ValueError, match="initial value must be positive"):
        portfolio_paths(paths, [1.0, -1.0])
    with pytest.raises(ValueError, match="initial value must be positive"):
        portfolio_paths(paths, [1.0, -2.0])
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_paths(paths, [0.5, 0.4], "initial_weights")
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_paths(paths, [1.0, 1.0], "initial_weights")
    # Weights summing to 1 with zero sum of positives impossible; short allowed.
    v = portfolio_paths(paths, [1.5, -0.5], "initial_weights", initial_value=10.0)
    assert v[0, 0] == pytest.approx(10.0, rel=1e-14)
    with pytest.raises(ValueError, match="kind"):
        portfolio_paths(paths, [1.0, 1.0], "weights")
    with pytest.raises(ValueError, match="entries"):
        portfolio_paths(paths, [1.0])
    for bad_v0 in (0.0, -1.0, np.inf, True):
        with pytest.raises(ValueError, match="initial_value"):
            portfolio_paths(paths, [0.5, 0.5], "initial_weights", initial_value=bad_v0)
    for bad_w in ([True, False], [np.nan, 1.0], [[1.0, 1.0]], 1.0):
        with pytest.raises(ValueError):
            portfolio_paths(paths, bad_w)


@pytest.mark.parametrize("bad", [
    np.ones((5, 3)), np.ones((1, 3, 2)), np.ones((3, 0, 2)), np.ones((3, 2, 0)),
    np.full((3, 2, 2), np.nan), -np.ones((3, 2, 2)), np.zeros((3, 2, 2)),
    np.ones((3, 2, 2), dtype=bool),
])
def test_portfolio_rejects_bad_asset_paths(bad):
    with pytest.raises(ValueError):
        portfolio_paths(bad, np.ones(bad.shape[-1] if bad.ndim == 3 and bad.shape[-1] else 2))


def test_portfolio_rejects_non_common_initial_price():
    paths = np.ones((3, 2, 2))
    paths[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="same initial price"):
        portfolio_paths(paths, [1.0, 1.0])

