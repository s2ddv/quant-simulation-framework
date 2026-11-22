"""Tests for randomized quasi-Monte Carlo in the GBM sampler (E2.3, offline, seeded).

Model (constant parameters, exact lognormal law on the grid)::

    log S_t = log S0 + (mu - sigma^2/2) * t + sigma * W_t
    E[S_T]  = S0 * exp(mu*T)
    Cov(W_s, W_t) = min(s, t)

QMC contract (docs/plano-expansao.md E2.3 and the principal's decisions):
``n_sims = R*N`` with ``N`` a power of two, scramble ``r`` in columns (or terminal
entries) ``[r*N, (r+1)*N)``, scrambled Sobol (``bits=30``) with ``u`` clipped to
``[2^-53, 1 - 2^-53]`` and ``Z = ndtri(u)``; terminal uses one dimension, paths
use ``steps`` dimensions through a breadth-first Brownian bridge (dimension 0
gives ``W_T``); scramble ``r`` of the ``c``-th QMC call of an instance uses
``default_rng(SeedSequence(seed, spawn_key=(crc32(b"qmc_scramble"), c, r)))``.

References are independent of ``simulation.variance_reduction``: analytic GBM
moments, the matrix ``min(t_i, t_j)``, ``scipy.stats`` laws, and ``scipy.stats.qmc``
called directly for the stream convention.

Tolerances (section 2.5): means at 4 standard errors. With ``R = 16`` the SE
comes from 16 independent scramble means, so the false alarm of ``|z| <= 4`` is
``P(|t_15| > 4) ~ 1.2e-3`` under normal block means. Seeds are fixed, so every
assertion is deterministic for a given NumPy/SciPy version.

Why no i.i.d. KS p-value on a QMC block: the points of one scramble are
dependent and far more uniform than an i.i.d. sample (for ``d = 1`` the star
discrepancy is at most ``1/N``), so the KS p-value is ~1 and has no i.i.d.
meaning. Instead (i) the deterministic net property is checked (KS statistic
``<= 1/N`` against the analytic law, exactly one point per cell ``[k/N, (k+1)/N)``)
and (ii) a valid KS test with ``p >= 1e-3`` is run on ``N = 1`` per scramble, where
the paths come from independent scrambles and are therefore i.i.d.

Units: prices in currency, time in years, mu in 1/year, sigma in 1/sqrt(year).
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr
from scipy.stats import qmc as scipy_qmc

from simulation.gbm import GbmSimulator, simulate_gbm, simulate_terminal_batched
from simulation.variance_reduction import qmc_mean

S0, MU, SIGMA, T = 100.0, 0.07, 0.3, 1.0
K_SE = 4.0
SOBOL_BITS = 30
STREAM = zlib.crc32(b"qmc_scramble")


def _expected_terminal(s0: float = S0, mu: float = MU, horizon: float = T) -> float:
    return s0 * math.exp(mu * horizon)


def _sim(steps: int, n_sims: int, seed: int | None, *, sigma: float = SIGMA,
         horizon: float = T, **kwargs) -> GbmSimulator:
    return GbmSimulator(S0, horizon, steps, n_sims, seed, mu=MU, sigma=sigma, **kwargs)


def _qmc_terminal(n_sims: int, seed: int | None, scrambles: int = 16, **kwargs) -> np.ndarray:
    return _sim(1, n_sims, seed, **kwargs).generate_terminal(
        variance_reduction="qmc", qmc_scrambles=scrambles)


def _qmc_paths(steps: int, n_sims: int, seed: int | None, scrambles: int = 16,
               **kwargs) -> np.ndarray:
    return _sim(steps, n_sims, seed, **kwargs).generate_paths(
        variance_reduction="qmc", qmc_scrambles=scrambles)


def _block_mean_and_se(values: np.ndarray, scrambles: int) -> tuple[float, float]:
    """Direct reference: mean of block means and std(block means, ddof=1)/sqrt(R)."""
    means = np.asarray(values, dtype=np.float64).reshape(scrambles, -1).mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(scrambles))


def _brownian_from_paths(paths: np.ndarray, horizon: float, sigma: float = SIGMA) -> np.ndarray:
    """W on the grid t_k = k*T/steps (rows 1..steps) from log prices."""
    steps = paths.shape[0] - 1
    times = horizon * np.arange(1, steps + 1) / steps
    return (np.log(paths[1:] / S0) - (MU - sigma**2 / 2) * times[:, None]) / sigma


def _sobol_cells(z: np.ndarray, log2_points: int) -> np.ndarray:
    """Cell index floor(u*N) of u = Phi(z), after snapping u to the 2^-30 grid.

    Scrambled 30-bit Sobol points are multiples of 2^-30 (the clip only moves
    u = 0 to 2^-53, which snaps back to 0). Snapping removes the ~1e-15 error of
    the price -> Z -> u round trip, so points on a cell boundary are classified
    exactly; the residual is checked to be far below half a grid step.
    """
    scaled = ndtr(z) * 2.0**SOBOL_BITS
    grid = np.rint(scaled)
    assert np.max(np.abs(scaled - grid)) < 1e-3
    return grid.astype(np.int64) >> (SOBOL_BITS - log2_points)


# ---------------------------------------------------------------------------
# (a) Terminal mean within 4 SE (R = 16, N = 2^12), terminal and paths[-1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [2301, 2302, 2303])
def test_terminal_mean_within_four_se(seed):
    scrambles, points = 16, 2**12
    terminal = _qmc_terminal(scrambles * points, seed, scrambles)
    estimate, se = _block_mean_and_se(terminal, scrambles)
    assert se > 0
    assert abs(estimate - _expected_terminal()) <= K_SE * se


@pytest.mark.parametrize("seed", [2311, 2312])
def test_paths_terminal_row_mean_within_four_se(seed):
    scrambles, points = 16, 2**12
    paths = _qmc_paths(12, scrambles * points, seed, scrambles)
    estimate, se = _block_mean_and_se(paths[-1], scrambles)
    assert se > 0
    assert abs(estimate - _expected_terminal()) <= K_SE * se
    # Every grid point has the analytic mean S0*exp(mu*t_k).
    times = T * np.arange(1, 13) / 12
    for k, t_k in enumerate(times, start=1):
        est_k, se_k = _block_mean_and_se(paths[k], scrambles)
        assert abs(est_k - S0 * math.exp(MU * t_k)) <= K_SE * se_k


# ---------------------------------------------------------------------------
# (b) RMS error over 50 seeds: QMC versus plain MC with the same R*N
# ---------------------------------------------------------------------------
#
# Threshold: RMS(QMC)/RMS(MC) < 0.5, i.e. an MSE ratio below 0.25. If QMC were
# no better than plain MC (equal error variance, roughly normal errors), the
# ratio of the two 50-seed mean squared errors would follow F(50, 50) and
# P(F < 0.25) ~ 1.3e-6, so a false pass is essentially impossible. Measured on
# these seeds (2400-2449, R = 16, N = 2^12, SciPy 1.18.1 / NumPy 2.5.5): RMS
# ratio 0.0105 for the terminal mean (variance gain ~9100); with 12 steps
# 0.0116 for paths[-1] and 0.0087 for the arithmetic average. The threshold
# leaves a margin of ~40x. No specific gain is fixed.

RMS_SEEDS = range(2400, 2450)
RMS_RATIO_MAX = 0.5


def _rms_errors(estimator, seeds, target: float) -> float:
    errors = np.array([estimator(seed) - target for seed in seeds])
    return float(np.sqrt(np.mean(errors**2)))


def test_qmc_terminal_rms_error_beats_plain_mc():
    scrambles, points = 16, 2**12
    n_sims = scrambles * points
    target = _expected_terminal()
    rms_qmc = _rms_errors(lambda s: _qmc_terminal(n_sims, s, scrambles).mean(),
                          RMS_SEEDS, target)
    rms_mc = _rms_errors(
        lambda s: simulate_terminal_batched(S0, MU, SIGMA, T, n_sims, s).mean(),
        RMS_SEEDS, target)
    assert rms_qmc > 0
    assert rms_qmc / rms_mc < RMS_RATIO_MAX


@pytest.mark.slow
def test_qmc_paths_rms_error_beats_plain_mc():
    """Full grid with the bridge: terminal mean and arithmetic-average mean."""
    scrambles, points, steps = 16, 2**12, 12
    n_sims = scrambles * points
    times = T * np.arange(1, steps + 1) / steps
    avg_target = float(S0 * np.mean(np.exp(MU * times)))
    results = {"qmc": ([], []), "mc": ([], [])}
    for seed in RMS_SEEDS:
        for mode in ("qmc", "mc"):
            if mode == "qmc":
                paths = _qmc_paths(steps, n_sims, seed, scrambles)
            else:
                paths = simulate_gbm(S0, MU, SIGMA, T, steps, n_sims, seed)
            results[mode][0].append(paths[-1].mean() - _expected_terminal())
            results[mode][1].append(paths[1:].mean() - avg_target)
    for index in (0, 1):
        rms_qmc = math.sqrt(np.mean(np.square(results["qmc"][index])))
        rms_mc = math.sqrt(np.mean(np.square(results["mc"][index])))
        assert 0 < rms_qmc / rms_mc < RMS_RATIO_MAX


# ---------------------------------------------------------------------------
# (c) Brownian bridge: exact covariance and statistical checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [1, 2, 3, 5, 7, 8, 12, 64, 252])
@pytest.mark.parametrize("horizon", [1.0, 1.3])
def test_bridge_matrix_reproduces_min_covariance(steps, horizon):
    """Applying the bridge to Z = I gives its matrix A; A A^T = min(t_i, t_j)."""
    from simulation.variance_reduction import brownian_bridge

    matrix = brownian_bridge(np.eye(steps), horizon)
    times = horizon * np.arange(1, steps + 1) / steps
    reference = np.minimum.outer(times, times)
    assert matrix.shape == (steps, steps)
    np.testing.assert_allclose(matrix @ matrix.T, reference, rtol=0, atol=1e-12)
    # Dimension 0 alone sets W_T = sqrt(T)*Z_0 (first, best-distributed coordinate)
    # and enters W_t linearly as t/sqrt(T).
    np.testing.assert_allclose(matrix[-1], np.eye(steps)[0] * math.sqrt(horizon),
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(matrix[:, 0], times / math.sqrt(horizon),
                               rtol=0, atol=1e-14)


def test_bridge_variance_matches_time_on_every_grid_point():
    """E[W_t^2] = t and E[W_t] = 0 at each grid point, 4 SE across scrambles.

    12 points x 2 moments at |z| <= 4 with t_15 errors: family false alarm about
    2.9e-2 under normal block means; deterministic for the fixed seed.
    """
    scrambles, points, steps, horizon = 16, 2**12, 12, 1.3
    paths = _qmc_paths(steps, scrambles * points, 2501, scrambles, horizon=horizon)
    brownian = _brownian_from_paths(paths, horizon)
    times = horizon * np.arange(1, steps + 1) / steps
    for row, t_k in zip(brownian, times):
        second, se2 = _block_mean_and_se(row**2, scrambles)
        assert abs(second - t_k) <= K_SE * se2
        first, se1 = _block_mean_and_se(row, scrambles)
        assert abs(first) <= K_SE * se1


def test_bridge_increments_uncorrelated_with_variance_dt():
    """E[dW_i dW_j]/dt = delta_ij, Bonferroni over all pairs on the R scrambles.

    Each block mean is unbiased and the R = 16 blocks are independent, so
    (estimate - target)/SE ~ t_15; the threshold keeps the family false alarm at
    1e-3 over the 78 entries of the upper triangle.
    """
    scrambles, points, steps, horizon = 16, 2**12, 12, 1.3
    paths = _qmc_paths(steps, scrambles * points, 2502, scrambles, horizon=horizon)
    brownian = _brownian_from_paths(paths, horizon)
    increments = np.diff(np.vstack([np.zeros((1, brownian.shape[1])), brownian]), axis=0)
    dt = horizon / steps
    pairs = [(i, j) for i in range(steps) for j in range(i, steps)]
    threshold = stats.t.ppf(1 - 1e-3 / (2 * len(pairs)), scrambles - 1)
    plain_sd = 1 / math.sqrt(scrambles * points)  # i.i.d. sd of a correlation
    for i, j in pairs:
        product = increments[i] * increments[j] / dt
        estimate, se = _block_mean_and_se(product, scrambles)
        target = 1.0 if i == j else 0.0
        assert abs(estimate - target) <= threshold * se
        if i != j:
            corr = np.corrcoef(increments[i], increments[j])[0, 1]
            assert abs(corr) < 4 * plain_sd


def test_independent_scrambles_give_iid_gbm_paths_ks():
    """N = 1 per scramble: columns come from independent scrambles, hence i.i.d.

    A single scrambled Sobol point is uniform on the 2^-30 grid in every
    coordinate, so the paths have the exact GBM law (up to that grid) and an
    i.i.d. KS test is valid: p >= 1e-3 for each grid marginal (8 tests, family
    false alarm <= 8e-3) and for the pooled standardized log increments.
    """
    steps, scrambles, horizon, sigma = 8, 1024, 1.3, 0.4
    paths = _qmc_paths(steps, scrambles, 2601, scrambles, horizon=horizon, sigma=sigma)
    dt = horizon / steps
    logs = np.log(paths)
    for k in range(1, steps + 1):
        t_k = k * dt
        loc = math.log(S0) + (MU - sigma**2 / 2) * t_k
        assert stats.kstest(logs[k], stats.norm(loc, sigma * math.sqrt(t_k)).cdf).pvalue >= 1e-3
    standardized = (np.diff(logs, axis=0) - (MU - sigma**2 / 2) * dt) / (sigma * math.sqrt(dt))
    assert stats.kstest(standardized.ravel(), "norm").pvalue >= 1e-3


# ---------------------------------------------------------------------------
# (d) Layout and net structure
# ---------------------------------------------------------------------------

def test_paths_shape_and_initial_row():
    scrambles, points, steps = 4, 8, 5
    paths = _qmc_paths(steps, scrambles * points, 7, scrambles)
    assert paths.shape == (steps + 1, scrambles * points)
    assert paths.dtype == np.float64
    assert np.all(paths[0] == S0)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
    terminal = _qmc_terminal(scrambles * points, 7, scrambles)
    assert terminal.shape == (scrambles * points,) and terminal.dtype == np.float64


@pytest.mark.parametrize("log2_points", [0, 4, 12])
def test_terminal_blocks_are_one_dimensional_nets(log2_points):
    """Each block [r*N, (r+1)*N) has exactly one point in every [k/N, (k+1)/N).

    A scrambled 2^m-point Sobol set in d = 1 is a (0, m, 1)-net; this also pins
    the contiguous block layout (a strided layout would break the per-block
    stratification). Its star discrepancy is at most 1/N, so the KS statistic of
    each block (and of the pooled sample) against the analytic law is <= 1/N.
    """
    scrambles, points = 16, 2**log2_points
    terminal = _qmc_terminal(scrambles * points, 2701 + log2_points, scrambles)
    z = (np.log(terminal / S0) - (MU - SIGMA**2 / 2) * T) / (SIGMA * math.sqrt(T))
    cells = _sobol_cells(z, log2_points).reshape(scrambles, points)
    for block in cells:
        assert np.array_equal(np.sort(block), np.arange(points))
    law = stats.lognorm(s=SIGMA * math.sqrt(T), scale=S0 * math.exp((MU - SIGMA**2 / 2) * T))
    for block in terminal.reshape(scrambles, points):
        assert stats.kstest(block, law.cdf).statistic <= 1 / points + 1e-9
    assert stats.kstest(terminal, law.cdf).statistic <= 1 / points + 1e-9


def test_paths_first_two_dimensions_form_two_dimensional_net():
    """steps = 2: dimension 0 -> W_T, dimension 1 -> midpoint W_{T/2}.

    W_{T/2} = W_T/2 + sqrt(T/4)*Z_1. The first two Sobol coordinates form a
    (0, m, 2)-net preserved by scrambling: every elementary box of area 1/N
    holds exactly one point of each block.
    """
    scrambles, log2_points, horizon = 4, 6, 1.3
    points = 2**log2_points
    paths = _qmc_paths(2, scrambles * points, 2801, scrambles, horizon=horizon)
    brownian = _brownian_from_paths(paths, horizon)
    z0 = brownian[1] / math.sqrt(horizon)
    z1 = (brownian[0] - brownian[1] / 2) / math.sqrt(horizon / 4)
    grid0 = _sobol_cells(z0, SOBOL_BITS).reshape(scrambles, points)
    grid1 = _sobol_cells(z1, SOBOL_BITS).reshape(scrambles, points)
    for g0, g1 in zip(grid0, grid1):
        for bits0 in range(log2_points + 1):
            bits1 = log2_points - bits0
            boxes = ((g0 >> (SOBOL_BITS - bits0)) << bits1) + (g1 >> (SOBOL_BITS - bits1))
            assert np.array_equal(np.sort(boxes), np.arange(points))


def test_paths_terminal_row_blocks_are_nets():
    scrambles, log2_points = 8, 10
    paths = _qmc_paths(6, scrambles * 2**log2_points, 2802, scrambles)
    z = (np.log(paths[-1] / S0) - (MU - SIGMA**2 / 2) * T) / (SIGMA * math.sqrt(T))
    for block in _sobol_cells(z, log2_points).reshape(scrambles, -1):
        assert np.array_equal(np.sort(block), np.arange(2**log2_points))


# ---------------------------------------------------------------------------
# (e) Streams and reproducibility
# ---------------------------------------------------------------------------

def _reference_normals(seed: int, call: int, scramble: int, dimension: int,
                       log2_points: int) -> np.ndarray:
    """Documented stream convention, written directly against SciPy."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(STREAM, call, scramble)))
    u = scipy_qmc.Sobol(dimension, scramble=True, bits=SOBOL_BITS, rng=rng).random_base2(log2_points)
    return stats.norm.ppf(np.clip(u, 2.0**-53, 1 - 2.0**-53))


def test_stream_convention_matches_documented_seed_sequence():
    """Terminal = call 0; a following paths call is call 1 (per-instance counter)."""
    seed, scrambles, log2_points = 2901, 4, 5
    points = 2**log2_points
    sim = _sim(3, scrambles * points, seed)
    terminal = sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=scrambles)
    expected = np.concatenate([
        S0 * np.exp((MU - SIGMA**2 / 2) * T
                    + SIGMA * math.sqrt(T) * _reference_normals(seed, 0, r, 1, log2_points)[:, 0])
        for r in range(scrambles)])
    np.testing.assert_allclose(terminal, expected, rtol=1e-12, atol=0)

    paths = sim.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    w_terminal = np.concatenate([
        math.sqrt(T) * _reference_normals(seed, 1, r, 3, log2_points)[:, 0]
        for r in range(scrambles)])
    expected_last = S0 * np.exp((MU - SIGMA**2 / 2) * T + SIGMA * w_terminal)
    np.testing.assert_allclose(paths[-1], expected_last, rtol=1e-12, atol=0)


def test_same_seed_new_instance_reproduces_and_repeated_calls_differ():
    n_sims, scrambles = 64, 4
    first = _sim(4, n_sims, 3001)
    second = _sim(4, n_sims, 3001)
    a1 = first.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    a2 = first.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    b1 = second.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    b2 = second.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)
    assert not np.array_equal(a1, a2)
    t1 = _qmc_terminal(n_sims, 3002, scrambles)
    np.testing.assert_array_equal(t1, _qmc_terminal(n_sims, 3002, scrambles))
    assert not np.array_equal(t1, _qmc_terminal(n_sims, 3003, scrambles))
    # Wrappers equal a fresh instance with the same seed.
    np.testing.assert_array_equal(
        simulate_gbm(S0, MU, SIGMA, T, 4, n_sims, 3001, variance_reduction="qmc",
                     qmc_scrambles=scrambles), a1)
    np.testing.assert_array_equal(
        simulate_terminal_batched(S0, MU, SIGMA, T, n_sims, 3002, variance_reduction="qmc",
                                  qmc_scrambles=scrambles), t1)


def test_qmc_does_not_consume_instance_rng():
    n_sims = 32
    plain_paths = _sim(4, n_sims, 3101).generate_paths()
    plain_terminal = _sim(4, n_sims, 3101).generate_terminal()
    sim = _sim(4, n_sims, 3101)
    sim.generate_paths(variance_reduction="qmc", qmc_scrambles=2)
    sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=2)
    np.testing.assert_array_equal(sim.generate_paths(), plain_paths)
    sim2 = _sim(4, n_sims, 3101)
    sim2.generate_terminal(variance_reduction="qmc", qmc_scrambles=4)
    np.testing.assert_array_equal(sim2.generate_terminal(), plain_terminal)


def test_seed_none_draws_fresh_entropy_per_instance():
    a = _sim(3, 16, None)
    b = _sim(3, 16, None)
    a1 = a.generate_paths(variance_reduction="qmc", qmc_scrambles=2)
    a2 = a.generate_paths(variance_reduction="qmc", qmc_scrambles=2)
    b1 = b.generate_paths(variance_reduction="qmc", qmc_scrambles=2)
    assert not np.array_equal(a1, a2)
    assert not np.array_equal(a1, b1)


# ---------------------------------------------------------------------------
# (f) Validation and rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_sims, scrambles", [(48, 16), (100, 16), (16, 32), (24, 2), (12, 1)])
def test_points_per_scramble_must_be_power_of_two(n_sims, scrambles):
    sim = _sim(3, n_sims, 1)
    with pytest.raises(ValueError):
        sim.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)
    with pytest.raises(ValueError):
        sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=scrambles)


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.0, "16", None, np.float64(4.0)])
def test_invalid_scramble_count_rejected(bad):
    sim = _sim(3, 16, 1)
    with pytest.raises(ValueError):
        sim.generate_paths(variance_reduction="qmc", qmc_scrambles=bad)
    with pytest.raises(ValueError):
        sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=bad)
    with pytest.raises(ValueError):
        qmc_mean(np.ones(16), bad)


def test_numpy_integer_scramble_count_accepted():
    terminal = _qmc_terminal(16, 1, np.int64(4))
    assert terminal.shape == (16,)


def test_steps_above_sobol_dimension_limit_rejected():
    sim = _sim(scipy_qmc.Sobol.MAXDIM + 1, 2, 1)
    with pytest.raises(ValueError, match="21201|Sobol"):
        sim.generate_paths(variance_reduction="qmc", qmc_scrambles=2)


@pytest.mark.slow
def test_steps_at_sobol_dimension_limit_accepted():
    """steps = MAXDIM with N = 1, R = 1: measured ~2 s and ~160 MB peak (SciPy
    scrambling matrices), hence slow."""
    steps = scipy_qmc.Sobol.MAXDIM
    paths = _qmc_paths(steps, 1, 1, 1)
    assert paths.shape == (steps + 1, 1)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)


def test_t_student_with_qmc_rejected():
    sim = _sim(4, 16, 1, df=5.0)
    with pytest.raises(ValueError):
        sim.generate_paths("t_student", variance_reduction="qmc", qmc_scrambles=4)
    assert sim.generate_paths("t_student").shape == (5, 16)


def test_terminal_memory_budget_includes_sobol_workspace():
    """Documented budget: 8*n_sims + 16*N bytes (N = n_sims / R)."""
    n_sims, scrambles = 32, 2
    required = 8 * n_sims + 16 * (n_sims // scrambles)
    sim = _sim(1, n_sims, 1)
    with pytest.raises(ValueError):
        sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=scrambles,
                              max_working_bytes=required - 1)
    out = sim.generate_terminal(variance_reduction="qmc", qmc_scrambles=scrambles,
                                max_working_bytes=required)
    assert out.shape == (n_sims,)


# ---------------------------------------------------------------------------
# (g) Finiteness, clipping and degenerate parameters
# ---------------------------------------------------------------------------

def test_clip_maps_endpoints_to_finite_normals(monkeypatch):
    """Force Sobol to emit 0 and 1: the clip [2^-53, 1 - 2^-53] keeps Z finite."""
    forced = np.array([0.0, 2.0**-60, 0.5, 1.0])

    def fake_random_base2(self, m):
        assert 2**m == forced.size
        return np.tile(forced[:, None], (1, self.d))

    monkeypatch.setattr(scipy_qmc.Sobol, "random_base2", fake_random_base2)
    z_expected = stats.norm.ppf(np.clip(forced, 2.0**-53, 1 - 2.0**-53))
    assert np.all(np.isfinite(z_expected))
    terminal = _qmc_terminal(4, 1, 1)
    np.testing.assert_allclose(
        terminal, S0 * np.exp((MU - SIGMA**2 / 2) * T + SIGMA * math.sqrt(T) * z_expected),
        rtol=1e-12, atol=0)
    paths = _qmc_paths(3, 4, 1, 1)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
    np.testing.assert_allclose(
        paths[-1], S0 * np.exp((MU - SIGMA**2 / 2) * T + SIGMA * math.sqrt(T) * z_expected),
        rtol=1e-12, atol=0)


def test_many_scrambles_stay_finite_and_bounded():
    """|Z| <= Phi^-1(1 - 2^-53) ~ 8.2 over 512 scrambles x 2^10 points."""
    scrambles, points = 512, 2**10
    terminal = _qmc_terminal(scrambles * points, 3201, scrambles)
    assert np.all(np.isfinite(terminal)) and np.all(terminal > 0)
    z = (np.log(terminal / S0) - (MU - SIGMA**2 / 2) * T) / (SIGMA * math.sqrt(T))
    assert np.max(np.abs(z)) <= -stats.norm.ppf(2.0**-53) + 1e-9


def test_sigma_zero_is_deterministic_growth():
    steps, n_sims = 6, 16
    paths = _qmc_paths(steps, n_sims, 1, 4, sigma=0.0)
    times = T * np.arange(steps + 1) / steps
    np.testing.assert_allclose(paths, np.broadcast_to(S0 * np.exp(MU * times)[:, None],
                                                       paths.shape), rtol=1e-14, atol=0)
    terminal = _qmc_terminal(n_sims, 1, 4, sigma=0.0)
    np.testing.assert_allclose(terminal, _expected_terminal(), rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# (h) qmc_mean: estimator, SE and validation
# ---------------------------------------------------------------------------

def test_qmc_mean_uses_contiguous_blocks():
    """values = [0]*N + [1]*N + ...: block means are 0..R-1 (strided would give
    equal means and SE = 0)."""
    scrambles, points = 5, 8
    values = np.repeat(np.arange(scrambles, dtype=np.float64), points)
    result = qmc_mean(values, scrambles)
    assert result["scrambles"] == scrambles
    assert result["points_per_scramble"] == points
    assert result["estimate"] == pytest.approx(2.0, rel=1e-15)
    assert result["standard_error"] == pytest.approx(
        math.sqrt(2.5) / math.sqrt(scrambles), rel=1e-14)
    assert result["note"] is None


def test_qmc_mean_matches_direct_calculation_on_simulated_values():
    scrambles = 16
    terminal = _qmc_terminal(scrambles * 2**8, 3301, scrambles)
    result = qmc_mean(terminal, scrambles)
    estimate, se = _block_mean_and_se(terminal, scrambles)
    assert result["estimate"] == pytest.approx(estimate, rel=1e-14)
    assert result["standard_error"] == pytest.approx(se, rel=1e-12)


def test_qmc_mean_single_scramble_has_no_standard_error():
    values = _qmc_terminal(2**6, 3401, 1)
    result = qmc_mean(values, 1)
    assert result["standard_error"] is None
    assert isinstance(result["note"], str) and result["note"]
    assert result["estimate"] == pytest.approx(float(np.mean(values)), rel=1e-14)


@pytest.mark.parametrize("values, scrambles", [
    (np.ones((4, 4)), 2),
    (np.array([1.0, np.nan, 2.0, 3.0]), 2),
    (np.array([1.0, np.inf, 2.0, 3.0]), 2),
    (np.ones(5), 2),
    (np.ones(1), 2),
    (np.array([], dtype=float), 1),
    (np.array([True, False]), 1),
    (np.array(["a", "b"]), 1),
])
def test_qmc_mean_rejects_invalid_values(values, scrambles):
    with pytest.raises(ValueError):
        qmc_mean(values, scrambles)


# ---------------------------------------------------------------------------
# (i) Standard error across seeds: calibration and interval coverage
# ---------------------------------------------------------------------------
#
# The R block means are i.i.d. and unbiased, so E[SE^2] = Var(estimate) exactly
# and D_s = (estimate_s - truth)^2 - SE_s^2 has mean 0 over seeds: a
# distribution-free check of the SE (4 SE of the mean of D, finite fourth moment
# assumed). Coverage of the t_{R-1} 95% interval is NOT nominal: for this smooth
# integrand the block means of scrambled Sobol (LMS + digital shift) are skewed
# (~1.6) and leptokurtic (excess kurtosis ~4-5), and the measured coverage with
# R = 16 was 0.91-0.93 (4000 seeds, N = 64: 0.929). The coverage assertion is
# therefore only a gross-error guard: [0.85, 0.99] detects an SE off by a
# factor of ~1.4 either way; it does not certify 95% coverage.

def _seed_study(seeds, scrambles: int, points: int) -> tuple[np.ndarray, float]:
    truth = _expected_terminal()
    q = stats.t.ppf(0.975, scrambles - 1)
    diffs, hits = [], 0
    for seed in seeds:
        terminal = _qmc_terminal(scrambles * points, seed, scrambles)
        estimate, se = _block_mean_and_se(terminal, scrambles)
        diffs.append((estimate - truth) ** 2 - se**2)
        hits += abs(estimate - truth) <= q * se
    return np.array(diffs), hits / len(seeds)


def _assert_se_calibrated(diffs: np.ndarray) -> None:
    se_of_mean = diffs.std(ddof=1) / math.sqrt(diffs.size)
    assert abs(diffs.mean()) <= K_SE * se_of_mean


def test_se_is_calibrated_across_seeds():
    diffs, coverage = _seed_study(range(10_000, 10_400), 16, 64)
    _assert_se_calibrated(diffs)
    assert 0.85 <= coverage <= 0.99


@pytest.mark.slow
@pytest.mark.parametrize("scrambles, points", [(16, 64), (16, 1024), (4, 256)])
def test_se_is_calibrated_across_seeds_full(scrambles, points):
    diffs, coverage = _seed_study(range(20_000, 22_000), scrambles, points)
    _assert_se_calibrated(diffs)
    if scrambles == 16:
        assert 0.85 <= coverage <= 0.99
