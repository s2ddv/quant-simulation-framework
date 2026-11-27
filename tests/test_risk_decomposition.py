"""CVaR component contributions (E6.4): identity, symmetry, ties, bootstrap.

Conventions (risk.statistics): portfolio loss L = sum_i L_i, VaR = linear
percentile at p, tail = {L >= VaR} (ties included), CVaR = mean(L | tail),
C_i = mean(L_i | tail), so sum_i C_i = CVaR by linearity (up to rounding).
Statistical tolerances: 4 standard errors (docs/plano-expansao.md Sec. 2.5).
"""

import zlib

import numpy as np
import pytest

from risk.decomposition import (
    contribution_batch_size,
    cvar_contribution_bootstrap,
    cvar_contributions,
    portfolio_cvar_contributions,
)
from risk.statistics import risk_summary
from simulation.multi_asset import MultiAssetGbmSimulator, portfolio_paths

K_SE = 4.0
_STREAM = zlib.crc32(b"cvar_contribution_bootstrap")  # documented spawn key


def _iid_paths(n_sims, seed, rho=0.0, steps=3):
    return MultiAssetGbmSimulator([100.0, 100.0], 1.0, steps, n_sims, seed,
                                  mu=[0.05, 0.05], sigma=[0.25, 0.25],
                                  correlation=[[1.0, rho], [rho, 1.0]]).generate_paths()


def _empirical_cvar(losses, p):
    var = np.percentile(losses, 100 * p)  # NumPy "linear", independent of the module
    return var, losses[losses >= var].mean(), losses >= var


# --------------------------------------------------------------------------- identity


@pytest.mark.parametrize("kind, weights, p", [
    ("initial_weights", [0.5, 0.5], 0.95), ("initial_weights", [1.3, -0.3], 0.99),
    ("quantities", [2.0, 0.5], 0.9), ("quantities", [1.0, -0.4], 0.975),
])
def test_identity_and_exact_agreement_with_risk_summary(kind, weights, p):
    paths = _iid_paths(20_000, 1, rho=0.3)
    v0 = 1000.0
    result = portfolio_cvar_contributions(paths, weights, kind, p, initial_value=v0)
    values = portfolio_paths(paths, weights, kind, initial_value=v0)
    summary = risk_summary(values, p)
    assert result["var"] == summary["var"]   # bitwise, same functions and mask
    assert result["cvar"] == summary["cvar"]
    scale = result["sum_identity_scale"]
    assert abs(sum(result["contributions"]) - result["cvar"]) <= 1e-12 * scale
    assert result["sum_identity_error"] <= 1e-12 * scale
    # Independent recomputation of the decomposition.
    q = np.asarray(result["quantities"])
    components = q * (paths[0, 0] - paths[-1])
    losses = values[0, 0] - values[-1]
    var, cvar, mask = _empirical_cvar(losses, p)
    assert result["var"] == pytest.approx(var, rel=1e-12)
    assert result["cvar"] == pytest.approx(cvar, rel=1e-12)
    np.testing.assert_allclose(result["contributions"], components[mask].mean(axis=0),
                               rtol=1e-12, atol=1e-12 * scale)
    assert result["n_tail"] == int(mask.sum()) and result["n"] == 20_000
    if kind == "initial_weights":
        assert result["initial_value"] == pytest.approx(v0, rel=1e-14)
        np.testing.assert_allclose(q, np.asarray(weights) * v0 / 100.0, rtol=1e-15)


def test_contributions_are_euler_derivatives_of_empirical_cvar():
    # For a fixed tail set, d CVaR(sum_j u_j L_j)/d u_i at u = 1 is exactly mean(L_i | tail).
    rng = np.random.default_rng(5)
    components = rng.standard_normal((5000, 3)) * [1.0, 2.0, 0.5] + [0.0, 0.1, -0.2]
    result = cvar_contributions(components, 0.95)
    h = 1e-7
    for i in range(3):
        up, down = np.ones(3), np.ones(3)
        up[i] += h
        down[i] -= h
        derivative = (_empirical_cvar(components @ up, 0.95)[1]
                      - _empirical_cvar(components @ down, 0.95)[1]) / (2 * h)
        assert derivative == pytest.approx(result["contributions"][i], rel=1e-5, abs=1e-6)


def test_column_swap_swaps_contributions_exactly():
    rng = np.random.default_rng(1)
    components = rng.standard_normal((3000, 2))
    a = cvar_contributions(components, 0.95)
    b = cvar_contributions(components[:, ::-1], 0.95)
    assert a["contributions"] == b["contributions"][::-1]
    assert a["cvar"] == b["cvar"]


# --------------------------------------------------------------------------- symmetry


@pytest.mark.parametrize("n_sims, seed", [(20_000, 11), pytest.param(200_000, 12, marks=pytest.mark.slow)])
def test_identical_independent_assets_have_equal_contributions(n_sims, seed):
    paths = _iid_paths(n_sims, seed)
    result = portfolio_cvar_contributions(paths, [0.5, 0.5], "initial_weights", 0.95,
                                          initial_value=200.0)
    c = np.asarray(result["contributions"])
    # SE of the difference from our own i.i.d. bootstrap (the module's SEs are per
    # component; C_1 and C_2 are negatively correlated in the tail, so the hypot of
    # the marginal SEs would not be the SE of the difference).
    q = np.asarray(result["quantities"])
    comps = q * (paths[0, 0] - paths[-1])
    losses = comps.sum(axis=1)
    rng = np.random.default_rng(seed + 1000)
    diffs = []
    for _ in range(300):
        idx = rng.integers(0, n_sims, n_sims)
        _, _, mask = _empirical_cvar(losses[idx], 0.95)
        tail = comps[idx][mask]
        diffs.append(tail[:, 0].mean() - tail[:, 1].mean())
    se_diff = float(np.std(diffs, ddof=1))
    assert abs(c[0] - c[1]) <= K_SE * se_diff
    # Each equals CVaR/2 within the bootstrap SE of the module.
    boot = cvar_contribution_bootstrap(comps, 0.95, n_bootstrap=200, seed=3)
    for ci, se in zip(c, boot["contribution_standard_errors"]):
        assert abs(ci - result["cvar"] / 2) <= K_SE * se


# --------------------------------------------------------------------------- ties


def test_ties_at_var_are_included_by_hand():
    total = np.array([0.0, 1, 2, 3, 4, 4, 4, 6, 7, 8])
    first = np.array([0.0, 1, 1, 1, 1, 3, 4, 2, 3, 4])
    components = np.column_stack([first, total - first])
    # p = 0.5: rank 4.5 between sorted[4] = 4 and sorted[5] = 4 -> VaR = 4;
    # tail = {4, 4, 4, 6, 7, 8} (rows 4..9), 6 rows; a strict ">" mask would keep 3.
    result = cvar_contributions(components, 0.5)
    assert result["var"] == 4.0
    assert result["n_tail"] == 6
    assert result["cvar"] == pytest.approx(33 / 6, abs=1e-15)
    assert result["contributions"] == pytest.approx([17 / 6, 16 / 6], abs=1e-15)
    # All-equal losses: everything is in the tail.
    flat = cvar_contributions(np.ones((7, 3)), 0.99)
    assert flat["n_tail"] == 7 and flat["contributions"] == [1.0, 1.0, 1.0]
    assert flat["cvar"] == 3.0


def test_negative_losses_and_mixed_sign_components():
    components = np.array([[-5.0, 1.0], [-4.0, 2.0], [-1.0, -1.0], [3.0, -2.5], [0.0, -10.0]])
    # L = [-4, -2, -2, 0.5, -10]; p = 0.5: sorted [-10,-4,-2,-2,0.5], rank 2 -> VaR = -2.
    result = cvar_contributions(components, 0.5)
    assert result["var"] == -2.0 and result["n_tail"] == 3
    assert result["cvar"] == pytest.approx(-3.5 / 3, abs=1e-15)
    assert result["contributions"] == pytest.approx([(-4 - 1 + 3) / 3, (2 - 1 - 2.5) / 3])


# --------------------------------------------------------------------------- bootstrap


def _reference_bootstrap(components, p, n_bootstrap, seed):
    """Re-implementation from the documented RNG contract (one replicate at a time)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_STREAM,)))
    losses = components.sum(axis=1)
    n = losses.size
    reps, cvars = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        _, cvar, mask = _empirical_cvar(losses[idx], p)
        reps.append(components[idx][mask].mean(axis=0))
        cvars.append(cvar)
    return np.std(reps, axis=0, ddof=1), float(np.std(cvars, ddof=1))


def test_bootstrap_reproducible_batch_independent_and_matches_reference():
    rng = np.random.default_rng(7)
    components = rng.standard_normal((1500, 3)) * [1.0, 0.5, 2.0]
    base = cvar_contribution_bootstrap(components, 0.9, n_bootstrap=60, seed=42)
    for batch in (1, 3, 59, 60, 1000):
        other = cvar_contribution_bootstrap(components, 0.9, n_bootstrap=60, seed=42,
                                            batch_size=batch)
        assert other["contribution_standard_errors"] == base["contribution_standard_errors"]
        assert other["cvar_standard_error"] == base["cvar_standard_error"]
    different = cvar_contribution_bootstrap(components, 0.9, n_bootstrap=60, seed=43)
    assert different["contribution_standard_errors"] != base["contribution_standard_errors"]
    assert all(se > 0 for se in base["contribution_standard_errors"])
    assert base["cvar_standard_error"] > 0
    assert base["n_bootstrap"] == 60 and base["seed"] == 42 and "PCG64" in base["rng"]
    assert "note" not in base
    ref_se, ref_cvar_se = _reference_bootstrap(components, 0.9, 60, 42)
    np.testing.assert_allclose(base["contribution_standard_errors"], ref_se, rtol=1e-10)
    assert base["cvar_standard_error"] == pytest.approx(ref_cvar_se, rel=1e-10)
    point = cvar_contributions(components, 0.9)
    for key in ("var", "cvar", "contributions", "n_tail"):
        assert base[key] == point[key]


def test_bootstrap_se_agrees_with_monte_carlo_replication():
    # Sampling SD of C_i over 200 independent replications vs. the bootstrap SE of one
    # sample. Each SD estimate has relative noise ~1/sqrt(2*200) = 5%; the bootstrap
    # of a tail mean at n_tail = 100 has O(1/sqrt(n_tail)) bias. Band [0.7, 1.4].
    n, p = 2000, 0.95
    rng = np.random.default_rng(2024)
    reps = []
    for _ in range(200):
        comps = rng.standard_normal((n, 2)) * [1.0, 2.0]
        reps.append(cvar_contributions(comps, p)["contributions"])
    mc_sd = np.std(reps, axis=0, ddof=1)
    sample = rng.standard_normal((n, 2)) * [1.0, 2.0]
    boot = cvar_contribution_bootstrap(sample, p, n_bootstrap=400, seed=1)
    ratio = np.asarray(boot["contribution_standard_errors"]) / mc_sd
    assert np.all((0.7 <= ratio) & (ratio <= 1.4)), ratio


def test_bootstrap_small_tail_note():
    components = np.arange(10.0).reshape(5, 2)
    result = cvar_contribution_bootstrap(components, 0.99, n_bootstrap=5, seed=0)
    assert result["n_tail"] == 1 and "fewer than 2" in result["note"]
    # The single tail row is the maximum.
    assert result["contributions"] == [8.0, 9.0]


def test_batch_size_helper():
    assert contribution_batch_size(10, 2) >= 1
    assert contribution_batch_size(10**8, 50) == 1
    assert contribution_batch_size(1000, 2) > contribution_batch_size(100_000, 2)


# --------------------------------------------------------------------------- invalid inputs


@pytest.mark.parametrize("bad", [
    np.ones(5), np.ones((0, 2)), np.ones((3, 0)), np.array([[1.0, np.nan], [0.0, 1.0]]),
    np.array([[1.0, np.inf], [0.0, 1.0]]), np.ones((3, 2), dtype=bool),
    np.array([["a", "b"]]), np.ones((2, 2, 2)),
])
def test_rejects_invalid_component_losses(bad):
    with pytest.raises(ValueError):
        cvar_contributions(bad)
    with pytest.raises(ValueError):
        cvar_contribution_bootstrap(bad, n_bootstrap=2, seed=0)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, True, np.nan, "0.9"])
def test_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        cvar_contributions(np.ones((4, 2)), confidence)
    with pytest.raises(ValueError):
        cvar_contribution_bootstrap(np.ones((4, 2)), confidence, n_bootstrap=2, seed=0)


def test_rejects_inconsistent_portfolio_losses():
    comps = np.random.default_rng(0).standard_normal((100, 2))
    row = comps.sum(axis=1)
    ok = cvar_contributions(comps, 0.9, portfolio_losses=row + 1e-12)
    assert ok["n"] == 100
    for bad in (row + 1e-3, row[:-1], np.append(row[:-1], np.nan), row.astype(bool),
                row.reshape(-1, 1)):
        with pytest.raises(ValueError, match="portfolio_losses"):
            cvar_contributions(comps, 0.9, portfolio_losses=bad)


@pytest.mark.parametrize("kwargs, message", [
    ({"n_bootstrap": 1}, "n_bootstrap"), ({"n_bootstrap": True}, "n_bootstrap"),
    ({"n_bootstrap": 10.0}, "n_bootstrap"), ({"seed": -1}, "seed"), ({"seed": True}, "seed"),
    ({"batch_size": 0}, "batch_size"), ({"batch_size": False}, "batch_size"),
])
def test_bootstrap_rejects_invalid_arguments(kwargs, message):
    args = {"n_bootstrap": 5, "seed": 0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        cvar_contribution_bootstrap(np.ones((10, 2)) + np.arange(10)[:, None], 0.9, **args)


def test_portfolio_contributions_reject_bad_portfolios():
    paths = _iid_paths(50, 0)
    with pytest.raises(ValueError, match="initial value must be positive"):
        portfolio_cvar_contributions(paths, [1.0, -1.0])
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_cvar_contributions(paths, [0.3, 0.3], "initial_weights")
    with pytest.raises(ValueError):
        portfolio_cvar_contributions(paths[:, :, :1], [1.0, 1.0])
    with pytest.raises(ValueError):
        portfolio_cvar_contributions(paths, [1.0, 1.0], confidence=1.0)
    single = portfolio_cvar_contributions(paths[:, :, :1], [1.0])
    summary = risk_summary(paths[:, :, 0])
    assert single["cvar"] == summary["cvar"] and single["var"] == summary["var"]
    assert single["contributions"][0] == pytest.approx(single["cvar"], rel=1e-14)
