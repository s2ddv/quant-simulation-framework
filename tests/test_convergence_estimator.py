"""Tests for ``validation.statistics.convergence_with_estimator`` (E2.4, offline, seeded).

Model (constant parameters, exact lognormal law)::

    S_T = S0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)
    E[S_T] = S0 * exp(mu*T)
    E_P[(S_T - K)^+] = S0*exp(mu*T)*Phi(d1) - K*Phi(d2),
        d1 = (ln(S0/K) + (mu + sigma^2/2)*T) / (sigma*sqrt(T)),  d2 = d1 - sigma*sqrt(T)

The call value is the undiscounted expectation under the *historical* measure P
(drift ``mu``), not a risk-neutral price; it is only an analytic target for the
estimators. Units: prices in currency, T in years, mu in 1/year, sigma in
1/sqrt(year).

Contract under test (docstring of ``convergence_with_estimator``): checkpoints
``{n_units, n_evaluations, estimate, standard_error, absolute_error,
relative_error}`` over nested prefixes of one run; ``n_evaluations = n_units *
evaluations_per_unit``; errors ``None`` without ``theoretical``;
``relative_error`` ``None`` when ``theoretical == 0``; JSON-safe with
``allow_nan=False``; non-finite estimator output raises ``ValueError`` naming
``n_units``; ``None`` SE is preserved. Layouts: antithetic ``[originals...,
mirrors...]`` -> ``(m, 2)`` pairs; QMC blocks per scramble -> ``(N, R)``.

References are independent of the implementation: analytic GBM moments and the
call formula via ``statistics.NormalDist``, the antithetic pair identity
``log S+ + log S- = 2 log S0 + 2 (mu - sigma^2/2) T``, the one-dimensional Sobol
net property, and plain NumPy means / ``std(ddof=1)/sqrt(n)``.

Tolerances (plan section 2.5): final errors within ``4 * SE``. For i.i.d. and
antithetic pairs (32768+ units) the false alarm is ~6e-5 per assertion under
asymptotic normality; for QMC with ``R = 16`` scrambles the SE has 15 degrees of
freedom, so ``P(|t_15| > 4) ~ 1.2e-3``. Equal-cost SE comparisons use margins far
below the expected gains (see each test). Seeds are fixed, so assertions are
deterministic for a given NumPy/SciPy version. Prefixes are nested (correlated);
no monotone error decrease is required.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pytest
from scipy.special import ndtr

from simulation.gbm import GbmSimulator
from simulation.variance_reduction import (antithetic_mean, control_variate_mean,
                                           gbm_terminal_control_mean, qmc_mean)
from validation.statistics import (antithetic_estimator, antithetic_units,
                                   compare_terminal, control_estimator,
                                   convergence_analysis, convergence_with_estimator,
                                   mean_estimator, qmc_estimator, qmc_units,
                                   terminal_theory)

S0, MU, SIGMA, T, K = 100.0, 0.05, 0.2, 1.0, 100.0
BUDGET = 2**16
SCRAMBLES = 16
SEED = 2404
K_SE = 4.0
SOBOL_BITS = 30
KEYS = {"n_units", "n_evaluations", "estimate", "standard_error",
        "absolute_error", "relative_error"}


def _expected_terminal() -> float:
    return S0 * math.exp(MU * T)


def _expected_call() -> float:
    vol = SIGMA * math.sqrt(T)
    d1 = (math.log(S0 / K) + (MU + SIGMA**2 / 2) * T) / vol
    d2 = d1 - vol
    nd = NormalDist()
    return S0 * math.exp(MU * T) * nd.cdf(d1) - K * nd.cdf(d2)


def _terminal(n: int, seed: int, **kwargs) -> np.ndarray:
    return GbmSimulator(S0, T, 1, n, seed, mu=MU, sigma=SIGMA).generate_terminal(**kwargs)


def _std_normal(terminal: np.ndarray) -> np.ndarray:
    return (np.log(terminal / S0) - (MU - SIGMA**2 / 2) * T) / (SIGMA * math.sqrt(T))


def _naive(values: np.ndarray) -> tuple[float, float]:
    """Independent i.i.d. reference: plain mean and std(ddof=1)/sqrt(n)."""
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def _constant(estimate, standard_error=0.0):
    return lambda prefix: {"estimate": estimate, "standard_error": standard_error}


@pytest.fixture(scope="module")
def samples():
    return {
        "none": _terminal(BUDGET, SEED),
        "antithetic": _terminal(BUDGET, SEED, variance_reduction="antithetic"),
        "qmc": _terminal(BUDGET, SEED, variance_reduction="qmc", qmc_scrambles=SCRAMBLES),
    }


# ---------------------------------------------------------------------------
# (a) Checkpoint contract and JSON safety


def test_checkpoint_contract_and_json_safety(samples):
    values = samples["none"]
    theory = _expected_terminal()
    result = convergence_with_estimator(values, mean_estimator, theoretical=theory,
                                        evaluations_per_unit=3)
    assert result and all(set(c) == KEYS for c in result)
    for c in result:
        assert isinstance(c["n_units"], int) and isinstance(c["n_evaluations"], int)
        assert c["n_evaluations"] == 3 * c["n_units"]
        assert c["absolute_error"] == pytest.approx(abs(c["estimate"] - theory), abs=0, rel=1e-15)
        assert c["relative_error"] == pytest.approx(c["absolute_error"] / theory, rel=1e-15)
    json.dumps(result, allow_nan=False)


def test_without_theoretical_errors_are_none(samples):
    result = convergence_with_estimator(samples["none"][:100], mean_estimator)
    assert all(c["absolute_error"] is None and c["relative_error"] is None for c in result)
    assert all(c["standard_error"] is not None for c in result)
    json.dumps(result, allow_nan=False)


def test_zero_theoretical_keeps_absolute_and_drops_relative_error():
    values = np.array([-1.0, 0.5, 2.0, -0.25, 0.0, 1.5])
    result = convergence_with_estimator(values, mean_estimator, theoretical=0.0)
    for c in result:
        assert c["relative_error"] is None
        assert c["absolute_error"] == abs(c["estimate"])
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# (b) Equivalence with convergence_analysis on i.i.d. data


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 13, BUDGET])
def test_mean_estimator_matches_convergence_analysis(samples, n):
    values = samples["none"][:n]
    theory = terminal_theory(S0, MU, SIGMA, T)
    old = convergence_analysis(values, theory)
    new = convergence_with_estimator(values, mean_estimator, theoretical=theory["mean"])
    assert [c["n_sims"] for c in old] == [c["n_units"] for c in new]
    assert [c["n_evaluations"] for c in new] == [c["n_units"] for c in new]
    counts = [c["n_units"] for c in new]
    assert counts[-1] == n and counts == sorted(set(counts)) and len(counts) <= 13
    for o, w in zip(old, new):
        assert w["estimate"] == pytest.approx(o["mean"]["estimate"], rel=1e-15, abs=0)
        assert w["absolute_error"] == pytest.approx(o["mean"]["absolute_error"],
                                                    rel=1e-9, abs=1e-12)
        k = w["n_units"]
        if k == 1:
            assert w["standard_error"] is None
        else:
            mean, se = _naive(values[:k])
            assert w["estimate"] == pytest.approx(mean, rel=1e-14)
            assert w["standard_error"] == pytest.approx(se, rel=1e-12)


def test_convergence_analysis_contract_unchanged(samples):
    """The old function keeps its keys and its exact reference SE sqrt(Var/n)."""
    values = samples["none"][:1000]
    theory = terminal_theory(S0, MU, SIGMA, T)
    old = convergence_analysis(values, theory)
    exact_var = S0**2 * math.exp(2 * MU * T) * math.expm1(SIGMA**2 * T)
    for c in old:
        assert set(c) == {"n_sims", "mean", "variance", "quantiles", "mean_standard_error"}
        assert c["mean_standard_error"] == pytest.approx(math.sqrt(exact_var / c["n_sims"]),
                                                         rel=1e-12)
        assert c["mean"] == compare_terminal(values[:c["n_sims"]], theory)["mean"]


# ---------------------------------------------------------------------------
# (c) Schedules, explicit counts and min_units


def test_geometric_schedule_respects_min_units():
    values = np.arange(100, dtype=float)
    counts = [c["n_units"] for c in convergence_with_estimator(values, mean_estimator,
                                                               min_units=3)]
    assert counts[0] == 3 and counts[-1] == 100
    assert counts == sorted(set(counts)) and len(counts) <= 13


def test_power2_schedule():
    values = np.arange(16, dtype=float)
    counts = [c["n_units"] for c in convergence_with_estimator(values, mean_estimator,
                                                               schedule="power2")]
    assert counts == [1, 2, 4, 8, 16]
    counts = [c["n_units"] for c in convergence_with_estimator(
        values, mean_estimator, schedule="power2", min_units=3)]
    assert counts == [4, 8, 16]
    with pytest.raises(ValueError, match="power-of-two"):
        convergence_with_estimator(np.arange(12.0), mean_estimator, schedule="power2")
    with pytest.raises(ValueError, match="power of two"):
        convergence_with_estimator(values, mean_estimator, schedule="power2", counts=[3, 16])
    result = convergence_with_estimator(values, mean_estimator, schedule="power2",
                                        counts=[2, 8, 16])
    assert [c["n_units"] for c in result] == [2, 8, 16]


def test_explicit_counts_are_used_verbatim():
    values = np.arange(10, dtype=float)
    result = convergence_with_estimator(values, mean_estimator, counts=[2, 3, 7, 10])
    assert [c["n_units"] for c in result] == [2, 3, 7, 10]
    for c in result:
        assert c["estimate"] == pytest.approx(np.mean(values[:c["n_units"]]))
    result = convergence_with_estimator(values, mean_estimator, counts=np.array([10]))
    assert [c["n_units"] for c in result] == [10]


@pytest.mark.parametrize("counts", [
    [], [2, 2, 10], [5, 3, 10], [2, 5], [2, 11], [0, 10], [[2, 10]],
    [2.0, 10.0], np.array([True, False]), "2,10",
])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):
        convergence_with_estimator(np.arange(10.0), mean_estimator, counts=counts)


def test_counts_below_min_units_and_min_units_above_total():
    with pytest.raises(ValueError, match="min_units"):
        convergence_with_estimator(np.arange(10.0), mean_estimator, counts=[2, 10],
                                   min_units=3)
    with pytest.raises(ValueError, match="min_units"):
        convergence_with_estimator(np.arange(10.0), mean_estimator, min_units=11)


def test_same_sample_control_needs_min_units_three(samples):
    y = np.maximum(samples["none"][:200] - K, 0.0)
    units = np.stack([y, samples["none"][:200]], axis=1)
    estimator = control_estimator(gbm_terminal_control_mean(S0, MU, T))
    with pytest.raises(ValueError):  # geometric grid starts at 2 < 3
        convergence_with_estimator(units, estimator)
    result = convergence_with_estimator(units, estimator, min_units=3)
    assert result[0]["n_units"] == 3 and result[-1]["n_units"] == 200
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# (d) Adapters follow the real sampler layouts


def test_antithetic_units_pair_originals_with_mirrors(samples):
    values = samples["antithetic"]
    m = values.size // 2
    units = antithetic_units(values)
    assert units.shape == (m, 2)
    np.testing.assert_array_equal(units[:, 0], values[:m])
    np.testing.assert_array_equal(units[:, 1], values[m:])
    # Independent pair identity: Z_- = -Z_+ for every row.
    z = _std_normal(units)
    np.testing.assert_allclose(z[:, 0] + z[:, 1], 0.0, atol=1e-9)
    # A prefix of pairs is not a prefix of the flat vector: the first 2k
    # entries are k+k originals with no mirrors (no pair identity holds).
    k = 512
    flat = _std_normal(values[:2 * k])
    assert np.min(np.abs(flat[:k] + flat[k:])) > 1e-6
    pair_mean = float(np.mean((values[:k] + values[m:m + k]) / 2))
    result = convergence_with_estimator(units, antithetic_estimator, counts=[k, m],
                                        evaluations_per_unit=2)
    assert result[0]["estimate"] == pytest.approx(pair_mean, rel=1e-14)
    assert result[0]["n_evaluations"] == 2 * k
    assert abs(result[0]["estimate"] - float(np.mean(values[:2 * k]))) > 1e-6


def _sobol_cells(z: np.ndarray, log2_points: int) -> np.ndarray:
    """floor(u * 2^log2_points) of u = Phi(z), snapped to the 2^-30 Sobol grid."""
    scaled = ndtr(z) * 2.0**SOBOL_BITS
    grid = np.rint(scaled)
    assert np.max(np.abs(scaled - grid)) < 1e-3
    return grid.astype(np.int64) >> (SOBOL_BITS - log2_points)


def test_qmc_units_rows_are_points_of_each_scramble(samples):
    values = samples["qmc"]
    points = BUDGET // SCRAMBLES
    units = qmc_units(values, SCRAMBLES)
    assert units.shape == (points, SCRAMBLES)
    for r in (0, 5, SCRAMBLES - 1):
        np.testing.assert_array_equal(units[:, r], values[r * points:(r + 1) * points])
    # Every power-of-two prefix of rows is a (0, m, 1)-net in each column.
    z = _std_normal(units)
    for log2_n in (0, 3, 7, 12):
        n = 2**log2_n
        cells = _sobol_cells(z[:n], log2_n)
        for r in range(SCRAMBLES):
            assert np.array_equal(np.sort(cells[:, r]), np.arange(n))
    # A wrong (row-major) reshape mixes points and loses the net structure.
    wrong = _std_normal(values.reshape(points, SCRAMBLES)[:64])
    cells = _sobol_cells(wrong, 6)
    assert not all(np.array_equal(np.sort(cells[:, r]), np.arange(64))
                   for r in range(SCRAMBLES))


# ---------------------------------------------------------------------------
# (e) Estimators on the full prefix equal the direct calls


def test_full_prefix_matches_direct_estimators(samples):
    anti, q, plain = samples["antithetic"], samples["qmc"], samples["none"]
    last = convergence_with_estimator(antithetic_units(anti), antithetic_estimator,
                                      evaluations_per_unit=2)[-1]
    direct = antithetic_mean(anti)
    assert last["estimate"] == direct["estimate"]
    assert last["standard_error"] == direct["standard_error"]
    assert last["n_units"] == direct["pairs"] and last["n_evaluations"] == anti.size

    last = convergence_with_estimator(qmc_units(q, SCRAMBLES), qmc_estimator,
                                      schedule="power2",
                                      evaluations_per_unit=SCRAMBLES)[-1]
    direct = qmc_mean(q, SCRAMBLES)
    assert last["estimate"] == direct["estimate"]
    assert last["standard_error"] == direct["standard_error"]
    assert last["n_evaluations"] == q.size

    y = np.maximum(plain - K, 0.0)
    units = np.stack([y, plain], axis=1)
    cm = gbm_terminal_control_mean(S0, MU, T)
    for kwargs in ({}, {"pilot_fraction": 0.1}, {"beta": 0.6}):
        # Explicit final count: a pilot fraction needs >= 20 units per prefix.
        last = convergence_with_estimator(units, control_estimator(cm, **kwargs),
                                          counts=[units.shape[0]])[-1]
        direct = control_variate_mean(y, plain, cm, **kwargs)
        assert last["estimate"] == direct["estimate"]
        assert last["standard_error"] == direct["standard_error"]


# ---------------------------------------------------------------------------
# (f) Equal-cost comparison: none vs antithetic vs QMC


def _final_rows(samples, payoff, theory):
    plain, anti, q = (payoff(samples[k]) for k in ("none", "antithetic", "qmc"))
    rows = {
        "none": convergence_with_estimator(plain, mean_estimator, theoretical=theory),
        "antithetic": convergence_with_estimator(
            antithetic_units(anti), antithetic_estimator, theoretical=theory,
            evaluations_per_unit=2),
        "qmc": convergence_with_estimator(
            qmc_units(q, SCRAMBLES), qmc_estimator, theoretical=theory,
            schedule="power2", evaluations_per_unit=SCRAMBLES),
    }
    for result in rows.values():
        json.dumps(result, allow_nan=False)
    return {k: v[-1] for k, v in rows.items()}


def test_equal_cost_terminal_mean(samples):
    """E[S_T]: SE(anti) < SE(none) and SE(qmc) < SE(anti) at 2^16 evaluations.

    Equal-cost theory for the antithetic SE ratio (independent of the code):
    SE_none^2 / SE_anti^2 = Var(S)/(Var(S) + Cov(S+, S-))
    = (e^{s} - 1)/(e^{s} + e^{-s} - 2), s = sigma^2 T, i.e. ~5.05 for s = 0.04.
    The SE estimates have relative sampling error ~1% at these sizes, so a 10%
    tolerance is generous. The QMC gain over antithetic is ~25 here (seeds
    11-13, 2404 gave 8.4-36); requiring a factor 4 leaves ample room for the
    ~18% relative noise of an SE built from R = 16 scramble means.
    """
    theory = _expected_terminal()
    rows = _final_rows(samples, lambda x: x, theory)
    assert {r["n_evaluations"] for r in rows.values()} == {BUDGET}
    s = SIGMA**2 * T
    ratio_theory = math.sqrt(math.expm1(s) / (math.expm1(s) + math.expm1(-s)))
    ratio = rows["none"]["standard_error"] / rows["antithetic"]["standard_error"]
    assert ratio == pytest.approx(ratio_theory, rel=0.10)
    assert rows["qmc"]["standard_error"] < rows["antithetic"]["standard_error"] / 4
    for name, row in rows.items():
        assert row["absolute_error"] <= K_SE * row["standard_error"], name


def test_equal_cost_call_under_p(samples):
    """Undiscounted E_P[(S_T - K)^+] with the analytic Black-Scholes form under P.

    The call is monotone in Z, so antithetics help, but less than for S_T
    (observed SE ratio ~1.41 for seeds 11-13, 2404); a threshold 1.2 leaves a
    margin well above the ~1% SE noise. QMC gains were 22-114 over antithetic;
    the threshold is 4 for the reason given in the terminal-mean test.
    """
    theory = _expected_call()
    rows = _final_rows(samples, lambda x: np.maximum(x - K, 0.0), theory)
    assert rows["none"]["standard_error"] / rows["antithetic"]["standard_error"] > 1.2
    assert rows["qmc"]["standard_error"] < rows["antithetic"]["standard_error"] / 4
    for name, row in rows.items():
        assert row["absolute_error"] <= K_SE * row["standard_error"], name


# ---------------------------------------------------------------------------
# (g) No monotonicity requirement


def test_error_may_increase_between_checkpoints(samples):
    """Nested prefixes are correlated; the observed error is not monotone."""
    result = convergence_with_estimator(samples["none"], mean_estimator,
                                        theoretical=_expected_terminal())
    errors = [c["absolute_error"] for c in result]
    assert any(b > a for a, b in zip(errors, errors[1:]))


# ---------------------------------------------------------------------------
# (h) Validation, None SE and read-only prefixes


@pytest.mark.parametrize("units", [
    np.array([1.0, np.nan]), np.array([1.0, np.inf]), np.array([True, False]),
    np.array([]), np.empty((0, 2)), np.float64(3.0), 3.0, np.array(["a", "b"]),
    [1 + 2j, 3j],
])
def test_invalid_units(units):
    with pytest.raises(ValueError):
        convergence_with_estimator(units, mean_estimator)


@pytest.mark.parametrize("estimator", [None, 5, "mean"])
def test_estimator_must_be_callable(estimator):
    with pytest.raises(ValueError, match="callable"):
        convergence_with_estimator(np.arange(4.0), estimator)


@pytest.mark.parametrize("schedule", ["linear", None, 2, "GEOMETRIC"])
def test_invalid_schedule(schedule):
    with pytest.raises(ValueError, match="schedule"):
        convergence_with_estimator(np.arange(4.0), mean_estimator, schedule=schedule)


@pytest.mark.parametrize("name", ["evaluations_per_unit", "min_units"])
@pytest.mark.parametrize("value", [0, -1, 1.5, 2.0, True, np.bool_(True), None, "2"])
def test_invalid_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        convergence_with_estimator(np.arange(4.0), mean_estimator, **{name: value})


def test_numpy_integer_parameters_are_accepted():
    result = convergence_with_estimator(np.arange(8.0), mean_estimator,
                                        evaluations_per_unit=np.int64(2),
                                        min_units=np.int32(2))
    assert result[-1]["n_evaluations"] == 16
    assert all(isinstance(c["n_evaluations"], int) for c in result)


@pytest.mark.parametrize("theoretical", [math.nan, math.inf, True, "1", 1j])
def test_invalid_theoretical(theoretical):
    with pytest.raises(ValueError, match="theoretical"):
        convergence_with_estimator(np.arange(4.0), mean_estimator, theoretical=theoretical)


@pytest.mark.parametrize("bad", [
    {"estimate": math.nan, "standard_error": 1.0},
    {"estimate": math.inf, "standard_error": 1.0},
    {"estimate": 1.0, "standard_error": math.nan},
    {"estimate": 1.0, "standard_error": math.inf},
    {"estimate": 1.0, "standard_error": -1e-3},
    {"estimate": True, "standard_error": 1.0},
    {"estimate": None, "standard_error": 1.0},
])
def test_non_finite_or_invalid_estimator_output_names_n_units(bad):
    with pytest.raises(ValueError, match="n_units=2"):
        convergence_with_estimator(np.arange(4.0), lambda p: bad, counts=[2, 4])


@pytest.mark.parametrize("bad", [None, 1.0, {"estimate": 1.0}, {"standard_error": 1.0}])
def test_estimator_must_return_mapping(bad):
    with pytest.raises(ValueError, match="mapping"):
        convergence_with_estimator(np.arange(4.0), lambda p: bad)


def test_overflowing_error_raises():
    with pytest.raises(ValueError, match="n_units"):
        convergence_with_estimator(np.arange(4.0), _constant(1.7e308), theoretical=-1.7e308)


def test_none_standard_error_preserved_qmc_one_scramble():
    values = _terminal(256, SEED, variance_reduction="qmc", qmc_scrambles=1)
    result = convergence_with_estimator(qmc_units(values, 1), qmc_estimator,
                                        theoretical=_expected_terminal(), schedule="power2")
    assert [c["n_units"] for c in result] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert all(c["standard_error"] is None for c in result)
    assert all(c["absolute_error"] is not None for c in result)
    assert "null" in json.dumps(result, allow_nan=False)


def test_none_standard_error_preserved_one_antithetic_pair(samples):
    units = antithetic_units(samples["antithetic"])[:8]
    result = convergence_with_estimator(units, antithetic_estimator, counts=[1, 2, 8],
                                        evaluations_per_unit=2)
    assert result[0]["standard_error"] is None
    assert result[1]["standard_error"] is not None
    json.dumps(result, allow_nan=False)


def test_prefix_is_read_only_and_input_untouched():
    values = np.linspace(1.0, 2.0, 32)
    original = values.copy()
    seen = []

    def estimator(prefix):
        assert not prefix.flags.writeable
        with pytest.raises(ValueError):
            prefix[0] = -1.0
        seen.append(prefix.shape[0])
        return {"estimate": float(np.mean(prefix)), "standard_error": None}

    convergence_with_estimator(values, estimator, schedule="power2")
    assert seen == [1, 2, 4, 8, 16, 32]
    np.testing.assert_array_equal(values, original)
    assert values.flags.writeable  # the caller's array keeps its flags

    units = np.stack([values, values[::-1]], axis=1)
    before = units.copy()
    convergence_with_estimator(units, antithetic_estimator, evaluations_per_unit=2)
    np.testing.assert_array_equal(units, before)
    assert units.flags.writeable
