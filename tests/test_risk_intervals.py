"""Tests for ``risk.intervals.prob_loss_interval`` (E1.1, offline, seeded).

What is tested is the Monte Carlo *estimator error* interval for
``p = P(S_T < s0)``: k = #{S_T < s0} ~ Binom(n, p) for i.i.d. terminal prices.

References used here are independent of the implementation under test:

* an inline Wilson score formula (textbook form, written here);
* the beta-quantile identity of the Clopper-Pearson bounds via
  ``scipy.stats.binom`` tail probabilities (not ``beta.ppf``);
* exact coverage obtained by summing the binomial pmf over every k for which
  the interval returned by ``prob_loss_interval`` contains p;
* the analytic GBM loss probability
  ``P(S_T < s0) = Phi(-(mu - sigma^2/2) * sqrt(T) / sigma)``.

Units: prices in currency, T in years, mu in 1/year, sigma in 1/sqrt(year).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import stats

from risk.intervals import prob_loss_interval
from risk.statistics import risk_summary
from simulation.gbm import simulate_gbm, simulate_terminal_batched

METHODS = ("wilson", "clopper_pearson")
LEVEL = 0.95

# GBM whose analytic loss probability is exactly P_LOSS:
# P(S_T < s0) = Phi(-(mu - sigma^2/2) sqrt(T) / sigma)  =>
# mu = sigma^2/2 - sigma * Phi^{-1}(P_LOSS) / sqrt(T).
S0, SIGMA, T = 100.0, 0.20, 1.0
P_LOSS = 0.3
MU = SIGMA**2 / 2 - SIGMA * stats.norm.ppf(P_LOSS) / math.sqrt(T)
N_PER_REP = 500


def _terminal_with_k_losses(k: int, n: int, s0: float = S0) -> np.ndarray:
    """Deterministic sample with exactly k strict losses (and n - k gains)."""
    return np.where(np.arange(n) < k, s0 - 1.0, s0 + 1.0)


def _wilson_reference(k: int, n: int, level: float) -> tuple[float, float]:
    """Textbook Wilson score interval, written independently of production."""
    z = stats.norm.isf((1 - level) / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = (z / (1 + z**2 / n)) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def _exact_coverage(n: int, p: float, method: str, level: float = LEVEL) -> float:
    """Exact coverage of prob_loss_interval: sum of Binom(n, p) pmf over k
    whose returned interval contains p."""
    pmf = stats.binom.pmf(np.arange(n + 1), n, p)
    covered = 0.0
    for k in range(n + 1):
        out = prob_loss_interval(_terminal_with_k_losses(k, n), S0, level, method)
        if out["lower"] <= p <= out["upper"]:
            covered += pmf[k]
    return covered


def _replicate_terminals(n_reps: int, seed: int) -> np.ndarray:
    """(n_reps, N_PER_REP) i.i.d. exact GBM terminals; rows are independent
    replicates (disjoint blocks of one i.i.d. stream, not nested prefixes)."""
    terminal = simulate_terminal_batched(S0, MU, SIGMA, T, n_reps * N_PER_REP, seed)
    return np.asarray(terminal).reshape(n_reps, N_PER_REP)


def _empirical_coverage(terminals: np.ndarray, method: str) -> float:
    hits = 0
    for row in terminals:
        out = prob_loss_interval(row, S0, LEVEL, method)
        hits += out["lower"] <= P_LOSS <= out["upper"]
    return hits / terminals.shape[0]


# --------------------------------------------------------------------------
# 0. Analytic setup sanity
# --------------------------------------------------------------------------

def test_gbm_parameters_give_analytic_loss_probability():
    d = (math.log(S0 / S0) - (MU - SIGMA**2 / 2) * T) / (SIGMA * math.sqrt(T))
    assert stats.norm.cdf(d) == pytest.approx(P_LOSS, abs=1e-12)


# --------------------------------------------------------------------------
# 1. Edge cases k = 0 and k = n
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 7, 500])
def test_k_zero_and_k_n_bounds(method, n):
    zero = prob_loss_interval(np.full(n, S0 + 1.0), S0, LEVEL, method)
    assert zero["k"] == 0 and zero["n"] == n and zero["estimate"] == 0.0
    assert zero["lower"] == 0.0
    assert 0.0 < zero["upper"] < 1.0

    full = prob_loss_interval(np.full(n, S0 - 1.0), S0, LEVEL, method)
    assert full["k"] == n and full["estimate"] == 1.0
    assert full["upper"] == 1.0
    assert 0.0 <= full["lower"] < 1.0

    for out in (zero, full):
        values = np.array([out["estimate"], out["lower"], out["upper"]])
        assert np.isfinite(values).all()
        assert (0.0 <= values).all() and (values <= 1.0).all()
        assert out["lower"] <= out["estimate"] <= out["upper"]


def test_k_zero_clopper_pearson_rule_of_three_closed_form():
    # k = 0: upper solves (1 - u)^n = a/2  =>  u = 1 - (a/2)^(1/n).
    n = 500
    out = prob_loss_interval(np.full(n, S0 + 1.0), S0, LEVEL, "clopper_pearson")
    assert out["upper"] == pytest.approx(1 - 0.025 ** (1 / n), rel=1e-9)
    full = prob_loss_interval(np.full(n, S0 - 1.0), S0, LEVEL, "clopper_pearson")
    assert full["lower"] == pytest.approx(0.025 ** (1 / n), rel=1e-9)


# --------------------------------------------------------------------------
# Independent reference values
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (3, 10), (150, 500), (499, 500)])
def test_wilson_matches_independent_formula(k, n):
    out = prob_loss_interval(_terminal_with_k_losses(k, n), S0, LEVEL, "wilson")
    lo, hi = _wilson_reference(k, n, LEVEL)
    assert out["lower"] == pytest.approx(lo, abs=1e-12)
    assert out["upper"] == pytest.approx(hi, abs=1e-12)


@pytest.mark.parametrize("k,n", [(1, 10), (3, 10), (150, 500), (499, 500)])
def test_clopper_pearson_satisfies_binomial_tail_definition(k, n):
    # Defining equations: P(X >= k | lower) = a/2 and P(X <= k | upper) = a/2.
    out = prob_loss_interval(_terminal_with_k_losses(k, n), S0, LEVEL, "clopper_pearson")
    a = 1 - LEVEL
    assert stats.binom.sf(k - 1, n, out["lower"]) == pytest.approx(a / 2, rel=1e-7)
    assert stats.binom.cdf(k, n, out["upper"]) == pytest.approx(a / 2, rel=1e-7)


def test_clopper_pearson_contains_wilson_at_moderate_k():
    out_w = prob_loss_interval(_terminal_with_k_losses(150, 500), S0, LEVEL, "wilson")
    out_c = prob_loss_interval(_terminal_with_k_losses(150, 500), S0, LEVEL,
                               "clopper_pearson")
    assert out_c["lower"] <= out_w["lower"] and out_w["upper"] <= out_c["upper"]


def test_interval_widens_with_level():
    terminal = _terminal_with_k_losses(30, 100)
    for method in METHODS:
        w90 = prob_loss_interval(terminal, S0, 0.90, method)
        w99 = prob_loss_interval(terminal, S0, 0.99, method)
        assert w99["lower"] < w90["lower"] and w90["upper"] < w99["upper"]


# --------------------------------------------------------------------------
# 2. Consistency with risk_summary and tie handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_estimate_matches_risk_summary_on_gbm_paths(method):
    paths = simulate_gbm(S0, MU, SIGMA, T, steps=12, n_sims=2_001, seed=20261122)
    out = prob_loss_interval(paths[-1], paths[0, 0], LEVEL, method)
    assert out["estimate"] == risk_summary(paths)["prob_loss"]
    assert out["n"] == paths.shape[1]
    assert out["k"] == int(np.count_nonzero(paths[-1] < paths[0, 0]))
    assert out["lower"] <= out["estimate"] <= out["upper"]


@pytest.mark.parametrize("method", METHODS)
def test_ties_at_s0_are_not_losses(method):
    terminal = np.array([S0, S0, S0, S0 - 1e-9, S0 + 5.0, S0])
    out = prob_loss_interval(terminal, S0, LEVEL, method)
    assert out["k"] == 1 and out["n"] == 6
    assert out["estimate"] == pytest.approx(1 / 6)
    # Same convention as risk_summary on a two-row path matrix.
    paths = np.vstack([np.full(terminal.size, S0), terminal])
    assert out["estimate"] == risk_summary(paths)["prob_loss"]

    all_ties = prob_loss_interval(np.full(5, S0), S0, LEVEL, method)
    assert all_ties["k"] == 0 and all_ties["lower"] == 0.0


def test_integer_terminal_and_integer_s0_accepted():
    out = prob_loss_interval(np.array([99, 100, 101, 98]), 100)
    assert out["k"] == 2 and out["n"] == 4 and out["estimate"] == 0.5


# --------------------------------------------------------------------------
# 3. Wilson 95% coverage (N = 500, p = 0.3)
# --------------------------------------------------------------------------

def test_wilson_exact_coverage_n500_p03_within_plan_band():
    """Exact coverage by pmf summation (no Monte Carlo error).

    The empirical band [0.93, 0.97] below is centred on 0.95, but Wilson's
    true coverage is not exactly the nominal level; its exact value for this
    (n, p) is ~0.9547 (computed at commit 764c1e7), well inside the band, so
    the band is not violated by the method's own bias.
    """
    coverage = _exact_coverage(N_PER_REP, P_LOSS, "wilson")
    assert 0.945 <= coverage <= 0.96
    assert 0.93 <= coverage <= 0.97


def test_wilson_empirical_coverage_fast():
    """Fast version: 400 replicates.

    SE of the empirical coverage ~ sqrt(0.955 * 0.045 / 400) ~ 0.0104; with
    k = 4 SE (plan sec. 2.5) around the exact coverage the band is about
    [0.913, 0.996]. Fixed seed, so the result is deterministic.
    """
    n_reps = 400
    exact = _exact_coverage(N_PER_REP, P_LOSS, "wilson")
    se = math.sqrt(exact * (1 - exact) / n_reps)
    coverage = _empirical_coverage(_replicate_terminals(n_reps, seed=1101), "wilson")
    assert abs(coverage - exact) <= 4 * se


@pytest.mark.slow
def test_wilson_empirical_coverage_plan_2000_replicates():
    """Plan E1.1: 2,000 replicates, N = 500, p = 0.3, coverage in [0.93, 0.97].

    The empirical coverage is Binom(2000, c)/2000 with c the true coverage.
    SE ~ sqrt(0.95 * 0.05 / 2000) ~ 0.0049, so +/-0.02 is ~4 SE around 0.95;
    with c ~ 0.9547 (exact, see test above) the band is still >= 3 SE on each
    side. A tighter, c-centred 4 SE check is also applied.
    """
    n_reps = 2_000
    exact = _exact_coverage(N_PER_REP, P_LOSS, "wilson")
    se = math.sqrt(exact * (1 - exact) / n_reps)
    coverage = _empirical_coverage(_replicate_terminals(n_reps, seed=2002), "wilson")
    assert 0.93 <= coverage <= 0.97
    assert abs(coverage - exact) <= 4 * se


def test_replicate_loss_frequency_matches_analytic_p():
    """The replicate sampler really has P(S_T < s0) = 0.3 (4 SE check)."""
    terminals = _replicate_terminals(400, seed=1101)
    freq = float(np.mean(terminals < S0))
    se = math.sqrt(P_LOSS * (1 - P_LOSS) / terminals.size)
    assert abs(freq - P_LOSS) <= 4 * se


# --------------------------------------------------------------------------
# 4. Clopper-Pearson is conservative
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,p", [(20, 0.05), (20, 0.3), (50, 0.5), (100, 0.9),
                                 (500, 0.3)])
def test_clopper_pearson_exact_coverage_at_least_level(n, p):
    # Clopper-Pearson guarantees coverage >= level for every (n, p).
    assert _exact_coverage(n, p, "clopper_pearson") >= LEVEL - 1e-12


@pytest.mark.parametrize("level", [0.80, 0.90, 0.99])
def test_clopper_pearson_exact_coverage_other_levels(level):
    assert _exact_coverage(60, 0.17, "clopper_pearson", level) >= level - 1e-12


def test_clopper_pearson_empirical_coverage_fast():
    """400 replicates; true coverage c >= 0.95, SE <= sqrt(.95*.05/400) ~ 0.011.

    One-sided 4 SE check against the nominal level (never fails because the
    interval is conservative, only if it were anti-conservative by > 4 SE).
    """
    n_reps = 400
    se = math.sqrt(LEVEL * (1 - LEVEL) / n_reps)
    coverage = _empirical_coverage(_replicate_terminals(n_reps, seed=3303),
                                   "clopper_pearson")
    assert coverage >= LEVEL - 4 * se


# --------------------------------------------------------------------------
# 5. Input validation, purity and JSON safety
# --------------------------------------------------------------------------

GOOD = np.array([99.0, 101.0, 100.0])


@pytest.mark.parametrize("terminal", [
    np.array([]),
    [],
    np.ones((2, 3)),
    np.ones((3, 1)),
    np.array(5.0),
    np.array([99.0, np.nan]),
    np.array([99.0, np.inf]),
    np.array([99.0, -np.inf]),
    np.array([99.0 + 1j, 101.0]),
    np.array([True, False]),
    np.array(["99", "101"]),
    np.array([99.0, None], dtype=object),
])
def test_invalid_terminal_raises(terminal):
    with pytest.raises(ValueError):
        prob_loss_interval(terminal, S0)


@pytest.mark.parametrize("s0", [True, False, np.bool_(True), float("nan"),
                                float("inf"), "100", None, 100 + 0j])
def test_invalid_s0_raises(s0):
    with pytest.raises(ValueError):
        prob_loss_interval(GOOD, s0)


@pytest.mark.parametrize("level", [0, 1, 0.0, 1.0, -0.1, 1.5, True, False,
                                   float("nan"), float("inf"), "0.95", None])
def test_invalid_level_raises(level):
    with pytest.raises(ValueError):
        prob_loss_interval(GOOD, S0, level)


@pytest.mark.parametrize("method", ["Wilson", "clopper-pearson", "normal", "",
                                    None, 1])
def test_unknown_method_raises(method):
    with pytest.raises(ValueError):
        prob_loss_interval(GOOD, S0, LEVEL, method)


@pytest.mark.parametrize("method", METHODS)
def test_input_not_modified(method):
    terminal = np.array([99.0, 101.0, 100.0, 50.0])
    original = terminal.copy()
    terminal_list = [99.0, 101.0, 100.0, 50.0]
    prob_loss_interval(terminal, S0, LEVEL, method)
    prob_loss_interval(terminal_list, S0, LEVEL, method)
    np.testing.assert_array_equal(terminal, original)
    assert terminal_list == [99.0, 101.0, 100.0, 50.0]
    assert terminal.dtype == np.float64


@pytest.mark.parametrize("method", METHODS)
def test_output_contract_and_json_safe(method):
    out = prob_loss_interval(np.float32([99.0, 101.0, 98.0]), np.float64(S0),
                             np.float64(0.9), method)
    assert set(out) == {"estimate", "lower", "upper", "method", "level", "n", "k"}
    assert out["method"] == method
    assert type(out["n"]) is int and type(out["k"]) is int
    for key in ("estimate", "lower", "upper", "level"):
        assert type(out[key]) is float
    assert out["level"] == 0.9
    assert json.loads(json.dumps(out, allow_nan=False)) == out


@pytest.mark.parametrize("kwargs", [{"s0": 10**400}, {"s0": S0, "level": 10**400}])
def test_huge_int_inputs_raise_value_error_not_overflow(kwargs):
    # Regression: float(10**400) raises OverflowError; contract is ValueError.
    with pytest.raises(ValueError, match="representable"):
        prob_loss_interval([1.0, 2.0], **kwargs)
