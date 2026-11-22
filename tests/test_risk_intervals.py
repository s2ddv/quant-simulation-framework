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
from scipy import integrate, stats

from risk.intervals import cvar_interval, prob_loss_interval, var_interval
from risk.statistics import risk_summary
from simulation.gbm import simulate_gbm, simulate_terminal_batched
from validation.statistics import terminal_quantile, terminal_theory

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


# ==========================================================================
# E1.2: analytic terminal_quantile and order-statistic var_interval
# ==========================================================================
#
# References written here, independent of the implementation:
# * scipy.stats.lognorm for Q_p(S_T): ln S_T ~ N(ln s0 + (mu - sigma^2/2) T,
#   sigma^2 T), so Q_p = lognorm.ppf(p, s=sigma sqrt(T),
#   scale=s0 exp((mu - sigma^2/2) T));
# * ranks from a cumulative-pmf search (smallest x with F(x) >= q), not
#   binom.ppf; endpoints from a full np.sort, not np.partition;
# * coverage F(k-1) - F(j-1) from the same cumulative pmf.
# Loss quantile at level p: q_p(L) = s0 - Q_{1-p}(S_T), since L = s0 - S_T
# is decreasing in S_T (continuous law).

VAR_P = 0.95
VAR_N = 2_000


def _binom_cdf_table(n: int, p: float) -> np.ndarray:
    return np.cumsum(stats.binom.pmf(np.arange(n + 1), n, p))


def _reference_ranks(n: int, p: float, level: float) -> tuple[int, int, bool, float]:
    """(j, k, saturated, coverage) via a cumulative-pmf search."""
    cdf = _binom_cdf_table(n, p)
    a = 1 - level

    def quantile(q: float) -> int:
        # Smallest x with F(x) >= q; tiny slack for summation rounding.
        return int(np.argmax(cdf >= q - 1e-12))

    j_req, k_req = quantile(a / 2), quantile(1 - a / 2) + 1
    j, k = min(max(j_req, 1), n), min(max(k_req, 1), n)

    def big_f(x: int) -> float:
        return 0.0 if x < 0 else float(cdf[min(x, n)])

    return j, k, (j, k) != (j_req, k_req), big_f(k - 1) - big_f(j - 1)


def _theoretical_loss_quantile(p: float) -> float:
    return S0 - terminal_quantile(S0, MU, SIGMA, T, 1 - p)


# --------------------------------------------------------------------------
# terminal_quantile
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params", [(100.0, 0.05, 0.2, 1.0), (37.5, -0.3, 0.6, 2.5),
                                    (1.0, 0.0, 1.5, 0.01)])
def test_terminal_quantile_matches_terminal_theory(params):
    theory = terminal_theory(*params)["quantiles"]
    for p in (0.05, 0.5, 0.95):
        assert terminal_quantile(*params, p) == pytest.approx(theory[str(p)],
                                                               rel=1e-14)


@pytest.mark.parametrize("params", [(100.0, 0.05, 0.2, 1.0), (37.5, -0.3, 0.6, 2.5),
                                    (1.0, 0.0, 1.5, 0.01), (S0, MU, SIGMA, T)])
@pytest.mark.parametrize("p", [1e-6, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999])
def test_terminal_quantile_matches_scipy_lognorm(params, p):
    s0, mu, sigma, horizon = params
    ref = stats.lognorm.ppf(p, s=sigma * math.sqrt(horizon),
                            scale=s0 * math.exp((mu - sigma**2 / 2) * horizon))
    assert terminal_quantile(s0, mu, sigma, horizon, p) == pytest.approx(ref, rel=1e-12)


def test_terminal_quantile_degenerate_cases():
    # sigma = 0: S_T = s0 exp(mu T) deterministically, for every p.
    for p in (0.01, 0.5, 0.99):
        assert terminal_quantile(100.0, 0.05, 0.0, 2.0, p) == pytest.approx(
            100.0 * math.exp(0.1), rel=1e-15)
        # T = 0: S_T = s0.
        assert terminal_quantile(100.0, 0.05, 0.3, 0.0, p) == 100.0
    # Tiny T: quantile collapses to s0 (to float precision), no nan/underflow.
    for p in (0.01, 0.5, 0.99):
        q = terminal_quantile(100.0, 0.05, 0.2, 1e-300, p)
        assert math.isfinite(q) and q == pytest.approx(100.0, rel=1e-14)


@pytest.mark.parametrize("p", [0, 1, 0.0, 1.0, -0.1, 1.1, True, False, np.bool_(True),
                               float("nan"), float("inf"), "0.5", None])
def test_terminal_quantile_invalid_p(p):
    with pytest.raises(ValueError):
        terminal_quantile(100.0, 0.05, 0.2, 1.0, p)


@pytest.mark.parametrize("params", [(100.0, 1000.0, 0.2, 1.0),   # overflow
                                    (100.0, -1000.0, 0.2, 1.0),  # underflow to 0
                                    (1e300, 10.0, 0.0, 100.0)])  # degenerate overflow
def test_terminal_quantile_overflow_underflow_raise(params):
    with pytest.raises(ValueError):
        terminal_quantile(*params, 0.5)


def test_terminal_quantile_monotone_in_p():
    ps = np.linspace(0.001, 0.999, 201)
    qs = np.array([terminal_quantile(S0, MU, SIGMA, T, float(p)) for p in ps])
    assert np.all(np.diff(qs) > 0)


def test_loss_quantile_setup_matches_analytic_prob_loss():
    # Sanity of the target: P(L <= q_p) = p  <=>  P(S_T >= Q_{1-p}) = p.
    q = _theoretical_loss_quantile(1 - P_LOSS)
    assert q == pytest.approx(0.0, abs=1e-10)  # P(S_T < s0) = 0.3 by construction


# --------------------------------------------------------------------------
# var_interval: consistency, ranks, coverage formula
# --------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.99])
def test_var_estimate_equals_risk_summary(confidence):
    paths = simulate_gbm(S0, MU, SIGMA, T, steps=4, n_sims=1_237, seed=4242)
    out = var_interval(paths[-1], paths[0, 0], confidence)
    assert out["estimate"] == risk_summary(paths, confidence)["var"]


@pytest.mark.parametrize("n,p,level", [
    (2_000, 0.95, 0.95), (2_000, 0.99, 0.90), (500, 0.5, 0.99), (137, 0.05, 0.95),
    (50, 0.9, 0.8), (10, 0.95, 0.95), (30, 0.05, 0.95), (3, 0.5, 0.5),
])
def test_var_ranks_and_endpoints_match_independent_reference(n, p, level):
    rng = np.random.default_rng(777 + n)
    terminal = S0 * np.exp(rng.normal(0.0, 0.2, n))
    out = var_interval(terminal, S0, p, level)
    j, k, saturated, coverage = _reference_ranks(n, p, level)
    assert (out["lower_rank"], out["upper_rank"]) == (j, k)
    assert out["saturated"] is saturated
    assert out["coverage"] == pytest.approx(coverage, abs=1e-10)
    sorted_losses = np.sort(S0 - terminal)
    assert out["lower"] == sorted_losses[j - 1]
    assert out["upper"] == sorted_losses[k - 1]
    assert 1 <= out["lower_rank"] <= out["upper_rank"] <= n
    if not saturated:
        assert out["coverage"] >= level


def test_var_unsaturated_coverage_at_least_level_grid():
    for n in (20, 57, 200, 2_000):
        for p in (0.5, 0.9, 0.95, 0.99):
            for level in (0.8, 0.9, 0.95, 0.99):
                out = var_interval(np.linspace(1.0, 2.0, n), S0, p, level)
                if not out["saturated"]:
                    assert out["coverage"] >= level, (n, p, level)


# --------------------------------------------------------------------------
# var_interval: empirical coverage (plan E1.2: N = 2000, p = 0.95)
# --------------------------------------------------------------------------

def _var_replicates(n_reps: int, n: int, seed: int) -> np.ndarray:
    """Independent replicates: disjoint (not nested) blocks of one i.i.d. stream."""
    terminal = simulate_terminal_batched(S0, MU, SIGMA, T, n_reps * n, seed)
    return np.asarray(terminal).reshape(n_reps, n)


def _var_empirical_coverage(terminals: np.ndarray, p: float, level: float = LEVEL
                            ) -> tuple[float, float]:
    target = _theoretical_loss_quantile(p)
    hits, coverage = 0, float("nan")
    for row in terminals:
        out = var_interval(row, S0, p, level)
        hits += out["lower"] <= target <= out["upper"]
        coverage = out["coverage"]  # depends only on (n, p, level)
    return hits / terminals.shape[0], coverage


def test_var_empirical_coverage_fast():
    """150 replicates of N = 2000, p = 0.95.

    Exact coverage c of the construction is >= 0.95 (reported, checked
    above). Empirical coverage ~ Binom(150, c)/150, SE <= sqrt(.95*.05/150)
    ~ 0.018; one-sided check at c - 4 SE (~0.88), per plan sec. 2.5.
    """
    n_reps = 150
    emp, c = _var_empirical_coverage(_var_replicates(n_reps, VAR_N, seed=5151), VAR_P)
    assert c >= LEVEL
    assert emp >= c - 4 * math.sqrt(c * (1 - c) / n_reps)


@pytest.mark.slow
def test_var_empirical_coverage_plan_1000_replicates():
    """Plan E1.2: 1,000 independent replicates, N = 2000, p = 0.95, >= 0.93.

    Exact coverage c >= 0.95; SE ~ sqrt(.95*.05/1000) ~ 0.0069, so 0.93 is
    ~3 SE below the nominal level (and further below c). A c-centred 4 SE
    two-sided check is applied as well.
    """
    n_reps = 1_000
    emp, c = _var_empirical_coverage(_var_replicates(n_reps, VAR_N, seed=6262), VAR_P)
    se = math.sqrt(c * (1 - c) / n_reps)
    assert emp >= 0.93
    assert abs(emp - c) <= 4 * se


# --------------------------------------------------------------------------
# var_interval: small N and saturation
# --------------------------------------------------------------------------

def test_var_upper_saturation_n10_p095():
    out = var_interval(S0 - np.arange(1.0, 11.0), S0, 0.95, 0.95)  # losses 1..10
    # Unclipped upper rank is binom.ppf(0.975, 10, 0.95) + 1 = 11 > N.
    assert out["saturated"] is True
    assert out["upper_rank"] == 10 and out["upper"] == 10.0
    assert out["lower_rank"] == 8 and out["lower"] == 8.0
    # Exact: F(9) - F(7) = 1 - 0.95^10 - P(B <= 7).
    exact = 1 - 0.95**10 - stats.binom.cdf(7, 10, 0.95)
    assert out["coverage"] == pytest.approx(exact, rel=1e-12)
    assert out["coverage"] < out["level"]


def test_var_lower_saturation_n30_confidence005():
    out = var_interval(S0 - np.arange(1.0, 31.0), S0, 0.05, 0.95)
    # Unclipped lower rank is binom.ppf(0.025, 30, 0.05) = 0 < 1.
    assert out["saturated"] is True
    assert (out["lower_rank"], out["upper_rank"]) == (1, 5)
    exact = stats.binom.cdf(4, 30, 0.05) - 0.95**30  # F(4) - F(0)
    assert out["coverage"] == pytest.approx(exact, rel=1e-12)
    assert out["coverage"] == pytest.approx(0.770, abs=5e-4)
    assert out["coverage"] < out["level"]


@pytest.mark.parametrize("n,p", [(10, 0.95), (30, 0.05)])
def test_var_saturated_empirical_coverage_matches_reported(n, p):
    """2,000 replicates; |empirical - reported| <= 4 SE (SE <= 0.0112)."""
    n_reps = 2_000
    emp, c = _var_empirical_coverage(_var_replicates(n_reps, n, seed=7000 + n), p)
    assert abs(emp - c) <= 4 * math.sqrt(c * (1 - c) / n_reps)


def test_var_n_equals_one():
    out = var_interval([90.0], S0, 0.95, 0.95)
    assert out["n"] == 1 and out["saturated"] is True
    assert out["lower"] == out["upper"] == out["estimate"] == 10.0
    assert (out["lower_rank"], out["upper_rank"]) == (1, 1)
    assert out["coverage"] == 0.0


# --------------------------------------------------------------------------
# var_interval: degenerate laws, atoms, documented corner case
# --------------------------------------------------------------------------

def test_var_sigma_zero_degenerate_interval():
    paths = simulate_gbm(S0, 0.05, 0.0, 1.0, steps=3, n_sims=400, seed=1)
    out = var_interval(paths[-1], S0, 0.95, 0.95)
    v = S0 - paths[-1, 0]
    assert np.all(paths[-1] == paths[-1, 0])
    assert out["lower"] == out["upper"] == out["estimate"] == v


def test_var_ties_endpoints_are_sample_values():
    terminal = np.repeat([95.0, 100.0, 103.0], [30, 50, 20])  # losses 5, 0, -3
    out = var_interval(terminal, S0, 0.5, 0.95)
    sorted_losses = np.sort(S0 - terminal)
    assert out["lower"] == sorted_losses[out["lower_rank"] - 1]
    assert out["upper"] == sorted_losses[out["upper_rank"] - 1]
    assert out["lower"] == out["upper"] == 0.0


def test_var_atoms_coverage_is_conservative():
    """Discrete loss L uniform on {0,...,9}, p = 0.55: unique quantile q = 5
    (P(L < 5) = 0.5 <= 0.55 <= 0.6 = P(L <= 5)). Docstring claims coverage
    >= reported value; one-sided 4 SE check over 1,000 replicates."""
    n, n_reps, p, level = 50, 1_000, 0.55, 0.90
    rng = np.random.default_rng(9090)
    losses = rng.integers(0, 10, size=(n_reps, n)).astype(float)
    hits, c = 0, float("nan")
    for row in losses:
        out = var_interval(S0 - row, S0, p, level)
        hits += out["lower"] <= 5.0 <= out["upper"]
        c = out["coverage"]
    emp = hits / n_reps
    assert emp >= c - 4 * math.sqrt(c * (1 - c) / n_reps)


def test_var_estimate_may_lie_outside_interval_without_violating_contract():
    """Documented, not required: for small N / small level the linear
    percentile can fall outside [lower, upper] (here N = 4, level = 0.05).
    The ranks and coverage remain those of the order-statistic construction."""
    out = var_interval(S0 - np.arange(4.0), S0, 0.35, 0.05)  # losses 0..3
    assert out["saturated"] is False
    assert not out["lower"] <= out["estimate"] <= out["upper"]
    j, k, _, coverage = _reference_ranks(4, 0.35, 0.05)
    assert (out["lower_rank"], out["upper_rank"]) == (j, k)
    assert out["coverage"] == pytest.approx(coverage, abs=1e-12)
    assert out["coverage"] >= out["level"]


# --------------------------------------------------------------------------
# var_interval: validation, purity, JSON
# --------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", [
    np.array([]), [], np.ones((2, 3)), np.array(5.0), np.array([99.0, np.nan]),
    np.array([99.0, np.inf]), np.array([99.0 + 1j, 101.0]), np.array([True, False]),
    np.array(["99", "101"]),
])
def test_var_invalid_terminal_raises(terminal):
    with pytest.raises(ValueError):
        var_interval(terminal, S0)


@pytest.mark.parametrize("kwargs", [
    {"s0": True}, {"s0": float("nan")}, {"s0": "100"}, {"s0": 10**400},
    *({"s0": S0, "confidence": v} for v in (0, 1, -0.5, 1.5, True, np.bool_(False),
                                            float("nan"), "0.95", None)),
    *({"s0": S0, "level": v} for v in (0, 1, -0.5, 1.5, True, np.bool_(False),
                                       float("nan"), "0.95", None)),
])
def test_var_invalid_scalars_raise(kwargs):
    with pytest.raises(ValueError):
        var_interval(GOOD, **kwargs)


def test_var_nonfinite_losses_raise():
    with pytest.raises(ValueError):
        var_interval(np.array([-1.7e308, 1.0]), 1.7e308)


def test_var_input_not_modified():
    terminal = np.array([101.0, 95.0, 100.0, 80.0, 120.0])
    original = terminal.copy()
    as_list = terminal.tolist()
    var_interval(terminal, S0, 0.6, 0.9)
    var_interval(as_list, S0, 0.6, 0.9)
    np.testing.assert_array_equal(terminal, original)
    assert as_list == original.tolist()


def test_var_output_contract_and_json_safe():
    out = var_interval(np.float32([99.0, 101.0, 98.0, 97.0]), np.float64(S0),
                       np.float64(0.75), np.float64(0.9))
    assert set(out) == {"estimate", "lower", "upper", "lower_rank", "upper_rank",
                        "saturated", "coverage", "method", "level", "confidence", "n"}
    assert out["method"] == "order_statistics"
    for key in ("lower_rank", "upper_rank", "n"):
        assert type(out[key]) is int
    assert type(out["saturated"]) is bool
    for key in ("estimate", "lower", "upper", "coverage", "level", "confidence"):
        assert type(out[key]) is float
    assert 0.0 <= out["coverage"] <= 1.0
    assert out["level"] == 0.9 and out["confidence"] == 0.75
    assert json.loads(json.dumps(out, allow_nan=False)) == out


_TQ_ARGS = {"s0": 1.0, "mu": 0.05, "sigma": 0.2, "horizon": 1.0}


@pytest.mark.parametrize("name,bad", [
    *[(n, b) for n in _TQ_ARGS for b in (True, np.bool_(True))],
    ("s0", "1"),
])
def test_terminal_quantile_rejects_bool_and_non_real_params(name, bad):
    # Regression: bool was silently cast (s0=True -> 1.03) and str raised
    # TypeError; the contract is ValueError (plan section 2.4).
    kwargs = {**_TQ_ARGS, name: bad}
    with pytest.raises(ValueError, match=name):
        terminal_quantile(kwargs["s0"], kwargs["mu"], kwargs["sigma"],
                          kwargs["horizon"], 0.5)


# ==========================================================================
# E1.3: CVaR interval (asymptotic and percentile bootstrap)
# ==========================================================================
#
# Analytic GBM CVaR at level p (losses L = s0 - S_T, x = Q_{1-p}(S_T)):
#   CVaR_p = s0 - E[S_T | S_T <= x]
#          = s0 - s0 e^{mu T} Phi(d - sigma sqrt(T)) / Phi(d),
#   d = (ln(x/s0) - (mu - sigma^2/2) T) / (sigma sqrt(T)) = Phi^{-1}(1 - p).
# Cross-checked below against scipy.integrate.quad over the lognormal pdf.

CVAR_METHODS = ("asymptotic", "bootstrap")
CVAR_P = 0.95
CVAR_N = 2_000  # expected tail size N (1 - p) = 100


def _lognormal_params(s0, mu, sigma, horizon):
    return sigma * math.sqrt(horizon), s0 * math.exp((mu - sigma**2 / 2) * horizon)


def _cvar_closed_form(s0, mu, sigma, horizon, p):
    shape, scale = _lognormal_params(s0, mu, sigma, horizon)
    x = stats.lognorm.ppf(1 - p, s=shape, scale=scale)
    d = (math.log(x / s0) - (mu - sigma**2 / 2) * horizon) / shape
    return s0 - s0 * math.exp(mu * horizon) * stats.norm.cdf(d - shape) / stats.norm.cdf(d)


def _cvar_quad(s0, mu, sigma, horizon, p):
    shape, scale = _lognormal_params(s0, mu, sigma, horizon)
    x = stats.lognorm.ppf(1 - p, s=shape, scale=scale)
    tail_mass, _ = integrate.quad(lambda s: stats.lognorm.pdf(s, s=shape, scale=scale),
                                  0.0, x, epsabs=0, epsrel=1e-12, limit=200)
    partial_mean, _ = integrate.quad(
        lambda s: s * stats.lognorm.pdf(s, s=shape, scale=scale),
        0.0, x, epsabs=0, epsrel=1e-12, limit=200)
    return s0 - partial_mean / tail_mass


CVAR_TRUE = _cvar_closed_form(S0, MU, SIGMA, T, CVAR_P)


def _cvar_kwargs(method, **extra):
    kwargs = {"method": method}
    if method == "bootstrap":
        kwargs.update({"n_bootstrap": 200, "seed": 12345})
    kwargs.update(extra)
    return kwargs


# --------------------------------------------------------------------------
# Analytic reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params", [(S0, MU, SIGMA, T), (100.0, 0.05, 0.2, 1.0),
                                    (50.0, -0.1, 0.5, 0.25)])
@pytest.mark.parametrize("p", [0.9, 0.95, 0.99])
def test_cvar_closed_form_matches_quadrature(params, p):
    assert _cvar_closed_form(*params, p) == pytest.approx(_cvar_quad(*params, p),
                                                          rel=1e-9)


def test_cvar_estimate_converges_to_analytic_value():
    """One large sample (N = 200,000, tail 10,000 points): |est - CVaR| <= 4 SE
    with SE the asymptotic one (validated by simulation below)."""
    terminal = simulate_terminal_batched(S0, MU, SIGMA, T, 200_000, seed=8080)
    out = cvar_interval(terminal, S0, CVAR_P)
    assert abs(out["estimate"] - CVAR_TRUE) <= 4 * out["standard_error"]
    assert out["var"] == pytest.approx(
        S0 - terminal_quantile(S0, MU, SIGMA, T, 1 - CVAR_P), abs=0.5)


# --------------------------------------------------------------------------
# Consistency with risk_summary (ties at VaR included in the tail)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", CVAR_METHODS)
@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.99])
def test_cvar_estimate_and_var_equal_risk_summary(method, confidence):
    paths = simulate_gbm(S0, MU, SIGMA, T, steps=3, n_sims=1_001, seed=31)
    out = cvar_interval(paths[-1], paths[0, 0], confidence, **_cvar_kwargs(method))
    summary = risk_summary(paths, confidence)
    assert out["estimate"] == summary["cvar"]
    assert out["var"] == summary["var"]
    assert out["n"] == 1_001


@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_ties_at_var_are_in_tail(method):
    # Losses: 20 x 5.0 (at VaR), 70 x 0.0, 10 x -3.0; VaR(0.9) = 5.0 exactly.
    terminal = np.repeat([95.0, 100.0, 103.0], [20, 70, 10])
    paths = np.vstack([np.full(terminal.size, S0), terminal])
    out = cvar_interval(terminal, S0, 0.9, **_cvar_kwargs(method))
    summary = risk_summary(paths, 0.9)
    assert out["var"] == summary["var"] == 5.0
    assert out["estimate"] == summary["cvar"] == 5.0
    assert out["n_tail"] == 20  # every tie at VaR, not just N (1 - p) = 10


# --------------------------------------------------------------------------
# Asymptotic SE vs simulation; empirical coverage
# --------------------------------------------------------------------------

def test_cvar_asymptotic_se_matches_replicate_sd():
    """R = 500 independent replicates (disjoint blocks), N = 2000, p = 0.95.

    The empirical SD of the estimator over R replicates has relative standard
    error ~ 1/sqrt(2 (R - 1)) ~ 0.032 (near-normal estimator); 4 x that is
    0.127. The mean asymptotic SE must match within that relative band. The
    mean of the estimates must be within 4 SD/sqrt(R) + an O(1/N) bias
    allowance of the analytic CVaR (bias allowance: 0.1 price units, ~0.13 SD).
    """
    n_reps = 500
    outs = [cvar_interval(row, S0, CVAR_P)
            for row in _var_replicates(n_reps, CVAR_N, seed=4401)]
    est = np.array([o["estimate"] for o in outs])
    se = np.array([o["standard_error"] for o in outs])
    sd = float(est.std(ddof=1))
    assert abs(se.mean() / sd - 1) <= 4 / math.sqrt(2 * (n_reps - 1))
    assert abs(est.mean() - CVAR_TRUE) <= 4 * sd / math.sqrt(n_reps) + 0.1


def _cvar_coverage(n_reps, seed, method, n_bootstrap=200):
    hits = 0
    for i, row in enumerate(_var_replicates(n_reps, CVAR_N, seed)):
        kwargs = {"n_bootstrap": n_bootstrap, "seed": i} if method == "bootstrap" else {}
        out = cvar_interval(row, S0, CVAR_P, LEVEL, method, **kwargs)
        hits += out["lower"] <= CVAR_TRUE <= out["upper"]
    return hits / n_reps


# Threshold rationale (plan E1.3): nominal 0.95, but both intervals are
# approximate with only N (1 - p) = 100 tail points: the asymptotic normal
# ignores the estimator's skewness and the percentile bootstrap of a tail mean
# under-covers in tails, so true coverage of ~0.93 is plausible. Plan floor:
# 0.90 for 500 replicates, i.e. ~3 binomial SE (sqrt(.93*.07/500) ~ 0.011)
# below 0.93.

@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_empirical_coverage_fast(method):
    """80 replicates, B = 200: SE ~ sqrt(.93*.07/80) ~ 0.029, floor
    0.93 - 3 SE ~ 0.84 (same 3 SE margin as the plan's 0.90 at R = 500)."""
    assert _cvar_coverage(80, seed=4502, method=method) >= 0.84


@pytest.mark.slow
@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_empirical_coverage_plan_500_replicates(method):
    """Plan E1.3: 500 replicates, N = 2000, p = 0.95, coverage >= 0.90.

    Bootstrap uses B = 400 instead of the default 1000 to keep cost ~12 s
    (one call with B = 1000 at N = 2000 takes ~0.09 s, i.e. ~45 s in total).
    B only adds Monte Carlo noise to the 2.5%/97.5% percentile endpoints
    (about 10 replicates beyond each endpoint at B = 400), which perturbs
    each interval slightly but does not bias the coverage materially.
    """
    assert _cvar_coverage(500, seed=4603, method=method, n_bootstrap=400) >= 0.90


# --------------------------------------------------------------------------
# Determinism and RNG
# --------------------------------------------------------------------------

def _small_sample(n=50, seed=99):
    return S0 * np.exp(np.random.default_rng(seed).normal(0.0, 0.2, n))


def test_cvar_bootstrap_same_seed_identical_different_seed_differs():
    x = _small_sample()
    a = cvar_interval(x, S0, 0.9, method="bootstrap", n_bootstrap=50, seed=7)
    b = cvar_interval(x, S0, 0.9, method="bootstrap", n_bootstrap=50, seed=7)
    c = cvar_interval(x, S0, 0.9, method="bootstrap", n_bootstrap=50, seed=8)
    assert a == b
    assert (a["lower"], a["upper"], a["bootstrap_standard_error"]) != (
        c["lower"], c["upper"], c["bootstrap_standard_error"])
    assert a["estimate"] == c["estimate"]  # point estimate is RNG-free


def test_cvar_bootstrap_independent_of_batch_size():
    x = _small_sample()
    outs = [cvar_interval(x, S0, 0.9, method="bootstrap", n_bootstrap=23, seed=3,
                          batch_size=bs) for bs in (1, 7, None, 23, 1000)]
    assert all(o == outs[0] for o in outs)


def test_cvar_bootstrap_seed_none_runs_and_reports_none():
    out = cvar_interval(_small_sample(), S0, 0.9, method="bootstrap", n_bootstrap=20)
    assert out["seed"] is None
    assert out["lower"] <= out["upper"]
    assert isinstance(out["rng"], str) and out["rng"]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_input_not_modified(method):
    x = _small_sample()
    original = x.copy()
    as_list = x.tolist()
    cvar_interval(x, S0, 0.9, **_cvar_kwargs(method))
    cvar_interval(as_list, S0, 0.9, **_cvar_kwargs(method))
    np.testing.assert_array_equal(x, original)
    assert as_list == original.tolist()


# --------------------------------------------------------------------------
# Degenerate cases
# --------------------------------------------------------------------------

def test_cvar_n_equals_one():
    asym = cvar_interval([90.0], S0, 0.95)
    assert asym["estimate"] == asym["var"] == 10.0 and asym["n_tail"] == 1
    assert asym["lower"] is None and asym["upper"] is None
    assert asym["standard_error"] is None and isinstance(asym["note"], str)
    json.dumps(asym, allow_nan=False)
    boot = cvar_interval([90.0], S0, 0.95, method="bootstrap", n_bootstrap=10, seed=1)
    assert boot["lower"] == boot["upper"] == boot["estimate"] == 10.0
    assert boot["bootstrap_standard_error"] == 0.0


def test_cvar_asymptotic_single_tail_point_gives_none_and_note():
    # N = 5, p = 0.95: VaR interpolates between the two largest losses, so
    # only the maximum is >= VaR.
    out = cvar_interval(S0 - np.array([1.0, 2.0, 3.0, 4.0, 10.0]), S0, 0.95)
    assert out["n_tail"] == 1 and out["estimate"] == 10.0
    assert out["lower"] is None and out["upper"] is None
    assert out["standard_error"] is None and "note" in out
    assert json.loads(json.dumps(out, allow_nan=False)) == out


@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_sigma_zero_degenerate(method):
    paths = simulate_gbm(S0, 0.05, 0.0, 1.0, steps=2, n_sims=300, seed=5)
    v = S0 - paths[-1, 0]
    out = cvar_interval(paths[-1], S0, 0.95, **_cvar_kwargs(method))
    assert out["estimate"] == out["var"] == out["lower"] == out["upper"] == v
    se_key = "standard_error" if method == "asymptotic" else "bootstrap_standard_error"
    assert out[se_key] == 0.0


@pytest.mark.parametrize("method", CVAR_METHODS)
@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_cvar_extreme_scales_finite(method, scale):
    x = _small_sample(200, seed=11)
    out = cvar_interval(x * (scale / S0), scale, 0.9, **_cvar_kwargs(method))
    values = [out["estimate"], out["var"], out["lower"], out["upper"]]
    assert np.isfinite(values).all()
    assert out["lower"] <= out["upper"]
    se_key = "standard_error" if method == "asymptotic" else "bootstrap_standard_error"
    assert np.isfinite(out[se_key]) and out[se_key] > 0
    # Scale-equivariance against the same sample at unit scale.
    ref = cvar_interval(x, S0, 0.9, **_cvar_kwargs(method))
    assert out["estimate"] == pytest.approx(ref["estimate"] * scale / S0, rel=1e-9)
    assert out[se_key] == pytest.approx(ref[se_key] * scale / S0, rel=1e-9)


# --------------------------------------------------------------------------
# Validation and output contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"method": "Asymptotic"}, {"method": "normal"}, {"method": None},
    *({"method": "bootstrap", "n_bootstrap": v} for v in (1, 0, True, 2.0, "10")),
    *({"method": "bootstrap", "seed": v} for v in (-1, True, 1.5, "1")),
    *({"method": "bootstrap", "batch_size": v} for v in (0, -3, False, 2.5)),
    *({"confidence": v} for v in (0, 1, 1.5, True, float("nan"))),
    *({"level": v} for v in (0, 1, -0.2, np.bool_(True), float("nan"))),
])
def test_cvar_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        cvar_interval(_small_sample(20), S0, **kwargs)


@pytest.mark.parametrize("terminal,s0", [
    (np.array([]), S0), (np.ones((2, 2)), S0), (np.array([1.0, np.nan]), S0),
    (np.array([True, False]), S0), (GOOD, True), (GOOD, float("inf")),
])
def test_cvar_invalid_terminal_or_s0_raise(terminal, s0):
    with pytest.raises(ValueError):
        cvar_interval(terminal, s0)


@pytest.mark.parametrize("method", CVAR_METHODS)
def test_cvar_output_contract_and_json_safe(method):
    out = cvar_interval(np.float32(_small_sample(40)), np.float64(S0), np.float64(0.9),
                        np.float64(0.8), **_cvar_kwargs(method))
    common = {"estimate", "lower", "upper", "method", "level", "confidence", "n",
              "n_tail", "var"}
    extra = ({"standard_error"} if method == "asymptotic"
             else {"bootstrap_standard_error", "n_bootstrap", "seed", "rng"})
    assert set(out) == common | extra
    assert out["method"] == method
    for key in ("estimate", "lower", "upper", "level", "confidence", "var",
                *(k for k in extra if k.endswith("standard_error"))):
        assert type(out[key]) is float, key
    for key in ("n", "n_tail", *(("n_bootstrap", "seed") if method == "bootstrap" else ())):
        assert type(out[key]) is int, key
    assert out["level"] == 0.8 and out["confidence"] == 0.9
    assert out["lower"] <= out["upper"]
    assert json.loads(json.dumps(out, allow_nan=False)) == out
