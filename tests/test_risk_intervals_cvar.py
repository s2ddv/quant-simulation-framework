"""Tests for ``risk.intervals.cvar_interval`` (E1.3, offline, seeded).

What is tested is the Monte Carlo *estimator error* interval for the CVaR
(expected shortfall) of the loss ``L = s0 - S_T`` at level ``p``, with the
CVaR defined exactly as in ``risk_summary`` (mean of losses >= VaR, ties
included), for the asymptotic-normal and the percentile-bootstrap methods.

Independent references (none reuses the code under test):

* GBM terminal law: ``ln S_T ~ N(ln s0 + (mu - sigma^2/2) T, sigma^2 T)``,
  built here with ``scipy.stats.lognorm``/``norm``.
* Closed-form tail moments of the lognormal below ``x = Q_{1-p}(S_T)``, with
  ``d = (ln(x/s0) - (mu - sigma^2/2) T) / (sigma sqrt(T))``:
  ``E[S_T | S_T <= x] = s0 e^{mu T} Phi(d - sigma sqrt(T)) / Phi(d)`` and
  ``E[S_T^2 | S_T <= x] = s0^2 e^{(2 mu + sigma^2) T} Phi(d - 2 sigma sqrt(T)) / Phi(d)``;
  hence ``CVaR = s0 - E[S_T | S_T <= x]`` and
  ``Var(L | L >= VaR) = Var(S_T | S_T <= x)``. The first moment is checked
  again by ``scipy.integrate.quad`` over the lognormal density.
* The asymptotic variance written from the plan (Brazauskas et al. 2008):
  ``[Var(L | L >= VaR) + p (CVaR - VaR)^2] / (N (1 - p))``.
* ``risk_summary`` for the point estimate.
* Replicated independent samples (rows of one seeded normal matrix, each row
  an independent GBM sample) for the sampling SD and empirical coverage.

Units: prices in currency, T in years, mu in 1/year, sigma in 1/sqrt(year).
"""

from __future__ import annotations

import json
import math
import tracemalloc
import zlib

import numpy as np
import pytest
from scipy import integrate, stats

from risk.intervals import cvar_interval
from risk.statistics import risk_summary

S0, MU, SIGMA, T = 100.0, 0.05, 0.20, 1.0
P = 0.95
LEVEL = 0.95
LOG_MEAN = math.log(S0) + (MU - SIGMA**2 / 2) * T
LOG_SD = SIGMA * math.sqrt(T)


def _gbm_cvar_reference(s0=S0, mu=MU, sigma=SIGMA, t=T, p=P):
    """Analytic VaR, CVaR and Var(L | L >= VaR) of GBM losses (closed form)."""
    log_sd = sigma * math.sqrt(t)
    terminal = stats.lognorm(s=log_sd,
                             scale=math.exp(math.log(s0) + (mu - sigma**2 / 2) * t))
    x = float(terminal.ppf(1 - p))  # Q_{1-p}(S_T); VaR = s0 - x
    d = (math.log(x / s0) - (mu - sigma**2 / 2) * t) / log_sd
    phi_d = stats.norm.cdf(d)
    first = s0 * math.exp(mu * t) * stats.norm.cdf(d - log_sd) / phi_d
    second = (s0**2 * math.exp((2 * mu + sigma**2) * t)
              * stats.norm.cdf(d - 2 * log_sd) / phi_d)
    return {"x": x, "var": s0 - x, "cvar": s0 - first,
            "tail_variance": second - first**2, "terminal": terminal}


REF = _gbm_cvar_reference()


def _asymptotic_se_true(n, ref=REF, p=P):
    """Plan's asymptotic SE evaluated at the exact GBM quantities."""
    return math.sqrt((ref["tail_variance"] + p * (ref["cvar"] - ref["var"])**2)
                     / (n * (1 - p)))


def _gbm_terminal_matrix(replicates, n, seed):
    """``replicates`` independent rows of n exact GBM terminal prices."""
    rng = np.random.default_rng(seed)
    return np.exp(LOG_MEAN + LOG_SD * rng.standard_normal((replicates, n)))


def _summary_cvar(terminal, s0, confidence):
    """CVaR of ``risk_summary`` for a terminal sample (2-row path matrix)."""
    terminal = np.asarray(terminal, dtype=float)
    paths = np.vstack([np.full(terminal.size, float(s0)), terminal])
    return risk_summary(paths, confidence)["cvar"]


def _coverage(terminal_rows, method, **kwargs):
    hits = 0
    for index, row in enumerate(terminal_rows):
        extra = dict(kwargs)
        if method == "bootstrap":
            extra["seed"] = index
        result = cvar_interval(row, S0, P, LEVEL, method=method, **extra)
        hits += result["lower"] <= REF["cvar"] <= result["upper"]
    return hits / len(terminal_rows)


# --------------------------------------------------------------------------
# Analytic reference
# --------------------------------------------------------------------------

def test_closed_form_tail_mean_matches_numerical_integration():
    terminal, x = REF["terminal"], REF["x"]
    assert terminal.cdf(x) == pytest.approx(1 - P, rel=1e-12)
    mass = integrate.quad(terminal.pdf, 0, x, epsabs=0, epsrel=1e-12)[0]
    first = integrate.quad(lambda y: y * terminal.pdf(y), 0, x,
                           epsabs=0, epsrel=1e-12)[0] / mass
    second = integrate.quad(lambda y: y * y * terminal.pdf(y), 0, x,
                            epsabs=0, epsrel=1e-12)[0] / mass
    assert S0 - first == pytest.approx(REF["cvar"], rel=1e-9)
    assert second - first**2 == pytest.approx(REF["tail_variance"], rel=1e-7)
    # Sanity on the reference: CVaR exceeds VaR, both in price units.
    assert REF["cvar"] > REF["var"] > 0


def test_closed_form_reference_degenerates_correctly_for_tiny_sigma():
    ref = _gbm_cvar_reference(sigma=1e-6)
    deterministic_loss = S0 - S0 * math.exp(MU * T)
    assert ref["cvar"] == pytest.approx(deterministic_loss, abs=1e-3)
    assert ref["var"] == pytest.approx(deterministic_loss, abs=1e-3)


# --------------------------------------------------------------------------
# Point estimate identical to risk_summary
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.99])
def test_estimate_equals_risk_summary_on_gbm_sample(method, confidence):
    terminal = _gbm_terminal_matrix(1, 997, seed=11)[0]
    result = cvar_interval(terminal, S0, confidence, method=method,
                           n_bootstrap=20, seed=0)
    assert result["estimate"] == _summary_cvar(terminal, S0, confidence)
    assert result["var"] == float(np.percentile(S0 - terminal, confidence * 100))


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_estimate_includes_ties_at_var(method):
    # Losses: 30 x 1.0, 60 x 5.0 (VaR atom), 10 x 9.0. At p = 0.8 the linear
    # percentile is 5.0 and the tail is every loss >= 5.0: 70 points.
    losses = np.repeat([1.0, 5.0, 9.0], [30, 60, 10])
    terminal = S0 - losses
    result = cvar_interval(terminal, S0, 0.8, method=method, n_bootstrap=10, seed=1)
    expected = (60 * 5.0 + 10 * 9.0) / 70
    assert result["var"] == 5.0
    assert result["n_tail"] == 70
    assert result["estimate"] == pytest.approx(expected, rel=1e-15)
    assert result["estimate"] == _summary_cvar(terminal, S0, 0.8)


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_estimate_with_many_ties_on_integer_grid(method):
    rng = np.random.default_rng(5)
    terminal = rng.integers(90, 111, size=501).astype(float)  # heavy ties
    for confidence in (0.6, 0.9, 0.95):
        result = cvar_interval(terminal, S0, confidence, method=method,
                               n_bootstrap=10, seed=2)
        losses = S0 - terminal
        assert result["estimate"] == _summary_cvar(terminal, S0, confidence)
        assert result["n_tail"] == int(np.count_nonzero(losses >= result["var"]))


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_negative_losses_are_not_clipped(method):
    terminal = S0 + np.linspace(1.0, 20.0, 200)  # every path gains
    result = cvar_interval(terminal, S0, 0.9, method=method, n_bootstrap=50, seed=3)
    assert result["estimate"] < 0
    assert result["estimate"] == _summary_cvar(terminal, S0, 0.9)
    assert result["lower"] < 0 and result["upper"] < 0


def test_large_sample_estimate_and_se_match_analytic_gbm():
    # N = 200 000 -> n_tail ~ 10 000. True SE ~ 0.074 (price units). The
    # plug-in SE uses a sample variance from ~10^4 tail points, whose relative
    # sampling error is O(10^-2); 5% leaves margin for the O(1/N) VaR/CVaR bias.
    n = 200_000
    terminal = _gbm_terminal_matrix(1, n, seed=2026)[0]
    result = cvar_interval(terminal, S0, P)
    se_true = _asymptotic_se_true(n)
    assert result["standard_error"] == pytest.approx(se_true, rel=0.05)
    assert abs(result["estimate"] - REF["cvar"]) < 4 * se_true


# --------------------------------------------------------------------------
# Asymptotic method: formula and sampling behaviour
# --------------------------------------------------------------------------

def test_asymptotic_se_matches_plan_formula_on_sample():
    terminal = _gbm_terminal_matrix(1, 1500, seed=8)[0]
    result = cvar_interval(terminal, S0, P, LEVEL)
    losses = S0 - terminal
    var = np.percentile(losses, P * 100)
    tail = losses[losses >= var]
    cvar = tail.mean()
    expected_se = math.sqrt((tail.var(ddof=1) + P * (cvar - var)**2)
                            / (losses.size * (1 - P)))
    z = stats.norm.ppf(1 - (1 - LEVEL) / 2)
    assert result["standard_error"] == pytest.approx(expected_se, rel=1e-12)
    assert result["lower"] == pytest.approx(cvar - z * expected_se, rel=1e-12)
    assert result["upper"] == pytest.approx(cvar + z * expected_se, rel=1e-12)
    assert result["n_tail"] == tail.size


def test_asymptotic_se_uses_expected_tail_size_with_ties():
    # Docstring: the denominator is N (1 - p), not the observed n_tail.
    losses = np.repeat([1.0, 5.0, 9.0], [30, 60, 10])
    result = cvar_interval(S0 - losses, S0, 0.8)
    tail = losses[losses >= 5.0]
    expected = math.sqrt((tail.var(ddof=1) + 0.8 * (tail.mean() - 5.0)**2)
                         / (100 * 0.2))
    assert result["standard_error"] == pytest.approx(expected, rel=1e-12)


def test_asymptotic_se_matches_replicate_sd():
    # R = 500 independent samples of N = 2000 (n_tail ~ 100). For a nearly
    # normal estimator the sample SD has relative SE ~ 1/sqrt(2(R-1)) = 3.2%;
    # the mean of the 500 plug-in SEs is far less noisy. A 12% band is ~4
    # SDs of the SD estimate plus room for the O(1/n_tail) finite-tail bias.
    replicates, n = 500, 2000
    rows = _gbm_terminal_matrix(replicates, n, seed=20261122)
    results = [cvar_interval(row, S0, P, LEVEL) for row in rows]
    estimates = np.array([r["estimate"] for r in results])
    mean_se = float(np.mean([r["standard_error"] for r in results]))
    empirical_sd = float(estimates.std(ddof=1))
    assert mean_se / empirical_sd == pytest.approx(1.0, abs=0.12)
    assert mean_se == pytest.approx(_asymptotic_se_true(n), rel=0.05)
    # Unbiasedness to O(1/N): 4 standard errors of the replicate mean.
    assert abs(estimates.mean() - REF["cvar"]) < 4 * empirical_sd / math.sqrt(replicates)


def test_asymptotic_empirical_coverage_plan_500_replicates():
    # Plan: >= 0.90 over 500 replicates at nominal 0.95. Binomial SD at 0.95
    # is 0.0097, so the 0.90 floor leaves ~2 points for the tail bias of a
    # normal approximation with n_tail ~ 100 plus 3 SDs of noise. The 0.99
    # ceiling (4 SDs above nominal) catches a grossly inflated SE.
    # Measured (seed 20261122): 0.950. Cheap (~0.1 s), so not marked slow.
    rows = _gbm_terminal_matrix(500, 2000, seed=20261122)
    coverage = _coverage(rows, "asymptotic")
    assert 0.90 <= coverage <= 0.99


@pytest.mark.slow
def test_asymptotic_empirical_coverage_4000_replicates():
    # Binomial SD at 0.95 with R = 4000 is 0.0034; [0.93, 0.97] is about
    # +-6 SDs around nominal (tail bias at n_tail ~ 100 is ~1 point at most).
    rows = _gbm_terminal_matrix(4000, 2000, seed=4242)
    coverage = _coverage(rows, "asymptotic")
    assert 0.93 <= coverage <= 0.97


# --------------------------------------------------------------------------
# Bootstrap method
# --------------------------------------------------------------------------

def test_bootstrap_empirical_coverage_fast():
    # R = 100, N = 1000 (n_tail ~ 50), B = 200: ~0.6 s. The percentile
    # bootstrap of a tail mean under-covers with few tail points (measured
    # 0.91-0.93 across seeds). Floor 0.80: ~3.5 binomial SDs (0.029) below 0.90.
    rows = _gbm_terminal_matrix(100, 1000, seed=31)
    coverage = _coverage(rows, "bootstrap", n_bootstrap=200)
    assert 0.80 <= coverage <= 1.0


@pytest.mark.slow
def test_bootstrap_empirical_coverage_plan_500_replicates():
    # Plan: >= 0.90 over 500 replicates. N = 2000 (n_tail ~ 100) and B = 200
    # keep it at ~5 s; B = 200 adds Monte Carlo noise to the 2.5%/97.5%
    # endpoints but does not bias coverage materially. Measured 0.93-0.95
    # across seeds 1-3; floor 0.90 is ~2.5-4 binomial SDs (0.011) below.
    rows = _gbm_terminal_matrix(500, 2000, seed=20261122)
    coverage = _coverage(rows, "bootstrap", n_bootstrap=200)
    assert coverage >= 0.90
    assert coverage <= 0.99


def test_bootstrap_se_agrees_with_asymptotic_se():
    # Same sample, N = 2000, B = 1000: both estimate the same sampling SD;
    # the bootstrap SD has relative MC error ~1/sqrt(2B) = 2.2%. 15% allows
    # for that and for the small-tail difference between the two estimators.
    terminal = _gbm_terminal_matrix(1, 2000, seed=7)[0]
    asymptotic = cvar_interval(terminal, S0, P, method="asymptotic")
    bootstrap = cvar_interval(terminal, S0, P, method="bootstrap",
                              n_bootstrap=1000, seed=3)
    assert bootstrap["bootstrap_standard_error"] == pytest.approx(
        asymptotic["standard_error"], rel=0.15)
    assert bootstrap["lower"] < bootstrap["estimate"] < bootstrap["upper"]


def test_bootstrap_same_seed_reproducible_and_seed_reported():
    terminal = _gbm_terminal_matrix(1, 800, seed=9)[0]
    first = cvar_interval(terminal, S0, P, method="bootstrap", n_bootstrap=300, seed=17)
    second = cvar_interval(terminal, S0, P, method="bootstrap", n_bootstrap=300, seed=17)
    other = cvar_interval(terminal, S0, P, method="bootstrap", n_bootstrap=300, seed=18)
    assert first == second
    assert first["seed"] == 17 and first["n_bootstrap"] == 300
    assert isinstance(first["rng"], str) and first["rng"]
    assert (other["lower"], other["upper"]) != (first["lower"], first["upper"])
    assert other["estimate"] == first["estimate"]


@pytest.mark.parametrize("batch_size", [1, 7, 300, 10_000])
def test_bootstrap_independent_of_batch_size(batch_size):
    # Docstring: "The replicates are independent of batch_size".
    terminal = _gbm_terminal_matrix(1, 500, seed=10)[0]
    reference = cvar_interval(terminal, S0, P, method="bootstrap",
                              n_bootstrap=300, seed=4)
    batched = cvar_interval(terminal, S0, P, method="bootstrap",
                            n_bootstrap=300, seed=4, batch_size=batch_size)
    assert batched == reference


def test_bootstrap_matches_documented_rng_stream_and_definition():
    # Independent rebuild from the documented recipe: RNG
    # default_rng(SeedSequence(seed, spawn_key=(crc32(b"cvar_bootstrap"),))),
    # B resamples of size N with replacement, CVaR of each as risk_summary,
    # linear np.quantile at a/2, 1 - a/2 and ddof=1 SD. The order of index
    # draws (one resample after another) is the natural reading of the recipe.
    n, replicates, seed = 300, 200, 21
    terminal = _gbm_terminal_matrix(1, n, seed=12)[0]
    result = cvar_interval(terminal, S0, P, LEVEL, method="bootstrap",
                           n_bootstrap=replicates, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence(
        seed, spawn_key=(zlib.crc32(b"cvar_bootstrap"),)))
    cvars = []
    for _ in range(replicates):
        sample = terminal[rng.integers(0, n, size=n)]
        cvars.append(_summary_cvar(sample, S0, P))
    lower, upper = np.quantile(cvars, [0.025, 0.975])
    assert result["lower"] == pytest.approx(lower, rel=1e-13)
    assert result["upper"] == pytest.approx(upper, rel=1e-13)
    assert result["bootstrap_standard_error"] == pytest.approx(
        np.std(cvars, ddof=1), rel=1e-12)


def test_bootstrap_seed_none_runs_and_reports_none():
    terminal = _gbm_terminal_matrix(1, 400, seed=13)[0]
    result = cvar_interval(terminal, S0, P, method="bootstrap", n_bootstrap=50)
    assert result["seed"] is None
    assert result["lower"] <= result["upper"]
    assert result["estimate"] == _summary_cvar(terminal, S0, P)


def test_bootstrap_stream_does_not_touch_global_numpy_rng():
    np.random.seed(123)
    expected = np.random.random_sample()
    np.random.seed(123)
    cvar_interval(_gbm_terminal_matrix(1, 100, seed=1)[0], S0, P,
                  method="bootstrap", n_bootstrap=20, seed=1)
    assert np.random.random_sample() == expected


def test_bootstrap_memory_is_linear_in_n_with_small_batches():
    # With batch_size=1 the working set is O(N): no B x N matrix. Bound:
    # 16 float64 vectors of length N, versus B * N * 8 = 32 MB if the
    # resamples were materialised at once.
    n, replicates = 20_000, 200
    terminal = _gbm_terminal_matrix(1, n, seed=14)[0]
    tracemalloc.start()
    try:
        cvar_interval(terminal, S0, P, method="bootstrap",
                      n_bootstrap=replicates, seed=1, batch_size=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * n * 8
    assert peak < replicates * n * 8 / 10


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_sigma_zero_gives_point_interval(method):
    terminal = np.full(250, 90.0)  # sigma = 0: every loss is exactly 10
    result = cvar_interval(terminal, S0, P, method=method, n_bootstrap=20, seed=0)
    assert result["estimate"] == result["lower"] == result["upper"] == 10.0
    assert result["var"] == 10.0 and result["n_tail"] == 250
    se_key = "standard_error" if method == "asymptotic" else "bootstrap_standard_error"
    assert result[se_key] == 0.0
    json.dumps(result, allow_nan=False)


def test_asymptotic_n_one_has_no_interval():
    result = cvar_interval([95.0], S0, P)
    assert result["estimate"] == 5.0 and result["n"] == 1 and result["n_tail"] == 1
    assert result["lower"] is None and result["upper"] is None
    assert result["standard_error"] is None
    assert isinstance(result["note"], str) and result["note"]
    json.dumps(result, allow_nan=False)


def test_bootstrap_n_one_is_degenerate_point():
    result = cvar_interval([95.0], S0, P, method="bootstrap", n_bootstrap=10, seed=0)
    assert result["estimate"] == result["lower"] == result["upper"] == 5.0
    assert result["bootstrap_standard_error"] == 0.0
    json.dumps(result, allow_nan=False)


def test_asymptotic_single_tail_point_has_no_interval():
    # N = 10, p = 0.95: linear percentile rank 8.55 -> only the maximum loss
    # is >= VaR for distinct values, so n_tail = 1 and the variance is undefined.
    terminal = S0 - np.arange(1.0, 11.0)
    result = cvar_interval(terminal, S0, P)
    assert result["n_tail"] == 1 and result["estimate"] == 10.0
    assert result["lower"] is None and result["upper"] is None
    assert result["standard_error"] is None and "note" in result
    json.dumps(result, allow_nan=False)


def test_asymptotic_two_tail_points_gives_finite_interval():
    terminal = S0 - np.array([1.0, 2.0, 3.0, 9.0, 9.0])
    result = cvar_interval(terminal, S0, 0.9)
    assert result["n_tail"] == 2 and result["estimate"] == 9.0
    assert result["standard_error"] == 0.0  # tail all at VaR
    assert result["lower"] == result["upper"] == 9.0
    assert "note" not in result


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_asymptotic_scales_linearly_without_overflow(scale):
    base = _gbm_terminal_matrix(1, 400, seed=15)[0]
    reference = cvar_interval(base, S0, P)
    scaled = cvar_interval(base * scale, S0 * scale, P)
    for key in ("estimate", "standard_error", "lower", "upper"):
        assert scaled[key] == pytest.approx(reference[key] * scale, rel=1e-10)


def test_nonfinite_losses_raise():
    with pytest.raises(ValueError):
        cvar_interval([-1e308, 1e308], 1e308, P)


# --------------------------------------------------------------------------
# Invalid inputs and output contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("terminal", [
    [], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf], [True, False],
    np.array([1 + 0j]), ["100"], None,
])
@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_invalid_terminal_raises(terminal, method):
    with pytest.raises(ValueError):
        cvar_interval(terminal, S0, P, method=method)


@pytest.mark.parametrize("kwargs", [
    {"s0": True}, {"s0": np.nan}, {"s0": "100"}, {"s0": 10**400},
    {"confidence": 0.0}, {"confidence": 1.0}, {"confidence": np.nan},
    {"confidence": True}, {"level": 0.0}, {"level": 1.0}, {"level": -0.5},
    {"method": "normal"}, {"method": None}, {"method": "Bootstrap"},
])
def test_invalid_scalars_raise(kwargs):
    arguments = {"terminal": [90.0, 95.0, 99.0], "s0": S0, "confidence": P}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        cvar_interval(**arguments)


@pytest.mark.parametrize("kwargs", [
    {"n_bootstrap": 1}, {"n_bootstrap": 0}, {"n_bootstrap": 2.5},
    {"n_bootstrap": True}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
    {"seed": "1"}, {"batch_size": 0}, {"batch_size": 1.0}, {"batch_size": True},
])
def test_invalid_bootstrap_options_raise(kwargs):
    with pytest.raises(ValueError):
        cvar_interval([90.0, 95.0, 99.0], S0, P, method="bootstrap", **kwargs)


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap"])
def test_input_not_modified(method):
    terminal = _gbm_terminal_matrix(1, 200, seed=16)[0]
    snapshot = terminal.copy()
    cvar_interval(terminal, S0, P, method=method, n_bootstrap=20, seed=0)
    np.testing.assert_array_equal(terminal, snapshot)


def test_output_contract_and_json_safe_asymptotic():
    result = cvar_interval(_gbm_terminal_matrix(1, 300, seed=17)[0], S0, P, LEVEL)
    assert set(result) == {"estimate", "lower", "upper", "method", "level",
                           "confidence", "n", "n_tail", "var", "standard_error"}
    assert result["method"] == "asymptotic"
    assert result["level"] == LEVEL and result["confidence"] == P
    assert type(result["n"]) is int and type(result["n_tail"]) is int
    for key in ("estimate", "lower", "upper", "var", "standard_error"):
        assert type(result[key]) is float
    assert result["lower"] < result["estimate"] < result["upper"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_output_contract_and_json_safe_bootstrap():
    result = cvar_interval(np.arange(80, 120), np.int64(100), 0.9, 0.8,
                           method="bootstrap", n_bootstrap=np.int64(40), seed=np.int64(3))
    assert set(result) == {"estimate", "lower", "upper", "method", "level",
                           "confidence", "n", "n_tail", "var",
                           "bootstrap_standard_error", "n_bootstrap", "seed", "rng"}
    assert result["method"] == "bootstrap"
    for key in ("n", "n_tail", "n_bootstrap", "seed"):
        assert type(result[key]) is int
    for key in ("estimate", "lower", "upper", "var", "bootstrap_standard_error"):
        assert type(result[key]) is float
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_wider_level_gives_wider_asymptotic_interval():
    terminal = _gbm_terminal_matrix(1, 1000, seed=18)[0]
    narrow = cvar_interval(terminal, S0, P, 0.8)
    wide = cvar_interval(terminal, S0, P, 0.99)
    assert wide["lower"] < narrow["lower"] < narrow["upper"] < wide["upper"]
    assert wide["standard_error"] == narrow["standard_error"]
