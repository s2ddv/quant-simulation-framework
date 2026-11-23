"""Law-level tests for the Merton and Kou jump-diffusion simulators (E4.1/E4.2).

Model (``simulation.jump_diffusion``): prices in asset currency, ``T`` in years,
``mu`` and ``lambda`` in year^-1, ``sigma`` in year^-1/2, log-jump parameters
dimensionless. With ``a = (mu - sigma^2/2 - lambda*k)T``::

    ln S_T = ln S0 + a + sigma*sqrt(T)*Z + sum_{i <= N_T} ln J_i,  N_T ~ Poisson(lambda*T).

Independent references written here (not copied from the code under test):

* raw moments ``E[S_T^q] = S0^q exp(q a + q^2 sigma^2 T / 2 + lambda T (E[J^q] - 1))``
  (Poisson compound MGF), with ``E[J^q] = exp(q m + q^2 delta^2/2)`` (Merton)
  and ``p eta1/(eta1 - q) + (1 - p) eta2/(eta2 + q)`` (Kou, ``eta1 > q``);
  cross-checked against the plan formula for ``E[S^2]``, an explicit Poisson
  sum over normal components (Merton) and ``scipy.integrate.quad`` (Kou).
* Merton CDF of ``S_T`` as ``sum_n poisson.pmf(n) * norm.cdf(...)``.
* Reference samplers that draw every jump individually (no Gamma shortcut).

Statistical tolerances (plan section 2.5): seeds fixed; means and variances at
``k = 4`` standard errors (false alarm ~6e-5 per assertion under asymptotic
normality). The SE of a sample mean uses the analytic variance; the SE of the
sample variance ``s^2`` is ``sqrt((mu4 - sigma^4 (n-3)/(n-1)) / n)`` with the
analytic fourth central moment ``mu4`` obtained from the raw moments above
(finite for all Merton parameters, and for Kou only when ``eta_up > 4``; the
fourth-moment estimate itself is not needed). Whole distributions: KS with a
minimum p-value of 1e-3 at a fixed seed. Quantiles: the analytic quantile must
lie inside the distribution-free order-statistic interval of E1.2
(``risk.intervals.var_interval``) at level 0.999. Tests marked ``slow`` repeat
the main checks with larger samples (they still run by default; deselect with
``-m "not slow"``).

Memory: every path matrix here has at most (steps + 1) * n_sims ~ 5.1e6
elements (~41 MB), below the ~2e7 element budget.
"""

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

from risk.intervals import var_interval
from simulation.gbm import GbmSimulator
from simulation.jump_diffusion import KouJumpSimulator, MertonJumpSimulator
from validation.jump_theory import (
    kou_terminal_moments,
    merton_terminal_cdf,
    merton_terminal_moments,
    merton_terminal_quantile,
)
from validation.statistics import terminal_quantile

S0 = 100.0
T = 1.0
MERTON = dict(mu=0.08, sigma=0.2, jump_intensity=3.0, jump_mean=-0.05, jump_std=0.1)
KOU = dict(mu=0.08, sigma=0.2, jump_intensity=3.0, p_up=0.4, eta_up=10.0, eta_down=5.0)
KS_MIN_P = 1e-3
K_SE = 4.0
CI_LEVEL = 0.999


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------
def _merton_jump_moment(q: float, m: float, d: float) -> float:
    """``E[J^q]`` for ``ln J ~ N(m, d^2)`` (lognormal moment)."""
    return math.exp(q * m + 0.5 * q * q * d * d)


def _kou_jump_moment(q: float, p: float, e1: float, e2: float) -> float:
    """``E[J^q]`` for the double exponential law (requires ``e1 > q`` if ``p > 0``)."""
    down = (1 - p) * e2 / (e2 + q)
    if p == 0:
        return down
    assert e1 > q
    return p * e1 / (e1 - q) + down


def _raw_moment(q: int, mu: float, sigma: float, lam: float,
                jump_moment: Any) -> float:
    """``E[S_T^q]`` via the compound Poisson MGF (independent derivation)."""
    k = jump_moment(1) - 1.0
    a = (mu - 0.5 * sigma**2 - lam * k) * T
    return S0**q * math.exp(q * a + 0.5 * q * q * sigma**2 * T + lam * T * (jump_moment(q) - 1.0))


def _reference_moments(mu: float, sigma: float, lam: float,
                       jump_moment: Any) -> dict[str, float]:
    """Mean, variance and fourth central moment from raw moments 1..4."""
    r1, r2, r3, r4 = (_raw_moment(q, mu, sigma, lam, jump_moment) for q in (1, 2, 3, 4))
    var = r2 - r1**2
    mu4 = r4 - 4 * r3 * r1 + 6 * r2 * r1**2 - 3 * r1**4
    return {"mean": r1, "variance": var, "mu4": mu4}


def _merton_ref(params: dict[str, float]) -> dict[str, float]:
    m, d = params["jump_mean"], params["jump_std"]
    return _reference_moments(params["mu"], params["sigma"], params["jump_intensity"],
                              lambda q: _merton_jump_moment(q, m, d))


def _kou_ref(params: dict[str, float]) -> dict[str, float]:
    p, e1, e2 = params["p_up"], params["eta_up"], params["eta_down"]
    return _reference_moments(params["mu"], params["sigma"], params["jump_intensity"],
                              lambda q: _kou_jump_moment(q, p, e1, e2))


def _merton_components(params: dict[str, float], tail: float = 1e-16) -> tuple[np.ndarray, ...]:
    """Poisson weights and normal (mean, sd) of ``ln S_T`` given ``N_T = n``."""
    lam, m, d = params["jump_intensity"], params["jump_mean"], params["jump_std"]
    sigma, mu = params["sigma"], params["mu"]
    k = math.exp(m + 0.5 * d * d) - 1.0
    n_max = int(stats.poisson.isf(tail, lam * T)) + 1 if lam > 0 else 0
    n = np.arange(n_max + 1, dtype=float)
    weights = stats.poisson.pmf(n, lam * T) if lam > 0 else np.ones(1)
    means = math.log(S0) + (mu - 0.5 * sigma**2 - lam * k) * T + n * m
    sds = np.sqrt(sigma**2 * T + n * d * d)
    return weights, means, sds


def _merton_cdf_ref(x: Any, params: dict[str, float]) -> np.ndarray:
    """``P(S_T <= x)`` as an explicit Poisson mixture of normal CDFs (sd > 0)."""
    weights, means, sds = _merton_components(params)
    y = np.log(np.asarray(x, dtype=float))[..., None]
    return np.asarray(stats.norm.cdf((y - means) / sds) @ weights)


def _reference_jump_sums(rng: np.random.Generator, counts: np.ndarray,
                         draw_one: Any) -> np.ndarray:
    """Sum ``counts[i]`` individually drawn log-jumps (no aggregation shortcut)."""
    total = int(counts.sum())
    jumps = draw_one(rng, total)
    owner = np.repeat(np.arange(counts.size), counts)
    return np.bincount(owner, weights=jumps, minlength=counts.size)


def _reference_terminal(n: int, seed: int, mu: float, sigma: float, lam: float,
                        k: float, draw_one: Any) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    counts = rng.poisson(lam * T, n)
    sums = _reference_jump_sums(rng, counts, draw_one)
    return S0 * np.exp((mu - 0.5 * sigma**2 - lam * k) * T + sigma * math.sqrt(T) * z + sums)


def _merton_draw(m: float, d: float) -> Any:
    return lambda rng, size: rng.normal(m, d, size)


def _kou_draw(p: float, e1: float, e2: float) -> Any:
    def draw(rng: np.random.Generator, size: int) -> np.ndarray:
        up = rng.random(size) < p
        return np.where(up, rng.exponential(1 / e1, size), -rng.exponential(1 / e2, size))
    return draw


def _assert_mean_and_variance(sample: np.ndarray, ref: dict[str, float]) -> tuple[float, float]:
    """Assert ``|mean - E| <= 4 SE`` and ``|s^2 - Var| <= 4 SE(s^2)``; return z-scores."""
    n = sample.size
    se_mean = math.sqrt(ref["variance"] / n)
    z_mean = (sample.mean() - ref["mean"]) / se_mean
    var_of_s2 = (ref["mu4"] - ref["variance"] ** 2 * (n - 3) / (n - 1)) / n
    z_var = (sample.var(ddof=1) - ref["variance"]) / math.sqrt(var_of_s2)
    assert abs(z_mean) <= K_SE, f"mean z = {z_mean:.3f}"
    assert abs(z_var) <= K_SE, f"variance z = {z_var:.3f}"
    return z_mean, z_var


def _gbm_log_law(mu: float, sigma: float, t: float = T) -> Any:
    return stats.norm(loc=math.log(S0) + (mu - 0.5 * sigma**2) * t, scale=sigma * math.sqrt(t))


# --------------------------------------------------------------------------
# Smoke tests (original six, kept)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls, params", [(MertonJumpSimulator, MERTON),
                                         (KouJumpSimulator, KOU)])
def test_paths_shape_row_zero_and_determinism(cls, params):
    paths = cls(100.0, 1.0, 12, 50, 3, **params).generate_paths()
    assert paths.shape == (13, 50) and paths.dtype == np.float64
    assert np.all(paths[0] == 100.0)
    assert np.isfinite(paths).all() and np.all(paths > 0)
    again = cls(100.0, 1.0, 12, 50, 3, **params).generate_paths()
    np.testing.assert_array_equal(paths, again)
    terminal = cls(100.0, 1.0, 12, 50, 3, **params).generate_terminal()
    assert terminal.shape == (50,) and terminal.dtype == np.float64


@pytest.mark.parametrize("cls, params, theory", [
    (MertonJumpSimulator, MERTON,
     merton_terminal_moments(100.0, 0.08, 0.2, 1.0, 3.0, -0.05, 0.1)),
    (KouJumpSimulator, KOU,
     kou_terminal_moments(100.0, 0.08, 0.2, 1.0, 3.0, 0.4, 10.0, 5.0)),
])
def test_terminal_mean_within_four_standard_errors(cls, params, theory):
    # Seeded; |mean - S0 e^{muT}| <= 4 SE (plan section 2.5).
    n = 200_000
    terminal = cls(100.0, 1.0, 1, n, 11, **params).generate_terminal()
    se = terminal.std(ddof=1) / math.sqrt(n)
    assert abs(terminal.mean() - theory["mean"]) <= 4 * se
    assert theory["mean"] == pytest.approx(100.0 * math.exp(0.08))


def test_zero_intensity_quantile_matches_gbm():
    for p in (0.01, 0.5, 0.95):
        jump = merton_terminal_quantile(p, 100.0, 0.08, 0.2, 1.0, 0.0, -0.05, 0.1)
        assert jump == pytest.approx(terminal_quantile(100.0, 0.08, 0.2, 1.0, p),
                                     rel=1e-11)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        MertonJumpSimulator(100.0, 1.0, 1, 1, **{**MERTON, "jump_intensity": True})
    with pytest.raises(ValueError):
        MertonJumpSimulator(100.0, 1.0, 1, 1, **{**MERTON, "jump_std": float("nan")})
    with pytest.raises(ValueError):
        KouJumpSimulator(100.0, 1.0, 1, 1, **{**KOU, "eta_up": 1.0})


# --------------------------------------------------------------------------
# 1. lambda = 0 recovers the GBM law
# --------------------------------------------------------------------------
def _zero_intensity(cls: Any, n: int, seed: int, steps: int = 1, **extra: Any) -> Any:
    params = {**(MERTON if cls is MertonJumpSimulator else KOU), "jump_intensity": 0.0, **extra}
    return cls(S0, T, steps, n, seed, **params)


@pytest.mark.parametrize("cls", [MertonJumpSimulator, KouJumpSimulator])
def test_zero_intensity_terminal_has_gbm_law(cls):
    # N = 50_000, seed 101: one-sample KS of ln S_T against the analytic
    # N(ln S0 + (mu - sigma^2/2)T, sigma^2 T), two-sample KS against an
    # independent GbmSimulator sample (seed 202), mean/variance at 4 SE.
    n = 50_000
    terminal = _zero_intensity(cls, n, 101).generate_terminal()
    law = _gbm_log_law(0.08, 0.2)
    assert stats.kstest(np.log(terminal), law.cdf).pvalue > KS_MIN_P
    gbm = GbmSimulator(S0, T, 1, n, 202, mu=0.08, sigma=0.2).generate_terminal()
    assert stats.ks_2samp(terminal, gbm).pvalue > KS_MIN_P
    ref = _reference_moments(0.08, 0.2, 0.0, lambda q: 1.0)
    _assert_mean_and_variance(terminal, ref)


@pytest.mark.parametrize("cls", [MertonJumpSimulator, KouJumpSimulator])
def test_zero_intensity_paths_have_gbm_law(cls):
    # steps = 50, N = 20_000 (1.02e6 elements). The terminal row and the
    # mid-horizon row (t = 0.5) must follow the GBM marginal law.
    steps, n = 50, 20_000
    paths = _zero_intensity(cls, n, 103, steps=steps).generate_paths()
    assert stats.kstest(np.log(paths[-1]), _gbm_log_law(0.08, 0.2).cdf).pvalue > KS_MIN_P
    mid = _gbm_log_law(0.08, 0.2, t=0.5)
    assert stats.kstest(np.log(paths[steps // 2]), mid.cdf).pvalue > KS_MIN_P
    ref = _reference_moments(0.08, 0.2, 0.0, lambda q: 1.0)
    _assert_mean_and_variance(paths[-1], ref)


@pytest.mark.slow
def test_zero_intensity_gbm_law_large_sample():
    n = 1_000_000
    terminal = _zero_intensity(MertonJumpSimulator, n, 104).generate_terminal()
    assert stats.kstest(np.log(terminal), _gbm_log_law(0.08, 0.2).cdf).pvalue > KS_MIN_P
    _assert_mean_and_variance(terminal, _reference_moments(0.08, 0.2, 0.0, lambda q: 1.0))


# --------------------------------------------------------------------------
# 2. Terminal mean and variance against theory; theory against references
# --------------------------------------------------------------------------
MERTON_CASES = [MERTON, dict(MERTON, jump_intensity=0.5, jump_mean=-0.2, jump_std=0.25),
                dict(MERTON, jump_mean=0.1, jump_std=0.0)]
KOU_CASES = [KOU, dict(KOU, p_up=0.0), dict(KOU, p_up=1.0, eta_up=6.0),
             dict(KOU, eta_up=1.5), dict(KOU, eta_up=2.0)]


@pytest.mark.parametrize("params", MERTON_CASES)
def test_merton_theory_matches_independent_references(params):
    theory = merton_terminal_moments(S0, params["mu"], params["sigma"], T,
                                     params["jump_intensity"], params["jump_mean"],
                                     params["jump_std"])
    ref = _merton_ref(params)
    assert theory["mean"] == pytest.approx(S0 * math.exp(params["mu"] * T), rel=1e-13)
    assert theory["mean"] == pytest.approx(ref["mean"], rel=1e-12)
    assert theory["variance"] == pytest.approx(ref["variance"], rel=1e-10)
    # Plan E4.1 formula, written out.
    m, d, lam, mu, sigma = (params[key] for key in
                            ("jump_mean", "jump_std", "jump_intensity", "mu", "sigma"))
    k = math.exp(m + d * d / 2) - 1
    plan_second = S0**2 * math.exp((2 * mu + sigma**2) * T
                                   + lam * T * (math.exp(2 * m + 2 * d * d) - 1 - 2 * k))
    assert theory["second_moment"] == pytest.approx(plan_second, rel=1e-12)
    # Explicit Poisson sum of lognormal moments of each normal component.
    weights, means, sds = _merton_components(params)
    first = float(weights @ np.exp(means + sds**2 / 2))
    second = float(weights @ np.exp(2 * means + 2 * sds**2))
    assert theory["mean"] == pytest.approx(first, rel=1e-12)
    assert theory["second_moment"] == pytest.approx(second, rel=1e-12)


@pytest.mark.parametrize("params", [KOU, dict(KOU, p_up=0.0), dict(KOU, p_up=1.0, eta_up=6.0)])
def test_kou_theory_matches_independent_references(params):
    p, e1, e2 = params["p_up"], params["eta_up"], params["eta_down"]
    theory = kou_terminal_moments(S0, params["mu"], params["sigma"], T,
                                  params["jump_intensity"], p, e1, e2)
    ref = _kou_ref(params)
    assert theory["mean"] == pytest.approx(ref["mean"], rel=1e-12)
    assert theory["variance"] == pytest.approx(ref["variance"], rel=1e-10)

    # E[J^q] by numerical integration of the double-exponential density.
    def quad_moment(q: float) -> float:
        up = integrate.quad(lambda y: e1 * math.exp((q - e1) * y), 0, np.inf)[0]
        down = integrate.quad(lambda y: e2 * math.exp(-(q + e2) * y), 0, np.inf)[0]
        return p * up + (1 - p) * down

    numeric = _reference_moments(params["mu"], params["sigma"], params["jump_intensity"],
                                 quad_moment)
    assert theory["mean"] == pytest.approx(numeric["mean"], rel=1e-9)
    assert theory["variance"] == pytest.approx(numeric["variance"], rel=1e-8)


def _merton_sim(params: dict[str, float], n: int, seed: int, steps: int = 1) -> MertonJumpSimulator:
    return MertonJumpSimulator(S0, T, steps, n, seed, **params)


def _kou_sim(params: dict[str, float], n: int, seed: int, steps: int = 1) -> KouJumpSimulator:
    return KouJumpSimulator(S0, T, steps, n, seed, **params)


@pytest.mark.parametrize("params", MERTON_CASES)
def test_merton_terminal_mean_and_variance(params):
    # N = 200_000, seed 21; theory from jump_theory, SE from the reference mu4.
    terminal = _merton_sim(params, 200_000, 21).generate_terminal()
    theory = merton_terminal_moments(S0, params["mu"], params["sigma"], T,
                                     params["jump_intensity"], params["jump_mean"],
                                     params["jump_std"])
    ref = {**_merton_ref(params), "mean": theory["mean"], "variance": theory["variance"]}
    _assert_mean_and_variance(terminal, ref)


@pytest.mark.parametrize("params", KOU_CASES[:3])
def test_kou_terminal_mean_and_variance(params):
    # eta_up in {10, 6} > 4: the sample variance has a finite variance.
    terminal = _kou_sim(params, 200_000, 22).generate_terminal()
    theory = kou_terminal_moments(S0, params["mu"], params["sigma"], T,
                                  params["jump_intensity"], params["p_up"],
                                  params["eta_up"], params["eta_down"])
    ref = {**_kou_ref(params), "mean": theory["mean"], "variance": theory["variance"]}
    _assert_mean_and_variance(terminal, ref)


@pytest.mark.parametrize("make, params, ref", [
    (_merton_sim, MERTON, _merton_ref(MERTON)),
    (_kou_sim, KOU, _kou_ref(KOU)),
])
def test_paths_terminal_row_mean_and_variance(make, params, ref):
    # 252 daily steps, N = 20_000 (5.06e6 elements, ~41 MB), seed 31.
    paths = make(params, 20_000, 31, steps=252).generate_paths()
    _assert_mean_and_variance(paths[-1], ref)


@pytest.mark.parametrize("params", [dict(KOU, eta_up=1.5), dict(KOU, eta_up=2.0)])
def test_kou_infinite_second_moment_contract(params):
    # eta_up in (1, 2]: E[J^2] = inf, so second_moment and variance are None
    # while the mean is still S0 e^{muT}; the simulator accepts the law.
    theory = kou_terminal_moments(S0, params["mu"], params["sigma"], T,
                                  params["jump_intensity"], params["p_up"],
                                  params["eta_up"], params["eta_down"])
    assert theory["second_moment"] is None and theory["variance"] is None
    assert theory["mean"] == pytest.approx(S0 * math.exp(params["mu"] * T))
    terminal = _kou_sim(params, 1_000, 5).generate_terminal()
    assert np.isfinite(terminal).all() and np.all(terminal > 0)
    # Documented: with lambda*T = 0 the jump law is irrelevant.
    no_jumps = kou_terminal_moments(S0, params["mu"], params["sigma"], T, 0.0,
                                    params["p_up"], params["eta_up"], params["eta_down"])
    assert no_jumps["variance"] == pytest.approx(
        _reference_moments(params["mu"], params["sigma"], 0.0, lambda q: 1.0)["variance"])


@pytest.mark.slow
@pytest.mark.parametrize("make, params, ref", [
    (_merton_sim, MERTON, _merton_ref(MERTON)),
    (_kou_sim, KOU, _kou_ref(KOU)),
])
def test_terminal_mean_and_variance_large_sample(make, params, ref):
    _assert_mean_and_variance(make(params, 2_000_000, 23).generate_terminal(), ref)


@pytest.mark.slow
def test_merton_paths_mean_unbiased_over_seeds():
    # 40 seeds x N = 20_000 x 252 steps (the backend's seed-3 concern):
    # pooled z = mean(z_i) * sqrt(40) must be within 4, and the spread of the
    # z_i must be compatible with 1 (chi-square two-sided at 1e-3).
    ref = _merton_ref(MERTON)
    z = np.array([(_merton_sim(MERTON, 20_000, seed, steps=252).generate_paths()[-1].mean()
                   - ref["mean"]) / math.sqrt(ref["variance"] / 20_000) for seed in range(40)])
    assert abs(z.mean() * math.sqrt(z.size)) <= K_SE
    chi2 = float(np.sum(z**2))
    assert stats.chi2.ppf(5e-4, z.size) <= chi2 <= stats.chi2.ppf(1 - 5e-4, z.size)


# --------------------------------------------------------------------------
# 3. Whole-law and quantile checks
# --------------------------------------------------------------------------
def _quantile_checks(terminal: np.ndarray, params: dict[str, float]) -> None:
    args = (S0, params["mu"], params["sigma"], T, params["jump_intensity"],
            params["jump_mean"], params["jump_std"])
    for p in (0.01, 0.05, 0.95):
        q = merton_terminal_quantile(p, *args)
        # Independent check of the analytic quantile with the test's own CDF.
        assert float(_merton_cdf_ref(q, params)) == pytest.approx(p, abs=1e-10)
        # S_T quantile p <=> loss L = S0 - S_T quantile c = 1 - p.
        interval = var_interval(terminal, S0, confidence=1 - p, level=CI_LEVEL)
        assert not interval["saturated"]
        assert interval["lower"] <= S0 - q <= interval["upper"], (p, q, interval)


def test_merton_terminal_quantiles_inside_order_statistic_interval():
    # N = 100_000, seed 41: three intervals at level 0.999 (joint false-alarm
    # probability <= 3e-3, conservative order-statistic coverage).
    params = dict(MERTON, jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15)
    _quantile_checks(_merton_sim(params, 100_000, 41).generate_terminal(), params)


@pytest.mark.slow
def test_merton_terminal_quantiles_large_sample():
    params = dict(MERTON, jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15)
    _quantile_checks(_merton_sim(params, 1_000_000, 42).generate_terminal(), params)


def test_merton_cdf_matches_independent_mixture():
    x = np.linspace(40.0, 200.0, 41)
    for params in MERTON_CASES[:2]:
        module = merton_terminal_cdf(x, S0, params["mu"], params["sigma"], T,
                                     params["jump_intensity"], params["jump_mean"],
                                     params["jump_std"])
        np.testing.assert_allclose(module, _merton_cdf_ref(x, params), atol=1e-13)


@pytest.mark.parametrize("params", MERTON_CASES[:2])
def test_merton_terminal_ks_against_mixture_cdf(params):
    terminal = _merton_sim(params, 50_000, 43).generate_terminal()
    assert stats.kstest(terminal, lambda x: _merton_cdf_ref(x, params)).pvalue > KS_MIN_P


@pytest.mark.parametrize("params", [MERTON, dict(MERTON, jump_intensity=0.5, jump_mean=-0.2, jump_std=0.25)])
def test_merton_paths_ks_against_mixture_cdf(params):
    paths = _merton_sim(params, 20_000, 44, steps=100).generate_paths()
    assert stats.kstest(paths[-1], lambda x: _merton_cdf_ref(x, params)).pvalue > KS_MIN_P


@pytest.mark.parametrize("params", KOU_CASES[:3] + [dict(KOU, eta_up=1.5)])
def test_kou_terminal_ks_against_individual_jump_sampler(params):
    # The reference draws every jump separately (Bernoulli sign and
    # exponential size) instead of Binomial + Gamma aggregates; N = 50_000.
    p, e1, e2 = params["p_up"], params["eta_up"], params["eta_down"]
    k = _kou_jump_moment(1, p, e1, e2) - 1
    ref = _reference_terminal(50_000, 9001, params["mu"], params["sigma"],
                              params["jump_intensity"], k, _kou_draw(p, e1, e2))
    terminal = _kou_sim(params, 50_000, 45).generate_terminal()
    assert stats.ks_2samp(terminal, ref).pvalue > KS_MIN_P
    paths = _kou_sim(params, 20_000, 46, steps=50).generate_paths()
    assert stats.ks_2samp(paths[-1], ref).pvalue > KS_MIN_P


def test_merton_terminal_ks_against_individual_jump_sampler():
    m, d = MERTON["jump_mean"], MERTON["jump_std"]
    ref = _reference_terminal(50_000, 9002, MERTON["mu"], MERTON["sigma"],
                              MERTON["jump_intensity"], math.exp(m + d * d / 2) - 1,
                              _merton_draw(m, d))
    terminal = _merton_sim(MERTON, 50_000, 47).generate_terminal()
    assert stats.ks_2samp(terminal, ref).pvalue > KS_MIN_P


@pytest.mark.slow
@pytest.mark.parametrize("params", [KOU, dict(KOU, p_up=1.0, eta_up=6.0)])
def test_kou_ks_against_individual_jump_sampler_large_sample(params):
    p, e1, e2 = params["p_up"], params["eta_up"], params["eta_down"]
    k = _kou_jump_moment(1, p, e1, e2) - 1
    ref = _reference_terminal(500_000, 9003, params["mu"], params["sigma"],
                              params["jump_intensity"], k, _kou_draw(p, e1, e2))
    terminal = _kou_sim(params, 500_000, 48).generate_terminal()
    assert stats.ks_2samp(terminal, ref).pvalue > KS_MIN_P


# --------------------------------------------------------------------------
# 4. Large lambda*dt, deterministic jumps
# --------------------------------------------------------------------------
LARGE = dict(mu=0.08, sigma=0.2, jump_intensity=50 * 252.0, jump_mean=0.0, jump_std=0.002)


def test_large_intensity_per_step_no_warnings_and_correct_mean():
    # lambda = 12_600/year, steps = 252 => lambda*dt = 50 jumps per step;
    # lambda*T*delta^2 = 0.0504 keeps the law moderate. All floating-point
    # warnings are turned into errors (both NumPy errstate and warnings).
    ref = _merton_ref(LARGE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            paths = _merton_sim(LARGE, 20_000, 51, steps=252).generate_paths()
            terminal = _merton_sim(LARGE, 200_000, 52).generate_terminal()
            kou = KouJumpSimulator(S0, T, 252, 5_000, 53, mu=0.08, sigma=0.2,
                                   jump_intensity=50 * 252.0, p_up=0.5,
                                   eta_up=500.0, eta_down=500.0).generate_paths()
    assert np.isfinite(paths).all() and np.isfinite(kou).all()
    _assert_mean_and_variance(paths[-1], ref)
    _assert_mean_and_variance(terminal, ref)
    kou_ref = _reference_moments(0.08, 0.2, 50 * 252.0,
                                 lambda q: _kou_jump_moment(q, 0.5, 500.0, 500.0))
    assert abs(kou[-1].mean() - kou_ref["mean"]) <= K_SE * math.sqrt(kou_ref["variance"] / 5_000)


def test_extreme_intensity_terminal_is_finite():
    # lambda*T = 1e7 (Poisson counts ~1e7 per draw), tiny jumps.
    params = dict(mu=0.0, sigma=0.1, jump_intensity=1e7, jump_mean=0.0, jump_std=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            terminal = _merton_sim(params, 50_000, 54).generate_terminal()
    assert np.isfinite(terminal).all() and np.all(terminal > 0)
    _assert_mean_and_variance(terminal, _merton_ref(params))


DETERMINISTIC = dict(mu=0.05, sigma=0.15, jump_intensity=2.0, jump_mean=-0.08, jump_std=0.0)


def test_deterministic_jumps_mean_and_cdf():
    # delta = 0: ln S_T = GBM log price - lambda*k*T + N_T*m (shifted GBM).
    terminal = _merton_sim(DETERMINISTIC, 100_000, 61).generate_terminal()
    _assert_mean_and_variance(terminal, _merton_ref(DETERMINISTIC))
    assert stats.kstest(terminal, lambda x: _merton_cdf_ref(x, DETERMINISTIC)).pvalue > KS_MIN_P
    x = np.array([60.0, 90.0, 100.0, 110.0, 150.0])
    module = merton_terminal_cdf(x, S0, *(DETERMINISTIC[key] for key in
                                          ("mu", "sigma")), T,
                                 *(DETERMINISTIC[key] for key in
                                   ("jump_intensity", "jump_mean", "jump_std")))
    np.testing.assert_allclose(module, _merton_cdf_ref(x, DETERMINISTIC), atol=1e-13)


def test_deterministic_jumps_without_diffusion_count_law():
    # sigma = 0 and delta = 0: ln(S_T/S0) - a = N_T * m exactly, so the
    # implied counts must be integers with a Poisson(lambda*T) law
    # (chi-square goodness of fit on cells 0..6 and >= 7, p > 1e-3).
    params = dict(DETERMINISTIC, sigma=0.0)
    lam, m = params["jump_intensity"], params["jump_mean"]
    k = math.exp(m) - 1
    terminal = _merton_sim(params, 50_000, 62).generate_terminal()
    implied = (np.log(terminal / S0) - (params["mu"] - lam * k) * T) / m
    counts = np.rint(implied)
    np.testing.assert_allclose(implied, counts, atol=1e-9)
    observed = np.array([np.sum(counts == j) for j in range(7)] + [np.sum(counts >= 7)])
    probs = np.append(stats.poisson.pmf(np.arange(7), lam * T), stats.poisson.sf(6, lam * T))
    assert stats.chisquare(observed, probs * counts.size).pvalue > KS_MIN_P


# --------------------------------------------------------------------------
# 5. Input validation
# --------------------------------------------------------------------------
NAN, INF = float("nan"), float("inf")
MERTON_BAD = [("mu", True), ("mu", np.bool_(True)), ("mu", NAN), ("mu", INF),
              ("sigma", -0.1), ("sigma", NAN), ("sigma", True),
              ("jump_intensity", -1.0), ("jump_intensity", INF), ("jump_intensity", True),
              ("jump_mean", NAN), ("jump_mean", -INF), ("jump_mean", 1000.0),
              ("jump_std", -0.1), ("jump_std", INF), ("jump_std", False),
              ("jump_mean", "0.1")]
KOU_BAD = [("p_up", -0.1), ("p_up", 1.1), ("p_up", NAN), ("p_up", True),
           ("eta_up", 1.0), ("eta_up", 0.5), ("eta_up", -3.0), ("eta_up", INF), ("eta_up", NAN),
           ("eta_down", 0.0), ("eta_down", -1.0), ("eta_down", NAN), ("eta_down", INF),
           ("sigma", -0.2), ("jump_intensity", -0.5), ("mu", NAN)]


@pytest.mark.parametrize("name, value", MERTON_BAD)
def test_merton_rejects_invalid_parameters(name, value):
    with pytest.raises(ValueError):
        MertonJumpSimulator(S0, T, 1, 1, 0, **{**MERTON, name: value})


@pytest.mark.parametrize("name, value", KOU_BAD)
def test_kou_rejects_invalid_parameters(name, value):
    with pytest.raises(ValueError):
        KouJumpSimulator(S0, T, 1, 1, 0, **{**KOU, name: value})


@pytest.mark.parametrize("name, value", [("p_up", 1.1), ("eta_up", 1.0), ("eta_up", 0.5),
                                         ("eta_down", 0.0), ("sigma", -0.1),
                                         ("jump_intensity", -1.0), ("mu", True), ("mu", NAN)])
def test_kou_theory_rejects_invalid_parameters(name, value):
    params = {**KOU, name: value}
    with pytest.raises(ValueError):
        kou_terminal_moments(S0, params["mu"], params["sigma"], T, params["jump_intensity"],
                             params["p_up"], params["eta_up"], params["eta_down"])


@pytest.mark.parametrize("name, value", [("jump_std", -0.1), ("sigma", -0.1),
                                         ("jump_intensity", -1.0), ("mu", True),
                                         ("jump_mean", NAN)])
def test_merton_theory_rejects_invalid_parameters(name, value):
    params = {**MERTON, name: value}
    with pytest.raises(ValueError):
        merton_terminal_moments(S0, params["mu"], params["sigma"], T,
                                params["jump_intensity"], params["jump_mean"],
                                params["jump_std"])


@pytest.mark.parametrize("args", [(0.0, T, 1, 1, 0), (S0, 0.0, 1, 1, 0), (S0, T, 0, 1, 0),
                                  (S0, T, 1, 0, 0), (S0, T, 1, 1, -1), (S0, T, 1, 1, True),
                                  (S0, T, 1.5, 1, 0), (NAN, T, 1, 1, 0)])
def test_rejects_invalid_base_arguments(args):
    with pytest.raises(ValueError):
        MertonJumpSimulator(*args, **MERTON)
    with pytest.raises(ValueError):
        KouJumpSimulator(*args, **KOU)


@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(batch_size=True),
                                    dict(max_working_bytes=1), dict(batch_size=1.5)])
def test_generate_terminal_rejects_bad_budget(kwargs):
    with pytest.raises(ValueError):
        _merton_sim(MERTON, 10, 0).generate_terminal(**kwargs)


def test_generate_paths_rejects_over_budget():
    sim = _merton_sim(MERTON, 10, 0, steps=4)
    sim.MAX_WORKING_BYTES = 8 * 5 * 10  # exactly the output, no room for scratch
    with pytest.raises(ValueError):
        sim.generate_paths()


@pytest.mark.parametrize("p_up, eta_up", [(0.0, 10.0), (1.0, 6.0), (0.0, 1.0001)])
def test_kou_boundary_probabilities_have_correct_mean(p_up, eta_up):
    # p in {0, 1} is accepted (one-signed jumps); N = 200_000, seed 71.
    params = dict(KOU, p_up=p_up, eta_up=eta_up)
    terminal = _kou_sim(params, 200_000, 71).generate_terminal()
    ref = _kou_ref(params)  # p = 0 or eta_up > 4: finite fourth moment
    z = (terminal.mean() - S0 * math.exp(KOU["mu"] * T)) / math.sqrt(ref["variance"] / terminal.size)
    assert abs(z) <= K_SE
    if p_up == 1.0:
        sims = _kou_sim(dict(params, sigma=0.0, mu=0.0), 1_000, 72).generate_terminal()
        # Only upward jumps: ln S_T >= ln S0 - lambda*k*T.
        k = _kou_jump_moment(1, 1.0, eta_up, KOU["eta_down"]) - 1
        assert np.all(np.log(sims / S0) >= -KOU["jump_intensity"] * k * T - 1e-12)


# --------------------------------------------------------------------------
# 6. Determinism, contracts and degenerate cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize("make, params", [(_merton_sim, MERTON), (_kou_sim, KOU)])
def test_same_seed_reproduces_and_different_seeds_differ(make, params):
    a = make(params, 300, 7, steps=20)
    b = make(params, 300, 7, steps=20)
    pa, pb = a.generate_paths(), b.generate_paths()
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(a.generate_terminal(), b.generate_terminal())
    assert not np.array_equal(pa, make(params, 300, 8, steps=20).generate_paths())
    # Repeated calls advance the instance's generators.
    assert not np.array_equal(pa, a.generate_paths())
    t1 = make(params, 300, 7).generate_terminal()
    t2 = make(params, 300, 7).generate_terminal()
    np.testing.assert_array_equal(t1, t2)
    assert not np.array_equal(t1, make(params, 300, 8).generate_terminal())


@pytest.mark.parametrize("make, params", [(_merton_sim, MERTON), (_kou_sim, KOU)])
@pytest.mark.parametrize("steps, n_sims", [(1, 1), (1, 7), (5, 1), (252, 3)])
def test_shape_contract_and_degenerate_sizes(make, params, steps, n_sims):
    paths = make(params, n_sims, 3, steps=steps).generate_paths()
    assert paths.shape == (steps + 1, n_sims) and paths.dtype == np.float64
    assert np.all(paths[0] == S0) and np.isfinite(paths).all() and np.all(paths > 0)
    terminal = make(params, n_sims, 3, steps=steps).generate_terminal()
    assert terminal.shape == (n_sims,) and terminal.dtype == np.float64


@pytest.mark.parametrize("make, params", [(_merton_sim, MERTON), (_kou_sim, KOU)])
def test_zero_sigma_zero_intensity_is_deterministic_growth(make, params):
    sim = make(dict(params, sigma=0.0, jump_intensity=0.0), 4, 9, steps=10)
    paths = sim.generate_paths()
    expected = S0 * np.exp(params["mu"] * np.linspace(0.0, T, 11))
    np.testing.assert_allclose(paths, np.repeat(expected[:, None], 4, axis=1), rtol=1e-13)
    np.testing.assert_allclose(make(dict(params, sigma=0.0, jump_intensity=0.0), 4, 9)
                               .generate_terminal(), S0 * math.exp(params["mu"] * T), rtol=1e-14)


@pytest.mark.parametrize("make, params", [(_merton_sim, MERTON), (_kou_sim, KOU)])
def test_very_small_horizon(make, params):
    # T = 1e-10 years: lambda*T = 3e-10 (jumps essentially never happen) and
    # sigma*sqrt(T) = 2e-6, so prices stay within ~1e-4 relative of S0.
    cls = MertonJumpSimulator if make is _merton_sim else KouJumpSimulator
    sim = cls(S0, 1e-10, 3, 1_000, 10, **params)
    paths = sim.generate_paths()
    assert paths.shape == (4, 1_000) and np.all(paths[0] == S0)
    np.testing.assert_allclose(paths, S0, rtol=1e-4)
    np.testing.assert_allclose(cls(S0, 1e-10, 3, 1_000, 10, **params).generate_terminal(),
                               S0, rtol=1e-4)


# --------------------------------------------------------------------------
# 7. Separate RNG streams (documented design: diffusion, counts, sizes)
# --------------------------------------------------------------------------
def _states(sim: Any) -> list[Any]:
    return [g.bit_generator.state for g in (sim.diffusion_rng, sim.count_rng, sim.size_rng)]


def test_streams_are_distinct_and_seed_dependent():
    a, b = _merton_sim(MERTON, 5, 1), _merton_sim(MERTON, 5, 2)
    draws_a = [g.random(4) for g in (a.diffusion_rng, a.count_rng, a.size_rng)]
    draws_b = [g.random(4) for g in (b.diffusion_rng, b.count_rng, b.size_rng)]
    for i in range(3):
        assert not np.array_equal(draws_a[i], draws_b[i])  # seed changes every stream
        for j in range(i + 1, 3):
            assert not np.array_equal(draws_a[i], draws_a[j])  # streams differ
    assert a.rng is a.diffusion_rng


@pytest.mark.parametrize("method", ["generate_paths", "generate_terminal"])
def test_zero_intensity_ignores_jump_law_and_jump_streams(method):
    # Docstring: with lambda = 0 neither the counts nor the sizes stream is
    # touched, and lambda*k = 0, so the jump law is irrelevant.
    base = _zero_intensity(MertonJumpSimulator, 200, 81, steps=10)
    before = _states(base)
    out = getattr(base, method)()
    after = _states(base)
    assert after[0] != before[0] and after[1] == before[1] and after[2] == before[2]
    other = _zero_intensity(MertonJumpSimulator, 200, 81, steps=10, jump_mean=0.3, jump_std=0.5)
    np.testing.assert_array_equal(out, getattr(other, method)())
    kou = _zero_intensity(KouJumpSimulator, 200, 81, steps=10)
    np.testing.assert_array_equal(out, getattr(kou, method)())


def test_zero_jump_std_draws_no_sizes():
    sim = _merton_sim(DETERMINISTIC, 500, 82, steps=12)
    before = _states(sim)
    sim.generate_paths()
    after = _states(sim)
    assert after[0] != before[0] and after[1] != before[1] and after[2] == before[2]


@pytest.mark.parametrize("method", ["generate_paths", "generate_terminal"])
def test_diffusion_stream_shared_across_intensities(method):
    # Separate streams: the same seed gives the same diffusion normals for
    # any lambda, so with delta = 0 the log ratio to the lambda = 0 run is
    # -lambda*k*t + N_t*m with integer N_t (coupling of the two runs).
    lam, m = DETERMINISTIC["jump_intensity"], DETERMINISTIC["jump_mean"]
    k = math.exp(m) - 1
    with_jumps = getattr(_merton_sim(DETERMINISTIC, 1_000, 83, steps=8), method)()
    without = getattr(_merton_sim(dict(DETERMINISTIC, jump_intensity=0.0), 1_000, 83, steps=8),
                      method)()
    times = np.linspace(0.0, T, 9)[:, None] if method == "generate_paths" else T
    implied = (np.log(with_jumps / without) + lam * k * times) / m
    np.testing.assert_allclose(implied, np.rint(implied), atol=1e-9)
    assert np.rint(implied).max() >= 1
