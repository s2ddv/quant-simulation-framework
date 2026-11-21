"""Statistical validation of GBM / t-shock tail probabilities (offline, seeded)."""

import numpy as np
import pytest
from scipy import stats

from simulation.gbm import GbmSimulator
from validation.tail_probability import (clopper_pearson_interval,
                                         estimate_loss_probability,
                                         normal_loss_probability, wilson_interval)

MU, SIG, T = 0.05, 0.20, 5 / 252


def test_intervals_known_values_and_edges():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04          # Wilson upper for 0/100 ~ 0.0370
    assert clopper_pearson_interval(0, 100)[1] == pytest.approx(1 - 0.025 ** 0.01, rel=1e-9)
    assert clopper_pearson_interval(100, 100)[1] == 1.0
    w, c = wilson_interval(30, 1000), clopper_pearson_interval(30, 1000)
    assert c[0] <= w[0] and w[1] <= c[1] + 1e-12  # exact is wider
    for bad in ((1, 0), (5, 4), (-1, 3)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)


def test_gbm_terminal_matches_lognormal_ks_moments_quantiles():
    n = 200_000
    s = GbmSimulator(100.0, 1.0, 1, n, 11, mu=0.08, sigma=0.25)
    x = s.generate_paths("normal")[-1]
    mean_log, sd_log = (0.08 - 0.5 * 0.25**2), 0.25
    ks = stats.kstest(np.log(x / 100), stats.norm(mean_log, sd_log).cdf)
    assert ks.pvalue > 0.001                       # fixed seed: deterministic
    # mean of log-returns: finite-variance CLT standard error is valid here (normal)
    assert abs(np.mean(np.log(x / 100)) - mean_log) < 4 * sd_log / np.sqrt(n)
    for q in (0.01, 0.05, 0.5, 0.95, 0.99):        # quantile CI by order statistics
        theo = 100 * np.exp(mean_log + sd_log * stats.norm.ppf(q))
        assert abs(np.quantile(x, q) / theo - 1) < 0.02


def test_event_probability_matches_analytic_within_ci_normal():
    p = normal_loss_probability(0.05, MU, SIG, T)
    r = estimate_loss_probability(0.05, mu=MU, sigma=SIG, horizon=T, steps=5,
                                  n_sims=400_000, seed=1)
    lo, hi = r["wilson"]
    assert lo <= p <= hi


def test_ci_coverage_over_seeds_is_near_nominal():
    p = normal_loss_probability(0.03, MU, SIG, T)
    cover = 0
    reps = 60
    for seed in range(reps):
        r = estimate_loss_probability(0.03, mu=MU, sigma=SIG, horizon=T, steps=1,
                                      n_sims=20_000, seed=seed)
        cover += r["wilson"][0] <= p <= r["wilson"][1]
    assert cover / reps >= 0.85                    # nominal 0.95; loose for 60 reps


def test_convergence_half_width_scales_inverse_sqrt_n():
    widths = []
    for n in (10_000, 40_000, 160_000):
        r = estimate_loss_probability(0.03, mu=MU, sigma=SIG, horizon=T, steps=1,
                                      n_sims=n, seed=3)
        widths.append(r["wilson"][1] - r["wilson"][0])
    assert widths[0] / widths[1] == pytest.approx(2, rel=0.15)
    assert widths[1] / widths[2] == pytest.approx(2, rel=0.15)


def test_seed_determinism_and_sensitivity():
    kw = dict(mu=MU, sigma=SIG, horizon=T, steps=5, n_sims=50_000)
    a = estimate_loss_probability(0.03, seed=5, **kw)
    assert a == estimate_loss_probability(0.03, seed=5, **kw)
    assert a["hits"] != estimate_loss_probability(0.03, seed=6, **kw)["hits"]
    # batch layout does not change the sampled size
    assert estimate_loss_probability(0.03, seed=5, batch_size=7_000, **kw)["params"]["n_sims"] == 50_000


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
def test_normal_probability_increases_with_volatility(sigma):
    base = normal_loss_probability(0.05, MU, sigma, T)
    higher = normal_loss_probability(0.05, MU, sigma * 1.5, T)
    assert higher > base
    r = estimate_loss_probability(0.05, mu=MU, sigma=sigma, horizon=T, steps=5,
                                  n_sims=300_000, seed=9, level=0.999)
    # 99.9% level: a fixed-seed test should not fail on a ~5% chance event.
    assert r["clopper_pearson"][0] <= base <= r["clopper_pearson"][1]


def test_t_tail_is_heavier_for_extreme_loss_and_lower_df_larger_tail():
    kw = dict(mu=MU, sigma=SIG, horizon=T, steps=5, n_sims=400_000, seed=21)
    normal = estimate_loss_probability(0.086, **kw)
    t8 = estimate_loss_probability(0.086, dist_type="t_student", df=8, **kw)
    t4 = estimate_loss_probability(0.086, dist_type="t_student", df=4, **kw)
    # Compare through disjoint exact intervals: no moment-based SE on prices.
    assert normal["clopper_pearson"][1] < t8["clopper_pearson"][0]
    assert t8["probability"] < t4["probability"]
    # Unit-variance t(4) summed over 5 days is much lighter than a weekly t(4).
    weekly_t4 = stats.t.cdf((np.log(1 - 0.086) - (MU - 0.02) * T) / (SIG * np.sqrt(T))
                            * np.sqrt(2), 4)
    assert t4["probability"] < weekly_t4


def test_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_loss_probability(1.0, mu=0, sigma=0.2, horizon=T, steps=5, n_sims=10, seed=1)
    with pytest.raises(ValueError):
        normal_loss_probability(-0.1, 0, 0.2, T)
