"""Monte Carlo confidence intervals for VaR, CVaR and probability of loss.

Coverage checks use exact lognormal GBM terminal draws (no path simulator, so
these tests need only NumPy) and the analytical risk metrics, with
L = s0 - S_T and ln(S_T/s0) ~ N((mu - sigma^2/2) T, sigma^2 T):

    VaR_a  = s0 - s0 exp(m + s z_{1-a})
    CVaR_a = s0 - s0 exp(mu T) Phi(z_{1-a} - s) / (1 - a)
    P(L>0) = Phi(-m / s),      m = (mu - sigma^2/2) T,  s = sigma sqrt(T).

Coverage tolerance: with R independent replications the empirical coverage of
a nominal 95% interval has standard error sqrt(0.95*0.05/R); thresholds sit
about 3 such errors (plus any documented method bias) below nominal. Seeds are
fixed, so results are deterministic for a given NumPy version.
"""

import unittest
from statistics import NormalDist

import numpy as np

from risk.intervals import (
    _binomial_order_indices,
    cvar_bootstrap_interval,
    prob_loss_wilson_interval,
    risk_confidence_intervals,
    var_confidence_interval,
)
from risk.statistics import risk_summary

S0, MU, SIGMA, T = 100.0, 0.08, 0.25, 1.0
PHI = NormalDist()


def analytical_risk(confidence):
    m, s = (MU - SIGMA**2 / 2) * T, SIGMA * np.sqrt(T)
    z = PHI.inv_cdf(1 - confidence)
    var = S0 - S0 * np.exp(m + s * z)
    cvar = S0 - S0 * np.exp(MU * T) * PHI.cdf(z - s) / (1 - confidence)
    return var, cvar, PHI.cdf(-m / s)


def terminal_losses(rng, n):
    m, s = (MU - SIGMA**2 / 2) * T, SIGMA * np.sqrt(T)
    return S0 - S0 * np.exp(m + s * rng.standard_normal(n))


def coverage(intervals, truth):
    hits = [lo is not None and up is not None and lo <= truth <= up
            for lo, up in intervals]
    return float(np.mean(hits))


class OrderStatisticTests(unittest.TestCase):
    def test_indices_match_brute_force_binomial(self):
        for n, q in ((20, 0.95), (50, 0.5), (137, 0.9), (400, 0.99)):
            alpha = 0.05
            k = np.arange(n + 1)
            log_pmf = np.array([
                np.sum(np.log(np.arange(1, n + 1)))
                - np.sum(np.log(np.arange(1, i + 1)))
                - np.sum(np.log(np.arange(1, n - i + 1)))
                + i * np.log(q) + (n - i) * np.log1p(-q) for i in k])
            cdf = np.cumsum(np.exp(log_pmf))
            below = k[cdf <= alpha / 2]
            above = k[(cdf >= 1 - alpha / 2) & (k <= n - 1)]
            lower, upper, achieved = _binomial_order_indices(n, q, alpha)
            self.assertEqual(lower, int(below[-1]) if below.size else None)
            self.assertEqual(upper, int(above[0]) if above.size else None)
            self.assertGreaterEqual(achieved, 1 - alpha - 1e-12)

    def test_small_n_has_unbounded_side_and_reports_coverage(self):
        result = var_confidence_interval(np.arange(20.0), 0.95)
        self.assertIsNone(result["upper"])
        self.assertEqual(result["lower_order_statistic"], 17)
        self.assertEqual(result["lower"], 16.0)
        self.assertGreaterEqual(result["achieved_coverage"], 0.95)
        single = var_confidence_interval([3.0], 0.95)
        self.assertIsNone(single["lower"])
        self.assertIsNone(single["upper"])
        self.assertEqual(single["achieved_coverage"], 1.0)

    def test_large_n_is_cheap_and_close_to_nominal(self):
        lower, upper, achieved = _binomial_order_indices(10**8, 0.95, 0.05)
        self.assertLess(lower, 0.95 * 10**8)
        self.assertLess(0.95 * 10**8, upper)
        # Discreteness makes the excess tiny for huge n.
        self.assertAlmostEqual(achieved, 0.95, delta=1e-3)

    def test_var_interval_covers_analytical_lognormal_var(self):
        confidence = 0.95
        var_true, _, _ = analytical_risk(confidence)
        rng = np.random.default_rng(20261120)
        intervals = []
        for _ in range(500):
            result = var_confidence_interval(terminal_losses(rng, 1000),
                                             confidence)
            intervals.append((result["lower"], result["upper"]))
            self.assertGreaterEqual(result["achieved_coverage"], 0.95)
        # Exact distribution-free coverage >= 0.95; SE(R=500) ~ 0.0097.
        self.assertGreaterEqual(coverage(intervals, var_true), 0.92)

    def test_negative_var_bounds_are_preserved(self):
        losses = -np.linspace(10.0, 50.0, 200)  # every path gains
        result = var_confidence_interval(losses, 0.95)
        self.assertLess(result["upper"], 0)
        self.assertLessEqual(result["lower"], result["upper"])


class CvarBootstrapTests(unittest.TestCase):
    def test_warns_when_tail_is_small(self):
        rng = np.random.default_rng(3)
        small = cvar_bootstrap_interval(rng.normal(size=100), 0.99,
                                        n_bootstrap=50, seed=1)
        self.assertAlmostEqual(small["tail_points"], 1.0)
        self.assertIn("warning", small)
        large = cvar_bootstrap_interval(rng.normal(size=2000), 0.95,
                                        n_bootstrap=50, seed=1)
        self.assertNotIn("warning", large)

    def test_reproducible_with_seed_and_contains_estimate(self):
        rng = np.random.default_rng(7)
        losses = terminal_losses(rng, 3000)
        first = cvar_bootstrap_interval(losses, n_bootstrap=300, seed=11)
        second = cvar_bootstrap_interval(losses, n_bootstrap=300, seed=11)
        self.assertEqual(first, second)
        estimate = risk_summary(np.vstack([np.full(3000, S0), S0 - losses]))["cvar"]
        self.assertLess(first["lower"], estimate)
        self.assertLess(estimate, first["upper"])

    def test_bootstrap_chunking_does_not_change_replicates(self):
        import risk.intervals as module
        losses = terminal_losses(np.random.default_rng(3), 500)
        reference = cvar_bootstrap_interval(losses, n_bootstrap=50, seed=5)
        original = module._BOOTSTRAP_CHUNK_ELEMENTS
        try:
            module._BOOTSTRAP_CHUNK_ELEMENTS = 1  # one resample per chunk
            chunked = cvar_bootstrap_interval(losses, n_bootstrap=50, seed=5)
        finally:
            module._BOOTSTRAP_CHUNK_ELEMENTS = original
        self.assertEqual(reference, chunked)

    def test_cvar_interval_covers_analytical_lognormal_cvar(self):
        confidence = 0.95
        _, cvar_true, _ = analytical_risk(confidence)
        rng = np.random.default_rng(424242)
        intervals = []
        for rep in range(200):
            result = cvar_bootstrap_interval(terminal_losses(rng, 2000),
                                             confidence, n_bootstrap=400,
                                             seed=rep)
            intervals.append((result["lower"], result["upper"]))
        # Percentile bootstrap with ~100 tail points is only asymptotically
        # valid and slightly anti-conservative; SE(R=200) ~ 0.015, so allow
        # ~3 SE plus a few points of known small-tail undercoverage.
        self.assertGreaterEqual(coverage(intervals, cvar_true), 0.87)

    def test_degenerate_and_tiny_samples(self):
        constant = cvar_bootstrap_interval(np.full(10, -5.0))
        self.assertEqual((constant["lower"], constant["upper"]), (-5.0, -5.0))
        zeros = cvar_bootstrap_interval(np.zeros(10))
        self.assertEqual((zeros["lower"], zeros["upper"]), (0.0, 0.0))
        single = cvar_bootstrap_interval([1.0])
        self.assertIsNone(single["lower"])
        self.assertIsNone(single["upper"])

    def test_huge_finite_losses_do_not_overflow(self):
        losses = np.array([-1e308, 1e308, 1.7e308, -1.7e308, 0.0, 1e300])
        result = cvar_bootstrap_interval(losses, 0.5, n_bootstrap=50)
        self.assertTrue(np.isfinite([result["lower"], result["upper"]]).all())


class WilsonTests(unittest.TestCase):
    def test_matches_closed_form_and_boundaries(self):
        result = prob_loss_wilson_interval(30, 100)
        z = PHI.inv_cdf(0.975)
        d = 1 + z**2 / 100
        center = (0.3 + z**2 / 200) / d
        half = z * np.sqrt(0.3 * 0.7 / 100 + z**2 / 40000) / d
        self.assertAlmostEqual(result["lower"], center - half)
        self.assertAlmostEqual(result["upper"], center + half)
        none = prob_loss_wilson_interval(0, 50)
        self.assertEqual(none["lower"], 0.0)
        self.assertGreater(none["upper"], 0.0)
        every = prob_loss_wilson_interval(50, 50)
        self.assertEqual(every["upper"], 1.0)
        self.assertLess(every["lower"], 1.0)

    def test_prob_loss_interval_covers_analytical_probability(self):
        _, _, p_true = analytical_risk(0.95)
        rng = np.random.default_rng(99)
        intervals = []
        for _ in range(1000):
            k = int(np.count_nonzero(terminal_losses(rng, 500) > 0))
            result = prob_loss_wilson_interval(k, 500)
            intervals.append((result["lower"], result["upper"]))
        # Wilson is near nominal at n=500; SE(R=1000) ~ 0.0069.
        self.assertGreaterEqual(coverage(intervals, p_true), 0.93)


class CombinedAndValidationTests(unittest.TestCase):
    def test_estimates_match_risk_summary_and_bracket(self):
        rng = np.random.default_rng(2026)
        terminal = S0 - terminal_losses(rng, 5000)
        paths = np.vstack([np.full(5000, S0), terminal])
        summary = risk_summary(paths)
        result = risk_confidence_intervals(paths, n_bootstrap=200, seed=1)
        self.assertEqual(result["var"]["estimate"], summary["var"])
        self.assertEqual(result["cvar"]["estimate"], summary["cvar"])
        self.assertEqual(result["prob_loss"]["estimate"], summary["prob_loss"])
        for key in ("var", "cvar", "prob_loss"):
            entry = result[key]
            self.assertLessEqual(entry["lower"], entry["estimate"])
            self.assertLessEqual(entry["estimate"], entry["upper"])
            self.assertIn("monte_carlo_error_only", entry["parameter_scope"])

    def test_cvar_ties_at_threshold_and_strict_loss(self):
        paths = np.array([[100] * 5, [80, 90, 90, 90, 110]])
        result = risk_confidence_intervals(paths, 0.5, n_bootstrap=20)
        self.assertEqual(result["cvar"]["estimate"], 12.5)
        flat = risk_confidence_intervals([[100] * 4, [100] * 4], n_bootstrap=20)
        self.assertEqual(flat["prob_loss"]["estimate"], 0.0)
        self.assertEqual(flat["var"]["estimate"], 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            var_confidence_interval([], 0.95)
        with self.assertRaises(ValueError):
            var_confidence_interval([1.0, np.nan], 0.95)
        with self.assertRaises(ValueError):
            var_confidence_interval([1.0, 2.0], 1.0)
        with self.assertRaises(ValueError):
            var_confidence_interval([1.0, 2.0], 0.95, level=0)
        with self.assertRaises(ValueError):
            cvar_bootstrap_interval([1.0, 2.0], n_bootstrap=1)
        with self.assertRaises(ValueError):
            cvar_bootstrap_interval([1.0, 2.0], n_bootstrap=True)
        with self.assertRaises(ValueError):
            prob_loss_wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            prob_loss_wilson_interval(1.5, 2)
        with self.assertRaises(ValueError):
            risk_confidence_intervals([[100, 101], [90, 95]])


if __name__ == "__main__":
    unittest.main()

