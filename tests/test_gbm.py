"""Offline checks for the public GBM and discrete t scenario contracts."""

from contextlib import contextmanager
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from scipy.stats import norm

from simulation.base import BaseStochasticSimulator
from simulation.gbm import GbmSimulator, simulate_gbm


@contextmanager
def raises_value_error():
    try:
        yield
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_zero_volatility_follows_exact_deterministic_solution():
    paths = simulate_gbm(100, 0.08, 0, 1, 4, 3, seed=42)
    expected = 100 * np.exp(0.08 * np.arange(5) / 4)
    assert paths.shape == (5, 3)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape))


def test_same_seed_reproduces_paths():
    first = simulate_gbm(100, 0.08, 0.2, 1, 4, 5, seed=42)
    second = simulate_gbm(100, 0.08, 0.2, 1, 4, 5, seed=42)
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)


def test_base_cannot_be_instantiated_and_validates_dimensions():
    try:
        BaseStochasticSimulator(100, 1, 4, 5)
        assert False, "abstract class should not instantiate"
    except TypeError:
        pass
    with raises_value_error():
        GbmSimulator(100, 1, 0, 5)


def test_constructor_rejects_invalid_parameters():
    bad_cases = [
        (0, 1, 4, 5, None), (np.inf, 1, 4, 5, None),
        (100, 0, 4, 5, None), (100, np.nan, 4, 5, None),
        (100, 1, 0, 5, None), (100, 1, 2.5, 5, None),
        (100, 1, 4, 0, None), (100, 1, 4, True, None),
        (100, 1, 4, 5, -1), (100, 1, 4, 5, 1.2),
        (100, np.nextafter(0.0, 1.0), 2, 5, None),
    ]
    for args in bad_cases:
        with raises_value_error():
            GbmSimulator(*args)
    with raises_value_error():
        GbmSimulator(100, 1, 4, 5, mu=0.1)
    with raises_value_error():
        GbmSimulator(100, 1, 4, 5, mu=0.1, sigma=-0.2)


def test_normal_matches_legacy_rng_convention_and_analytic_terminal_law():
    s0, mu, sigma, horizon, steps, n_sims, seed = 100, 0.07, 0.18, 1.2, 6, 60_000, 19
    paths = simulate_gbm(s0, mu, sigma, horizon, steps, n_sims, seed)
    # Independent construction using the public NumPy RNG convention. It also
    # protects the existing step-major draw order used by saved normal replays.
    z = np.random.default_rng(seed).standard_normal((steps, n_sims))
    expected = s0 * np.exp(np.cumsum(
        (mu - sigma**2 / 2) * horizon / steps
        + sigma * np.sqrt(horizon / steps) * z, axis=0))
    np.testing.assert_allclose(paths[1:], expected, rtol=2e-15, atol=1e-12)
    np.testing.assert_array_equal(paths[0], np.full(n_sims, s0))

    terminal = paths[-1]
    mean = s0 * np.exp(mu * horizon)
    variance = s0**2 * np.exp(2 * mu * horizon) * np.expm1(sigma**2 * horizon)
    # 60k draws: mean SE ~= 0.08 price units; 1% relative tolerance is
    # over 10 SE. Variance and tail quantiles fluctuate more, so use 4%.
    assert abs(terminal.mean() / mean - 1) < 0.01
    assert abs(terminal.var(ddof=1) / variance - 1) < 0.04
    theoretical = s0 * np.exp((mu - sigma**2 / 2) * horizon
                              + sigma * np.sqrt(horizon) * norm.ppf([0.05, 0.5, 0.95]))
    np.testing.assert_allclose(np.quantile(terminal, [0.05, 0.5, 0.95]),
                               theoretical, rtol=0.04)


def test_rng_state_advances_on_repeated_generation():
    model = GbmSimulator(100, 1, 3, 6, seed=71, mu=0.04, sigma=0.3)
    first, second = model.generate_paths(), model.generate_paths()
    assert not np.array_equal(first, second)
    replica = GbmSimulator(100, 1, 3, 6, seed=71, mu=0.04, sigma=0.3)
    np.testing.assert_array_equal(first, replica.generate_paths())
    np.testing.assert_array_equal(second, replica.generate_paths())


def test_calibration_matches_classical_moments_and_t_paths():
    rng = np.random.default_rng(7)
    returns = 0.0003 + 0.012 * rng.standard_t(5, size=700)
    prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]))
    model = GbmSimulator(float(prices.iloc[-1]), 1, 8, 2000, seed=23)
    model.calibrate(prices)

    expected_sigma = float(returns.std(ddof=1) * np.sqrt(252))
    expected_mu = float(returns.mean() * 252 + expected_sigma**2 / 2)
    np.testing.assert_allclose([model.mu, model.sigma], [expected_mu, expected_sigma])
    assert model.calibrated_df is not None and model.calibrated_df > 2
    first = model.generate_paths("t_student")
    duplicate = GbmSimulator(float(prices.iloc[-1]), 1, 8, 2000, seed=23)
    duplicate.calibrate(prices)
    np.testing.assert_array_equal(first, duplicate.generate_paths("t_student"))
    assert first.shape == (9, 2000)
    assert np.all(first[0] == model.s0)
    assert np.all(first > 0)


def test_t_shock_scale_and_invalid_mode():
    model = GbmSimulator(100, 1 / 252, 1, 80_000, seed=11, mu=0.05, sigma=0.2)
    model.calibrated_df = 5.0
    paths = model.generate_paths("t_student")
    increments = np.log(paths[-1] / 100)
    expected_std = 0.2 / np.sqrt(252)
    assert abs(increments.std() / expected_std - 1) < 0.03
    with raises_value_error():
        model.generate_paths("other")


def test_t_shocks_have_unit_variance_and_long_tails():
    model = GbmSimulator(100, 1 / 252, 1, 100_000, seed=108,
                         mu=0.05, sigma=0.2)
    model.calibrated_df = 5.0
    log_returns = np.log(model.generate_paths("t_student")[-1] / 100)
    shocks = (log_returns - (model.mu - model.sigma**2 / 2) * model.dt)
    shocks /= model.sigma * np.sqrt(model.dt)
    # For df=5, finite fourth moment but heavy tails. 100k samples make
    # a 3% SD band generous; exceedance at 3 SD is far above Gaussian 0.27%.
    assert abs(shocks.mean()) < 0.015
    assert abs(shocks.std(ddof=1) - 1) < 0.03
    assert np.mean(np.abs(shocks) > 3) > 0.007


def test_degenerate_calibration_and_t_df_boundary():
    model = GbmSimulator(100, 1, 4, 3, seed=42)
    with raises_value_error():
        model.calibrate(pd.Series([100, 101]))
    with raises_value_error():
        model.calibrate(pd.Series([100, 0, 101]))
    with raises_value_error():
        model.calibrate(pd.Series([100, np.nan, 101]))
    with raises_value_error():
        model.generate_paths()
    model.calibrate(pd.Series([100.0, 100.0, 100.0, 100.0]))
    assert model.mu == model.sigma == 0
    assert model.calibrated_df is None
    np.testing.assert_array_equal(model.generate_paths(), np.full((5, 3), 100.0))
    with raises_value_error():
        model.generate_paths("t_student")
    model.calibrated_df = 2.0
    with raises_value_error():
        model.generate_paths("t_student")


def test_calibration_rejects_reversed_dates_and_missing_close():
    model = GbmSimulator(100, 1, 4, 3, seed=42)
    dates = pd.date_range("2026-01-01", periods=4)
    reversed_history = pd.Series([100., 101., 102., 103.], index=dates[::-1])
    missing_history = pd.Series([100., 101., np.nan, 103.], index=dates)
    for prices in (reversed_history, missing_history):
        with raises_value_error():
            model.calibrate(prices)


def test_memory_limit_rejects_before_large_allocation():
    model = GbmSimulator(100, 1, 16, 16, seed=1, mu=0.1, sigma=0.2)
    model.MAX_WORKING_BYTES = 2_000  # normal output: 2,176 bytes
    with raises_value_error():
        model.generate_paths()
    model.MAX_WORKING_BYTES = 3_000  # normal fits, t needs an extra 2,048 bytes
    assert model.generate_paths().shape == (17, 16)
    model.calibrated_df = 5.0
    with raises_value_error():
        model.generate_paths("t_student")


def test_extreme_finite_parameters_keep_finite_increments():
    # sigma**2 and dt*(df-2) overflow separately in naive arithmetic, yet
    # the per-step volatility and standardized t factor are representable.
    model = GbmSimulator(100, 1e300, 1, 12, seed=14, mu=0, sigma=1e-150)
    model.calibrated_df = 1e20
    paths = model.generate_paths("t_student")
    assert paths.shape == (2, 12)
    assert np.isfinite(paths).all() and np.all(paths > 0)
    np.testing.assert_array_equal(paths[0], np.full(12, 100.0))

    # The annual volatility itself can overflow when squared, although its
    # product with sqrt(dt) and the per-step drift remain finite.
    tiny_horizon = GbmSimulator(100, 1e-320, 1, 12, seed=14,
                                mu=0, sigma=1e160)
    tiny_paths = tiny_horizon.generate_paths()
    assert np.isfinite(tiny_paths).all() and np.all(tiny_paths > 0)


def test_unrepresentable_increments_fail_before_rng_advances():
    model = GbmSimulator(100, 1, 2, 5, seed=29, mu=0, sigma=1e200)
    before = model.rng.bit_generator.state
    with raises_value_error():
        model.generate_paths()
    assert model.rng.bit_generator.state == before


def test_invalid_t_degrees_of_freedom_fail_before_rng_advances():
    model = GbmSimulator(100, 1, 2, 5, seed=31, mu=0, sigma=0.2)
    before = model.rng.bit_generator.state
    for df in (None, -1, 0, 2, np.nan, np.inf, True, "5"):
        model.calibrated_df = df
        with raises_value_error():
            model.generate_paths("t_student")
        assert model.rng.bit_generator.state == before


def test_extreme_initial_price_and_drift_preserve_finite_price():
    with localcontext() as context:
        context.prec = 80
        large_growth = float(Decimal("1e-300") * Decimal(800).exp())
        large_decline = float(Decimal("1e300") * Decimal(-800).exp())

    for initial, drift, expected in (
        (1e-300, 800, large_growth),
        (1e300, -800, large_decline),
    ):
        paths = simulate_gbm(initial, drift, 0, 1, 1, 1, seed=0)
        np.testing.assert_array_equal(paths[0], [initial])
        np.testing.assert_allclose(paths[1], [expected], rtol=1e-12)


def test_explicit_df_standardized_unit_variance_and_validation():
    import pytest
    from simulation.gbm import GbmSimulator
    model = GbmSimulator(100.0, 1.0, 1, 400_000, 7, mu=0.0, sigma=0.2, df=6)
    r = np.log(model.generate_paths("t_student")[-1] / 100.0)
    # Unit-variance t shocks: Var[log return] = sigma^2 * T = 0.04 (finite 2nd moment).
    assert abs(np.var(r) - 0.04) < 0.002
    for bad in (2, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            GbmSimulator(100.0, 1.0, 1, 10, 1, mu=0.0, sigma=0.2, df=bad)
