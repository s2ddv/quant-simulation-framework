"""Offline checks of the Merton jump calibration (E4.3); synthetic data only."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, poisson

from calibration.estimation import TRADING_DAYS
from calibration.jumps import (
    PARAM_NAMES,
    fit_merton_jumps,
    fit_merton_jumps_returns,
    merton_daily_loglik,
)

DT = 1.0 / TRADING_DAYS
TRUE = {"mu": 0.08, "sigma": 0.2, "jump_intensity": 10.0,
        "jump_mean": -0.03, "jump_std": 0.04}


def _merton_returns(n, mu, sigma, lam, m_j, delta, seed):
    """Exact daily log-returns from the Poisson-normal mixture (numpy only)."""
    rng = np.random.default_rng(seed)
    k = math.expm1(m_j + 0.5 * delta**2)
    a = mu - 0.5 * sigma**2 - lam * k
    counts = rng.poisson(lam * DT, n)
    return (a * DT + sigma * math.sqrt(DT) * rng.standard_normal(n)
            + counts * m_j + delta * np.sqrt(counts) * rng.standard_normal(n))


@pytest.fixture(scope="module")
def synthetic():
    x = _merton_returns(5000, *TRUE.values(), seed=12345)
    return x, fit_merton_jumps_returns(x, seed=0)


def test_recovers_parameters_within_four_fisher_standard_errors(synthetic):
    _, fit = synthetic
    assert fit["converged"] and fit["n_obs"] == 5000
    assert fit["std_errors"] is not None
    for name in PARAM_NAMES:
        se = fit["std_errors"][name]
        assert se is not None and se > 0.0
        z = (fit[name] - TRUE[name]) / se
        assert abs(z) <= 4.0, (name, fit[name], se, z)


def test_aic_and_bic_prefer_merton_on_jump_data(synthetic):
    _, fit = synthetic
    assert fit["aic"] < fit["gbm_aic"]
    assert fit["bic"] < fit["gbm_bic"]
    assert fit["nll"] < fit["gbm_nll"]


def test_result_is_json_safe_and_deterministic(synthetic):
    x, fit = synthetic
    json.dumps(fit, allow_nan=False)
    again = fit_merton_jumps_returns(x, seed=0)
    assert again == fit


def test_loglik_matches_independent_mixture_density(synthetic):
    x, _ = synthetic
    params = (0.05, 0.18, 8.0, -0.02, 0.05)
    mu, sigma, lam, m_j, delta = params
    a = mu - 0.5 * sigma**2 - lam * math.expm1(m_j + 0.5 * delta**2)
    dens = sum(poisson.pmf(n, lam * DT)
               * norm.pdf(x, a * DT + n * m_j, math.sqrt(sigma**2 * DT + n * delta**2))
               for n in range(40))
    assert merton_daily_loglik(params, x) == pytest.approx(np.log(dens).sum(), rel=1e-12)


def test_prices_entry_point_matches_returns():
    x = _merton_returns(400, *TRUE.values(), seed=7)
    prices = pd.Series(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(x)])))
    by_prices = fit_merton_jumps(prices, n_starts=3, seed=1)
    by_returns = fit_merton_jumps_returns(np.diff(np.log(prices.to_numpy())),
                                          n_starts=3, seed=1)
    for name in PARAM_NAMES + ("nll",):
        assert by_prices[name] == pytest.approx(by_returns[name], rel=1e-9, abs=1e-12)


def test_pure_diffusion_data_is_flagged():
    x = _merton_returns(3000, 0.08, 0.2, 0.0, 0.0, 0.04, seed=3)
    fit = fit_merton_jumps_returns(x, n_starts=4, seed=0)
    # Jumps are not supported by the data: the fit sits on a boundary or the
    # information criteria do not prefer it; warnings must say so.
    assert fit["warnings"]
    assert fit["bic"] > fit["gbm_bic"]


@pytest.mark.parametrize("bad", [
    np.array([0.01]), np.array([0.01, np.nan, 0.0]), np.zeros(10),
    np.array([True, False, True]), np.ones((3, 2)),
])
def test_invalid_returns_rejected(bad):
    with pytest.raises(ValueError):
        fit_merton_jumps_returns(bad)


@pytest.mark.parametrize("kwargs", [{"n_starts": 0}, {"seed": -1}, {"seed": True},
                                    {"dt": 0.0}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_merton_jumps_returns(np.array([0.01, -0.02, 0.005]), **kwargs)


def test_prices_validation():
    with pytest.raises(ValueError):
        fit_merton_jumps(pd.Series([100.0, 101.0]))
    with pytest.raises(ValueError):
        fit_merton_jumps(pd.Series([100.0, -1.0, 101.0]))
