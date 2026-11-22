"""Minimal smoke tests for the Merton and Kou jump-diffusion simulators (E4).

The tester extends these with law-level checks (KS, variance, quantiles).
"""

import math

import numpy as np
import pytest

from simulation.jump_diffusion import KouJumpSimulator, MertonJumpSimulator
from validation.jump_theory import (
    kou_terminal_moments,
    merton_terminal_moments,
    merton_terminal_quantile,
)
from validation.statistics import terminal_quantile

MERTON = dict(mu=0.08, sigma=0.2, jump_intensity=3.0, jump_mean=-0.05, jump_std=0.1)
KOU = dict(mu=0.08, sigma=0.2, jump_intensity=3.0, p_up=0.4, eta_up=10.0, eta_down=5.0)


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
