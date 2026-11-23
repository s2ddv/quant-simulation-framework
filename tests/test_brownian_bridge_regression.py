"""Regression: the Brownian bridge conditional sd must not underflow (P3-E2-2).

For the grid ``t_k = k*H/steps`` the bridge is linear, ``W = A Z``, and the
exact Brownian law requires ``A A^T = min(t_i, t_j)``. The conditional sd of a
midpoint is ``sqrt((t_m - t_l)(t_r - t_m)/(t_r - t_l))``; forming the product
``(t_m - t_l)(t_r - t_m)`` first underflows for tiny horizons (H = 1e-160 gave a
0.69% diagonal error, H = 1e-200 gave sd = 0). Time in years, W in year^1/2.

``A`` is obtained by applying the bridge to the identity (column j = image of
the unit vector e_j). Offline and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.variance_reduction import brownian_bridge, brownian_bridge_plan


def _bridge_matrix(steps: int, horizon: float) -> np.ndarray:
    return brownian_bridge(np.eye(steps), horizon)


@pytest.mark.parametrize("horizon", [1.0, 1e-160, 1e-200, 1e-300])
@pytest.mark.parametrize("steps", [7, 16])
def test_bridge_covariance_is_min_times(steps, horizon):
    a = _bridge_matrix(steps, horizon)
    times = np.arange(1, steps + 1, dtype=np.float64) * (horizon / steps)
    expected = np.minimum.outer(times, times)
    covariance = a @ a.T
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, expected, rtol=1e-12, atol=0.0)


def test_plan_sd_positive_for_tiny_horizon():
    plan = brownian_bridge_plan(16, 1e-200)
    assert all(entry[5] > 0 for entry in plan)
