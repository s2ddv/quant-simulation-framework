"""Regression: a fixed beta whose residual overflows must raise ValueError.

With ``beta`` fixed, ``beta * X`` can overflow to +-inf even though the
rescaled ``beta`` is finite; ``np.ptp`` of such a residual is ``nan`` and the
estimator used to return ``standard_error = 0.0`` and ``rho2 = 1.0`` instead of
the documented ``ValueError`` (offline, seeded).
"""

import numpy as np
import pytest

from simulation.variance_reduction import control_variate_mean


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    control = 1.9 + 0.04 * rng.random(100)
    values = 1.5 * rng.random(100)
    return values, control


@pytest.mark.parametrize("beta", [1.7e308, -1.7e308, 1e300])
def test_fixed_beta_overflowing_residual_raises(beta: float) -> None:
    values, control = _data()
    with pytest.raises(ValueError, match="finite precision"):
        control_variate_mean(values, control, float(np.mean(control)),
                             beta=beta)


def test_fixed_moderate_beta_still_finite() -> None:
    values, control = _data()
    result = control_variate_mean(values, control, float(np.mean(control)),
                                  beta=1.7)
    assert np.isfinite(result["estimate"])
    assert result["standard_error"] > 0
