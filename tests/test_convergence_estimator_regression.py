"""Regressions from the E2.4 review (offline, deterministic).

(1) ``mean_estimator`` on large finite values: the plain ``np.mean``/``np.std``
overflow to ``inf`` although the mean (1.4e308) and its standard error are
representable. A finite result is required, or a ``ValueError`` when the
result itself is not representable.
(2) ``convergence_with_estimator(..., counts=[True, 10])``: ``np.asarray``
turned the mixed list into integers ``[1, 10]``; bool elements must be rejected
as bool scalars already are.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from validation.statistics import convergence_with_estimator, mean_estimator


def test_mean_estimator_rescales_large_finite_values():
    values = np.array([1e308, 1.7e308, 1.5e308])
    result = mean_estimator(values)
    exact = [Fraction(v) for v in values]
    mean = sum(exact) / 3
    variance = sum((v - mean) ** 2 for v in exact) / 2
    expected_se = math.sqrt(float(variance / Fraction(10) ** 600)) * 1e300 / math.sqrt(3)
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["standard_error"])
    assert result["estimate"] == pytest.approx(float(mean), rel=1e-14)
    assert result["standard_error"] == pytest.approx(expected_se, rel=1e-12)


def test_mean_estimator_wide_spread_stays_finite():
    # The difference 3.4e308 and std(ddof=1) = 2.4e308 overflow, yet the
    # mean (0) and SE = |a - b|/2 = 1.7e308 are representable.
    result = mean_estimator(np.array([-1.7e308, 1.7e308]))
    assert result["estimate"] == 0.0
    assert result["standard_error"] == pytest.approx(1.7e308, rel=1e-14)


def test_mean_estimator_unchanged_for_ordinary_values():
    values = np.arange(10.0)
    result = mean_estimator(values)
    assert result["estimate"] == float(np.mean(values))
    assert result["standard_error"] == float(np.std(values, ddof=1)) / math.sqrt(10)


@pytest.mark.parametrize("counts", [[True, 10], [np.bool_(True), 10], (2, True, 10)])
def test_counts_with_bool_elements_are_rejected(counts):
    with pytest.raises(ValueError):
        convergence_with_estimator(np.arange(10.0), mean_estimator, counts=counts)


def test_integer_counts_still_accepted():
    checkpoints = convergence_with_estimator(np.arange(10.0), mean_estimator,
                                             counts=np.array([2, 5, 10]))
    assert [c["n_units"] for c in checkpoints] == [2, 5, 10]
