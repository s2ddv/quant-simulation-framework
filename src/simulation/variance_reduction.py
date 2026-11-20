"""Pair-aware Monte Carlo estimators for antithetic GBM samples.

Antithetic layout contract (shared with ``GbmSimulator``): for an even
``n_sims = 2*m``, column/element ``i`` uses shocks ``Z_i`` and element
``i + m`` uses ``-Z_i``, for ``i = 0, ..., m - 1``. The two halves are
therefore *not* independent samples. Classical ``std(x)/sqrt(n)`` treats
all ``2m`` values as i.i.d. and misstates the Monte Carlo error; the
correct estimator uses the ``m`` i.i.d. pair averages
``Y_i = (f(Z_i) + f(-Z_i)) / 2``:

    mean  = mean(Y)
    SE    = std(Y, ddof=1) / sqrt(m)
    CI    = mean +/- z_{(1+c)/2} * SE        (normal approximation)

The interval describes Monte Carlo error of the estimator of the mean only;
it does not cover calibration uncertainty or model error.
"""

from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class MeanEstimate:
    """Monte Carlo estimate of a mean with its standard error and CI.

    ``n_independent`` is the number of i.i.d. units used for the standard
    error: pairs for antithetic samples, draws for plain samples.
    """

    mean: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    n_independent: int


def _validate(values: np.ndarray, confidence: float) -> np.ndarray:
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not 0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return array


def _estimate(units: np.ndarray, confidence: float) -> MeanEstimate:
    if units.size < 2:
        raise ValueError("at least two independent units are required")
    mean = float(np.mean(units))
    se = float(np.std(units, ddof=1) / np.sqrt(units.size))
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    return MeanEstimate(mean, se, mean - z * se, mean + z * se,
                        float(confidence), int(units.size))


def plain_mean_estimate(values: np.ndarray, confidence: float = 0.95) -> MeanEstimate:
    """Mean, SE and normal CI assuming i.i.d. ``values`` (plain Monte Carlo)."""
    return _estimate(_validate(values, confidence), confidence)


def antithetic_pair_means(values: np.ndarray) -> np.ndarray:
    """Return the ``m`` pair averages ``(x[i] + x[i+m]) / 2`` of a 2m-vector.

    Odd lengths raise ``ValueError``: an antithetic sample is always even.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or array.size % 2:
        raise ValueError("antithetic values must be a nonempty even-length 1-D array")
    half = array.size // 2
    return 0.5 * (array[:half] + array[half:])


def antithetic_mean_estimate(values: np.ndarray,
                             confidence: float = 0.95) -> MeanEstimate:
    """Mean, pair-based SE and normal CI for antithetic samples.

    ``values`` must follow the half-split layout above, e.g. terminal prices
    from ``generate_terminal(antithetic=True)`` or ``paths[-1]`` from
    ``generate_paths(antithetic=True)``. Any function applied elementwise
    (e.g. losses ``s0 - terminal``) keeps the layout.
    """
    array = _validate(values, confidence)
    return _estimate(antithetic_pair_means(array), confidence)
