"""Regression: the QMC path memory check must include SciPy's Sobol scrambling.

``scipy.stats.qmc.Sobol(d, scramble=True, bits=30)`` builds ``d`` lower
triangular ``30 x 30`` scrambling matrices (about ``d * 30**2 * 8`` bytes,
~7.3 KB per dimension measured with tracemalloc, SciPy 1.18) before any point
is drawn. The former estimate ``output + 2*8*steps*N`` ignored it, so
``steps=21201, n_sims=1024, qmc_scrambles=8`` was accepted with an estimate of
207 MiB while the traced peak reached about 320 MiB (> 256 MiB limit).

The tests never allocate the large case: rejection must happen before the
output matrix or any Sobol sampler is created. Offline, seeded.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import simulation.gbm as gbm_module
from simulation.gbm import GbmSimulator
from simulation.variance_reduction import sobol_normals


def _model(steps: int, n_sims: int) -> GbmSimulator:
    return GbmSimulator(100.0, 1.0, steps, n_sims, seed=7, mu=0.07, sigma=0.3)


def test_reviewer_case_is_rejected_before_allocation(monkeypatch):
    def no_sobol(*args, **kwargs):  # pragma: no cover - reached only on failure
        raise AssertionError("Sobol sampler reached: the check came too late")

    monkeypatch.setattr(gbm_module, "sobol_normals", no_sobol)
    model = _model(21201, 1024)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="limit is"):
            model.generate_paths(variance_reduction="qmc", qmc_scrambles=8)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024 * 1024  # the bridge plan at most, no path matrix


def test_estimate_covers_traced_peak():
    # Sobol init dominates: 2000 dims * ~7.3 KB = ~14.6 MB versus a 4 MB output
    # and two 0.5 MB (N, steps) arrays. Warm up SciPy's one-time direction
    # number table so only per-call storage is traced.
    sobol_normals(1, 0, np.random.default_rng(0))
    steps, n_sims, scrambles = 2000, 256, 16
    model = _model(steps, n_sims)
    tracemalloc.start()
    try:
        paths = model.generate_paths(variance_reduction="qmc",
                                     qmc_scrambles=scrambles)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert paths.shape == (steps + 1, n_sims)
    del paths
    capped = _model(steps, n_sims)
    capped.MAX_WORKING_BYTES = peak - 1
    with pytest.raises(ValueError, match="limit is"):
        capped.generate_paths(variance_reduction="qmc", qmc_scrambles=scrambles)


def test_small_qmc_paths_remain_accepted():
    paths = _model(252, 2**16).generate_paths(variance_reduction="qmc",
                                              qmc_scrambles=16)
    assert paths.shape == (253, 2**16)
    assert np.all(paths[0] == 100.0)
