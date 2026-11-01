"""Conditional uncertainty of observed losses using a moving-block bootstrap.

Following the overlapping-block construction of Kunsch (1989),
doi:10.1214/aos/1176347265, sample starts uniformly from 0,...,n-L.
Draw ceil(n/L) noncircular blocks of fixed length L, concatenate, and truncate
the last block to n observations. This explicit finite-sample edge convention
gives interior observations more inclusion weight than the endpoints.
Percentile intervals concern the mean of the supplied series, in its units.
They are pointwise and assume weak temporal dependence and approximate
stationarity. No model fitting or parameter selection is repeated here.
"""

import numpy as np


def _integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def block_indices(n, block_length, replicates, *, rng):
    """Return a (replicates, n) array of shared, contiguous-block indices."""
    for name, value in (("n", n), ("block_length", block_length),
                        ("replicates", replicates)):
        _integer(name, value, 1)
    if block_length > n:
        raise ValueError("block_length cannot exceed the sample size.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator.")
    blocks = (n + block_length - 1) // block_length
    starts = rng.integers(0, n - block_length + 1, size=(replicates, blocks))
    return (starts[..., None] + np.arange(block_length)).reshape(replicates, -1)[:, :n]


def block_bootstrap(values, *, block_length=10, replicates=2000, seed=42,
                    batch_size=64):
    """Estimate means and pointwise percentile 95% intervals in bounded batches.

    ``values`` is a finite real array shaped (n,) or (n,k), already aligned by
    observation date. All columns use exactly the same indices. For paired
    comparisons, pass the per-date loss differences, first model minus second;
    negative differences favor the first model. Passing violation indicators
    instead yields intervals for violation rates. The input is not modified.

    Results always use lists for mean/lower/upper, including one-column input.
    RNG is local default_rng(seed), currently PCG64. Batch boundaries do not
    change draws. Exact identity across NumPy versions is not promised.
    Principal arrays use approximately 8*(n*k + replicates*k +
    batch_size*n*(k+2)) bytes, plus the caller's input and library overhead;
    only replicate means, never all resampled histories, are retained.
    Samples with fewer than ten effective blocks receive an explicit warning.
    n < block_length is rejected instead of silently changing the protocol.
    """
    for name, value, minimum in (("block_length", block_length, 1),
                                 ("replicates", replicates, 2),
                                 ("seed", seed, 0), ("batch_size", batch_size, 1)):
        _integer(name, value, minimum)
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("Complex observations are unsupported.")
        sample = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("values must contain finite real observations.") from exc
    if sample.ndim == 1:
        sample = sample[:, None]
    if (sample.ndim != 2 or 0 in sample.shape or not np.isfinite(sample).all()):
        raise ValueError("values must be a nonempty finite (n,) or (n,k) array.")
    n, columns = sample.shape
    if block_length > n:
        raise ValueError("block_length cannot exceed the sample size.")

    # Positive column scales preserve means and quantiles while avoiding
    # overflow in sums of finite losses, even near the float64 boundary.
    scale = np.max(np.abs(sample), axis=0)
    scale[scale == 0] = 1.0
    normalized = sample / scale
    means = np.empty((replicates, columns))
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        indices = block_indices(n, block_length, count, rng=rng)
        means[start:start + count] = normalized[indices].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975], axis=0, method="linear")
    estimate = normalized.mean(axis=0)
    restored = np.clip(np.stack((estimate, lower, upper)), -1, 1) * scale
    if not np.isfinite(restored).all():
        raise ValueError("Bootstrap results exceed finite float64 range.")
    return {
        "method": "noncircular_moving_block_percentile", "level": .95,
        "mean": restored[0].tolist(), "lower": restored[1].tolist(),
        "upper": restored[2].tolist(), "n": n, "block_length": block_length,
        "blocks_per_replicate": (n + block_length - 1) // block_length,
        "candidate_blocks": n - block_length + 1,
        "replicates": replicates, "seed": seed, "rng": "NumPy default_rng/PCG64",
        "warning": ("Fewer than 10 effective blocks (n / block_length); "
                    "intervals are unstable and do not support strong conclusions."
                    if n / block_length < 10 else None),
    }
