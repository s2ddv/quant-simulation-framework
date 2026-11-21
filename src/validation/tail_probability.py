"""Event-probability estimation for multi-day losses with proportion CIs.

Only binomial-proportion methods are used: terminal prices under t shocks have
no finite mean, so moment-based standard errors are deliberately not offered.
Returns are simple, ``S_T / S_0 - 1``; a loss threshold ``x`` means
``R <= -x``. Time in years, mu in year^-1, sigma in year^-1/2.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from simulation.gbm import GbmSimulator


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= k <= n or not 0 < level < 1:
        raise ValueError("require n > 0, 0 <= k <= n and 0 < level < 1")
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)   # exact boundary values
    hi = 1.0 if k == n else min(1.0, centre + half)
    return float(lo), float(hi)


def clopper_pearson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (conservative) interval; well defined when k == 0 or k == n."""
    if n <= 0 or not 0 <= k <= n or not 0 < level < 1:
        raise ValueError("require n > 0, 0 <= k <= n and 0 < level < 1")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def normal_loss_probability(loss: float, mu: float, sigma: float, horizon: float) -> float:
    """Exact P(S_T/S_0 - 1 <= -loss) under GBM, 0 <= loss < 1."""
    if not 0 <= loss < 1:
        raise ValueError("loss must satisfy 0 <= loss < 1")
    s = sigma * np.sqrt(horizon)
    return float(stats.norm.cdf((np.log1p(-loss) - (mu - 0.5 * sigma**2) * horizon) / s))


def estimate_loss_probability(loss: float, *, mu: float, sigma: float, horizon: float,
                              steps: int, n_sims: int, seed: int,
                              dist_type: str = "normal", df: float | None = None,
                              batch_size: int = 100_000, level: float = 0.95) -> dict:
    """Batched Monte Carlo estimate of P(simple return <= -loss) with CIs.

    Each batch uses an independent child of ``SeedSequence(seed)`` so results do
    not depend on memory-driven batch layout only through the stated batch_size.
    The returned dict records every parameter of the experiment.
    """
    if not 0 <= loss < 1:
        raise ValueError("loss must satisfy 0 <= loss < 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n_batches = -(-n_sims // batch_size)
    children = np.random.SeedSequence(seed).spawn(n_batches)
    hits, done = 0, 0
    for child in children:
        m = min(batch_size, n_sims - done)
        sim = GbmSimulator(1.0, horizon, steps, m, mu=mu, sigma=sigma, df=df)
        sim.rng = np.random.default_rng(child)
        terminal = sim.generate_paths(dist_type)[-1]
        hits += int(np.count_nonzero(terminal - 1.0 <= -loss))
        done += m
    return {
        "params": {"loss": loss, "mu": mu, "sigma": sigma, "horizon": horizon,
                   "steps": steps, "n_sims": n_sims, "seed": seed,
                   "dist_type": dist_type, "df": df, "batch_size": batch_size},
        "hits": hits, "probability": hits / n_sims,
        "wilson": wilson_interval(hits, n_sims, level),
        "clopper_pearson": clopper_pearson_interval(hits, n_sims, level),
        "level": level,
    }
