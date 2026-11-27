"""Component (Euler-style) contributions to the portfolio CVaR.

Definitions (conventions of ``risk.statistics.risk_summary``): component
losses ``L_i`` (price units of the portfolio currency; negative = gain),
portfolio loss ``L = sum_i L_i``, ``VaR_hat`` = NumPy linear percentile of
``L`` at ``p = confidence`` (``_linear_percentile``), tail mask
``L >= VaR_hat`` (ties included) and::

    CVaR_hat = mean(L[L >= VaR_hat]),     C_i = mean(L_i[L >= VaR_hat])

so ``sum_i C_i = CVaR_hat`` by linearity of the mean (an algebraic identity;
in floating point it holds up to rounding, reported as ``sum_identity_error``).

Interpretation: for continuous ``L`` near its VaR, ``C_i`` estimates the Euler
allocation ``E[L_i | L >= VaR_p] = d ES_p(sum_j u_j L_j) / d u_i`` at ``u = 1``
(Tasche 1999, "Risk contributions and performance measurement", TU Munich
report; Tasche 2002, J. Banking & Finance 26(7):1519-1533; McNeil, Frey and
Embrechts, *Quantitative Risk Management*, Sec. 8.5 (section to confirm)).
With atoms in ``L`` at the VaR, ``C_i`` is the project's tail conditional
expectation with ties included, not necessarily a derivative.

All draws must be i.i.d. (no antithetic or otherwise correlated paths) for
the bootstrap standard errors.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from risk.intervals import (
    _BOOTSTRAP_MEMORY_BUDGET,
    _row_linear_percentile,
    _validate_int,
    _validate_level,
)
from risk.statistics import _finite_mean, _linear_percentile
from simulation.multi_asset import PortfolioKind, portfolio_paths, portfolio_quantities

_CONTRIBUTION_BOOTSTRAP_STREAM = zlib.crc32(b"cvar_contribution_bootstrap")
# Per resampled element: int64 index, gathered float64 portfolio loss and the
# float64 copy made by np.partition (24 B), as in risk.intervals.
_BYTES_PER_ELEMENT = 24
# Rows of component losses must match the supplied portfolio losses to this
# relative tolerance (scale: max over rows of sum_i |L_i|).
_ROW_MATCH_RTOL = 1e-9


def _validate_components(component_losses: Any) -> np.ndarray:
    array = np.asarray(component_losses)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"component_losses must be real numbers, got dtype {array.dtype}")
    array = array.astype(np.float64, copy=False)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"component_losses must be shaped (n_sims, n_components), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("component_losses must be finite")
    return array


def _portfolio_losses(components: np.ndarray, portfolio_losses: Any) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        row_sums = np.sum(components, axis=1)
    if portfolio_losses is None:
        if not np.isfinite(row_sums).all():
            raise ValueError("portfolio losses exceed finite precision")
        return row_sums
    losses = np.asarray(portfolio_losses)
    if losses.dtype.kind not in "iuf":
        raise ValueError(f"portfolio_losses must be real numbers, got dtype {losses.dtype}")
    losses = losses.astype(np.float64, copy=False)
    if losses.shape != (components.shape[0],) or not np.isfinite(losses).all():
        raise ValueError(f"portfolio_losses must be finite with shape ({components.shape[0]},)")
    scale = float(np.max(np.sum(np.abs(components), axis=1)))
    if np.isfinite(scale) and scale > 0:
        mismatch = float(np.max(np.abs(losses - row_sums)))
        if mismatch > _ROW_MATCH_RTOL * scale:
            raise ValueError(
                f"portfolio_losses differ from the component row sums by {mismatch:.3g} "
                f"(> {_ROW_MATCH_RTOL:g} * {scale:.3g})"
            )
    return losses


def _column_means(tail: np.ndarray) -> list[float]:
    """Overflow-safe ``_finite_mean`` of each column of the tail rows."""
    return [float(_finite_mean(tail[:, i])) for i in range(tail.shape[1])]


def cvar_contributions(component_losses: Any, confidence: float = 0.95, *,
                       portfolio_losses: Any = None) -> dict[str, Any]:
    """Tail contributions ``C_i = mean(L_i | L >= VaR_hat)`` and the CVaR.

    Args:
        component_losses: Finite ``(n_sims, n_components)`` losses ``L_i``.
        confidence: VaR/CVaR level, strictly in (0, 1).
        portfolio_losses: Optional ``(n_sims,)`` portfolio losses defining
            VaR, the tail mask and CVaR (e.g. ``V0 - V_T`` from
            ``portfolio_paths``, which may differ from the row sums by
            rounding). It must match the row sums to a relative ``1e-9`` of
            ``max_t sum_i |L_i|``, else ``ValueError``. Default: row sums.

    Returns:
        ``var``, ``cvar`` (of the portfolio losses, computed like
        ``risk_summary``), ``contributions`` (list, one per column),
        ``n_tail``, ``n``, ``confidence``, ``sum_identity_error``
        (``|sum_i C_i - cvar|``) and ``sum_identity_scale``
        (``sum_i mean(|L_i| | tail)``, the natural scale of that rounding).
    """
    p = _validate_level(confidence, "confidence")
    components = _validate_components(component_losses)
    losses = _portfolio_losses(components, portfolio_losses)
    var = float(_linear_percentile(losses, p))
    mask = losses >= var
    cvar = float(_finite_mean(losses[mask]))
    contributions = _column_means(components[mask])
    scale = float(np.sum(np.mean(np.abs(components[mask]), axis=0)))
    error = float(abs(float(np.sum(contributions)) - cvar))
    if not np.isfinite([var, cvar, scale, error, *contributions]).all():
        raise ValueError("CVaR contributions exceed finite precision")
    return {
        "var": var,
        "cvar": cvar,
        "contributions": contributions,
        "n_tail": int(np.count_nonzero(mask)),
        "n": int(losses.size),
        "confidence": p,
        "sum_identity_error": error,
        "sum_identity_scale": scale,
    }


def portfolio_cvar_contributions(asset_paths: Any, weights: Any,
                                 kind: PortfolioKind = "quantities",
                                 confidence: float = 0.95, *,
                                 initial_value: float = 1.0) -> dict[str, Any]:
    """Per-asset CVaR contributions of a buy-and-hold portfolio.

    ``V = portfolio_paths(asset_paths, weights, kind, initial_value=...)`` and
    the portfolio loss ``L = V[0, 0] - V[-1]`` are exactly those of
    ``risk_summary(V, confidence)``, so ``var`` and ``cvar`` equal its values
    bit for bit (same functions, same mask). Component losses are
    ``L_i = q_i (S_i0 - S_iT)`` with ``q = portfolio_quantities(...)``; their
    row sums can differ from ``L`` by rounding, so ``sum_i C_i = cvar`` holds
    up to rounding (typically a few ulps of ``sum_identity_scale``; reported
    as ``sum_identity_error``), not exactly. Also returns ``quantities``
    (list) and ``initial_value`` (``V0``).
    """
    values = portfolio_paths(asset_paths, weights, kind, initial_value=initial_value)
    paths = np.asarray(asset_paths, dtype=np.float64)
    initial_prices = paths[0, 0]
    quantities = portfolio_quantities(initial_prices, weights, kind, initial_value=initial_value)
    with np.errstate(over="ignore", invalid="ignore"):
        components = quantities * (initial_prices - paths[-1])
        losses = values[0, 0] - values[-1]
    if not np.isfinite(losses).all():
        raise ValueError("Terminal losses exceed finite precision.")
    result = cvar_contributions(components, confidence, portfolio_losses=losses)
    result["quantities"] = [float(q) for q in quantities]
    result["initial_value"] = float(values[0, 0])
    return result


def contribution_batch_size(n: int, n_components: int) -> int:
    """Replicates per batch for a ~64 MiB budget (never below 1).

    Per replicate row: ``24 n`` bytes (index, gathered loss, partition copy)
    plus per-row temporaries kept below ``n (1 + 8 n_components) + 8 n``
    bytes (mask, gathered tail components, tail losses), counted once.
    """
    per_row = _BYTES_PER_ELEMENT * n
    per_row_temporaries = n * (1 + 8 * n_components) + 8 * n
    available = _BOOTSTRAP_MEMORY_BUDGET - per_row_temporaries
    return max(1, available // per_row)


def cvar_contribution_bootstrap(component_losses: Any, confidence: float = 0.95,
                                n_bootstrap: int = 1000, seed: int | None = None,
                                batch_size: int | None = None, *,
                                portfolio_losses: Any = None) -> dict[str, Any]:
    """Bootstrap standard errors of the CVaR contributions ``C_i``.

    i.i.d. bootstrap over simulations: each replicate draws ``n`` path
    indices with replacement (jointly for all components and the portfolio
    loss), recomputes ``VaR*`` (row-wise linear percentile, bitwise equal to
    ``_linear_percentile``), the mask ``L* >= VaR*`` and ``C_i*``, ``CVaR*``.
    Standard errors are the ddof=1 standard deviations of the replicates.

    RNG: ``default_rng(SeedSequence(seed, spawn_key=(crc32(b"cvar_contribution_bootstrap"),)))``;
    indices are drawn one replicate at a time, so results do not depend on
    ``batch_size`` (memory only; default ``contribution_batch_size``, ~64 MiB,
    never below one ``24 n``-byte row). ``seed=None`` uses OS entropy.

    Returns the point estimate dict of ``cvar_contributions`` plus
    ``contribution_standard_errors`` (list), ``cvar_standard_error``,
    ``n_bootstrap``, ``seed``, ``rng`` and a ``note`` when ``n_tail < 2``.
    """
    p = _validate_level(confidence, "confidence")
    replicates = _validate_int(n_bootstrap, "n_bootstrap", 2)
    seed_value = None if seed is None else _validate_int(seed, "seed", 0)
    batch_value = None if batch_size is None else _validate_int(batch_size, "batch_size", 1)
    components = _validate_components(component_losses)
    losses = _portfolio_losses(components, portfolio_losses)
    result = cvar_contributions(components, p, portfolio_losses=losses)

    n, k = components.shape
    batch = batch_value if batch_value is not None else contribution_batch_size(n, k)
    rng = np.random.default_rng(
        np.random.SeedSequence(seed_value, spawn_key=(_CONTRIBUTION_BOOTSTRAP_STREAM,)))
    contribution_reps = np.empty((replicates, k))
    cvar_reps = np.empty(replicates)
    for start in range(0, replicates, batch):
        rows = min(batch, replicates - start)
        indices = np.empty((rows, n), dtype=np.int64)
        for row in range(rows):
            indices[row] = rng.integers(0, n, size=n)
        samples = losses[indices]
        var_rows = _row_linear_percentile(samples, p)
        for row in range(rows):
            mask = samples[row] >= var_rows[row]
            cvar_reps[start + row] = _finite_mean(samples[row][mask])
            contribution_reps[start + row] = _column_means(components[indices[row][mask]])
        del indices, samples
    if not np.isfinite(contribution_reps).all() or not np.isfinite(cvar_reps).all():
        raise ValueError("Bootstrap contribution replicates exceed finite precision.")

    def _sd(values: np.ndarray) -> float:
        scale = float(np.max(np.abs(values)))
        return 0.0 if scale == 0 else float(np.std(values / scale, ddof=1)) * scale

    result.update({
        "contribution_standard_errors": [_sd(contribution_reps[:, i]) for i in range(k)],
        "cvar_standard_error": _sd(cvar_reps),
        "n_bootstrap": replicates,
        "seed": seed_value,
        "rng": "NumPy default_rng/PCG64",
    })
    if result["n_tail"] < 2:
        result["note"] = ("fewer than 2 tail losses: bootstrap standard errors "
                          "have no meaningful interpretation")
    return result
