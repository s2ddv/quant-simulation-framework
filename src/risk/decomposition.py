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
from collections.abc import Callable
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
# Per replicate row, besides the 24 n bytes: the float64 row results of
# _row_linear_percentile (the difference, both interpolations and the
# np.where output: 4 x 8 B).
_BYTES_PER_ROW = 32
# Row check of supplied portfolio losses, row by row t:
#   |L_t - sum_i L_ti| <= RTOL * sum_i a_ti + ATOL
# with magnitudes a_ti >= |L_ti| (``|L_ti|`` by default). A float sum of k
# terms has rounding <= (k - 1) eps sum_i |x_i| (Higham 2002, *Accuracy and
# Stability of Numerical Algorithms*, ch. 4, recursive summation), so a
# re-summed or differently computed L_t stays far below RTOL for any realistic
# k. ATOL (smallest normal float64) only covers rows whose magnitudes are all
# zero or subnormal; it does not widen the relative tolerance.
_ROW_MATCH_RTOL = 1e-9
_ROW_MATCH_ATOL = float(np.finfo(np.float64).tiny)
# The check works on row chunks sized so that its float64 (rows, k) work array
# plus at most 8 float64 row vectors fit in this many bytes (or one row), never
# on a full (n, k) temporary.
_ROW_CHECK_CHUNK_BYTES = 2**20


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


def _validate_losses(components: np.ndarray, portfolio_losses: Any) -> np.ndarray:
    losses = np.asarray(portfolio_losses)
    if losses.dtype.kind not in "iuf":
        raise ValueError(f"portfolio_losses must be real numbers, got dtype {losses.dtype}")
    losses = losses.astype(np.float64, copy=False)
    if losses.shape != (components.shape[0],) or not np.isfinite(losses).all():
        raise ValueError(f"portfolio_losses must be finite with shape ({components.shape[0]},)")
    return losses


def _check_row_sums(components: np.ndarray, losses: np.ndarray,
                    magnitude: Callable[[int, int], np.ndarray] | None = None) -> None:
    """Reject rows with ``|L_t - sum_i L_ti| > RTOL sum_i a_ti + ATOL``.

    ``magnitude(start, stop)`` returns a fresh float64 ``(stop - start, k)``
    array ``a >= |L|`` (default ``|L|``). Each row is divided by
    ``max_i a_ti`` before summing, so neither the scale nor the row sum can
    overflow; the check is never skipped. Works in row chunks whose
    temporaries (one ``(rows, k)`` float64 array and a few row vectors) stay
    within 1 MiB, or one row.
    """
    n, k = components.shape
    rows = max(1, _ROW_CHECK_CHUNK_BYTES // (8 * (k + 8)))
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        chunk = components[start:stop]
        with np.errstate(over="ignore", invalid="ignore"):
            work = np.abs(chunk) if magnitude is None else magnitude(start, stop)
        if not np.isfinite(work).all():
            raise ValueError("portfolio loss tolerance scale exceeds finite precision")
        peak = np.max(work, axis=1)
        divisor = np.where(peak > 0, peak, 1.0)
        with np.errstate(over="ignore", under="ignore"):
            np.divide(work, divisor[:, None], out=work)
            size = np.sum(work, axis=1)
            np.divide(chunk, divisor[:, None], out=work)
            mismatch = np.abs(losses[start:stop] / divisor - np.sum(work, axis=1))
            allowed = _ROW_MATCH_RTOL * size + _ROW_MATCH_ATOL / divisor
        bad = ~(mismatch <= allowed)
        if bad.any():
            row = int(np.argmax(bad))
            raise ValueError(
                f"portfolio_losses[{start + row}] differs from its component row sum by "
                f"{mismatch[row]:.3g} (> {_ROW_MATCH_RTOL:g} * {size[row]:.3g} + "
                f"{_ROW_MATCH_ATOL:.3g} / {divisor[row]:.3g}, in units of "
                f"max_i a_ti = {divisor[row]:.3g})"
            )


def _portfolio_losses(components: np.ndarray, portfolio_losses: Any) -> np.ndarray:
    if portfolio_losses is None:
        with np.errstate(over="ignore", invalid="ignore"):
            row_sums = np.sum(components, axis=1)
        if not np.isfinite(row_sums).all():
            raise ValueError("portfolio losses exceed finite precision")
        return row_sums
    losses = _validate_losses(components, portfolio_losses)
    _check_row_sums(components, losses)
    return losses


def _column_means(tail: np.ndarray) -> list[float]:
    """Overflow-safe ``_finite_mean`` of each column of the tail rows."""
    return [float(_finite_mean(tail[:, i])) for i in range(tail.shape[1])]


def _contributions(components: np.ndarray, p: float, losses: np.ndarray) -> dict[str, Any]:
    """Point estimates from validated components, level and portfolio losses.

    Temporaries: the ``n``-byte mask, ``8 n_tail`` tail losses and one
    ``8 n_tail k`` tail copy (its absolute values are taken in place after
    the contributions are computed).
    """
    var = float(_linear_percentile(losses, p))
    mask = losses >= var
    cvar = float(_finite_mean(losses[mask]))
    tail = components[mask]
    contributions = _column_means(tail)
    np.abs(tail, out=tail)
    scale = float(np.sum(np.mean(tail, axis=0)))
    del tail
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


def cvar_contributions(component_losses: Any, confidence: float = 0.95, *,
                       portfolio_losses: Any = None) -> dict[str, Any]:
    """Tail contributions ``C_i = mean(L_i | L >= VaR_hat)`` and the CVaR.

    Args:
        component_losses: Finite ``(n_sims, n_components)`` losses ``L_i``.
        confidence: VaR/CVaR level, strictly in (0, 1).
        portfolio_losses: Optional ``(n_sims,)`` portfolio losses defining
            VaR, the tail mask and CVaR, which may differ from the row sums
            by rounding. Each row must satisfy
            ``|L_t - sum_i L_ti| <= 1e-9 sum_i |L_ti| + tiny`` (``tiny`` =
            smallest normal float64), evaluated in units of
            ``max_i |L_ti|`` so it cannot overflow, else ``ValueError``.
            The tolerance is per row: one large row does not loosen the
            others. Rounding proportional to a notional larger than
            ``sum_i |L_ti|`` (e.g. ``V0 - V_T`` with prices that barely move)
            can exceed it; ``portfolio_cvar_contributions`` checks such
            losses against the position notional instead. Default: row sums.

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
    return _contributions(components, p, losses)


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

    Both quantities are built here, so their row mismatch is pure rounding of
    order ``(k + 2) eps sum_i |q_i| max(S_i0, S_iT)`` (the position notional,
    which can be far larger than ``sum_i |L_ti|`` for short horizons or tiny
    volatilities). The row check of ``cvar_contributions`` is therefore
    applied with magnitudes ``a_ti = |q_i| max(S_i0, S_iT) >= |L_ti|``:
    ``|L_t - sum_i L_ti| <= 1e-9 sum_i a_ti + tiny``. It only guards against
    internal inconsistency; it cannot fail by rounding for ``k`` below
    ~``4e6`` assets.
    """
    p = _validate_level(confidence, "confidence")
    values = portfolio_paths(asset_paths, weights, kind, initial_value=initial_value)
    paths = np.asarray(asset_paths, dtype=np.float64)
    initial_prices = paths[0, 0]
    terminal_prices = paths[-1]
    quantities = portfolio_quantities(initial_prices, weights, kind, initial_value=initial_value)
    with np.errstate(over="ignore", invalid="ignore"):
        components = quantities * (initial_prices - terminal_prices)
        losses = values[0, 0] - values[-1]
    if not np.isfinite(losses).all():
        raise ValueError("Terminal losses exceed finite precision.")
    components = _validate_components(components)
    abs_quantities = np.abs(quantities)

    def notional(start: int, stop: int) -> np.ndarray:
        magnitude = np.maximum(initial_prices, terminal_prices[start:stop])
        magnitude *= abs_quantities
        return magnitude

    _check_row_sums(components, losses, notional)
    result = _contributions(components, p, losses)
    result["quantities"] = [float(q) for q in quantities]
    result["initial_value"] = float(values[0, 0])
    return result


def contribution_batch_size(n: int, n_components: int) -> int:
    """Replicates per batch for the 64 MiB bootstrap budget (never below 1).

    ``batch = max(1, (64 MiB - T) // (24 n + 32))`` where ``24 n + 32`` bytes
    per replicate row hold the int64 indices, the gathered losses, the
    ``np.partition`` copy and four float64 row results, and
    ``T = n (25 + 8 k)`` (``k = n_components``) bounds the per-replicate
    temporaries, counted once: ``n`` bytes of mask, ``8 n_tail`` tail
    indices, ``8 n_tail k`` gathered tail components and ``16 n_tail`` for
    the overflow fallback of ``_finite_mean``, with the worst case
    ``n_tail = n`` (ties at VaR can put every row in the tail). Without ties
    ``n_tail ~ n (1 - p)``, so ``T`` is conservative by about ``1 / (1 - p)``.
    When ``T + 24 n + 32`` exceeds 64 MiB the batch is 1 and the bound
    exceeds the budget. See ``cvar_contribution_bootstrap`` for the full
    peak, including what lies outside this budget.
    """
    per_row = _BYTES_PER_ELEMENT * n + _BYTES_PER_ROW
    per_row_temporaries = n * (25 + 8 * n_components)
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
    ``portfolio_losses`` is checked as in ``cvar_contributions``.

    RNG: ``default_rng(SeedSequence(seed, spawn_key=(crc32(b"cvar_contribution_bootstrap"),)))``;
    indices are drawn one replicate at a time, so results do not depend on
    ``batch_size`` (memory only; default ``contribution_batch_size``).
    ``seed=None`` uses OS entropy.

    Memory (``n`` paths, ``k`` components, ``B = n_bootstrap``,
    ``b = min(batch, B)`` rows per batch), bytes allocated by this call::

        peak <= b (24 n + 32) + max(n (25 + 8 k), 1 MiB) + 8 B (k + 3) + 8 n

    ``b (24 n + 32) + n (25 + 8 k)`` is the 64 MiB budget of
    ``contribution_batch_size`` (exceeded only when ``b = 1``). Outside it:
    the replicate outputs and the standard-deviation temporaries
    ``8 B (k + 3)``, the ``8 n`` portfolio-loss vector (row sums, or a float64
    copy of non-float64 ``portfolio_losses``) and, when ``portfolio_losses``
    is given, the chunked row check (at most 1 MiB, or one row of
    ``8 (k + 8)`` bytes), which runs before the batches; the point estimate
    (``n + 8 n_tail (k + 1)``, plus ``16 n_tail`` in the overflow fallback)
    also runs before them, within ``n (25 + 8 k)``. Not counted: the
    caller's inputs (and a float64 copy of non-float64 ``component_losses``),
    NumPy ufunc buffers and Python object overhead.

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
    result = _contributions(components, p, losses)

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
