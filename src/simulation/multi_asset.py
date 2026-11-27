"""Correlated multi-asset GBM and buy-and-hold portfolio values.

Model (constant parameters, ``i, j = 1..n_assets``)::

    dS_i = mu_i S_i dt + sigma_i S_i dW_i,     d<W_i, W_j> = rho_ij dt

Units: prices in each asset's currency, time in years, ``mu`` in year^-1,
``sigma`` in year^-1/2; ``R = (rho_ij)`` is dimensionless. Exact solution on
any grid (Glasserman 2003, *Monte Carlo Methods in Financial Engineering*,
Sec. 3.2.3, "Multiple Dimensions")::

    ln S(t + dt) = ln S(t) + (mu - sigma**2/2) dt + sqrt(dt) sigma (.) (L Z)

with ``Z ~ N(0, I)`` independent per step and ``L L^T = R``. Consequently
(same reference, section number to confirm)::

    Cov(ln S_i(T), ln S_j(T)) = rho_ij sigma_i sigma_j T
    Cov(S_i(T), S_j(T)) = S_i0 S_j0 exp((mu_i + mu_j) T) (exp(rho_ij sigma_i sigma_j T) - 1)

The increments are exact lognormal draws, not Euler-Maruyama: the grid adds no
discretization error to the joint law at the grid points.

Array contract (approved 2026-11-27): ``generate_paths`` returns a float64
array shaped ``(steps + 1, n_sims, n_assets)`` whose row 0 is ``s0`` broadcast
to every simulation. The single-asset contract ``(steps + 1, n_sims)`` of
``simulation.base`` is unchanged; ``portfolio_paths`` reduces the 3-D array to
that 2-D contract so ``risk.statistics.risk_summary`` applies unchanged.

Correlation matrices are validated, never repaired silently.
``nearest_correlation`` (Higham 2002) is a separate, explicit function.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from simulation.base import BaseStochasticSimulator

PortfolioKind = Literal["quantities", "initial_weights"]
Factorization = Literal["cholesky", "eigh_clipped"]

# Absolute tolerance of the correlation checks (symmetry, unit diagonal,
# |rho_ij| <= 1 and the smallest eigenvalue).
CORRELATION_TOL = 1e-12
_WEIGHT_SUM_TOL = 1e-12


def _real_array(value: Any, name: str) -> np.ndarray:
    """Return a float64 copy of a real (int/float) array; reject bool/complex."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, got dtype {array.dtype}")
    return np.array(array, dtype=np.float64)


def validate_correlation(correlation: Any, *, tol: float = CORRELATION_TOL) -> np.ndarray:
    """Validate a correlation matrix and return it as a float64 copy.

    Requirements (each failure raises ``ValueError`` with a specific message):
    real dtype (bool and complex rejected), 2-D square with ``n >= 1``, finite,
    symmetric (``max |R - R^T| <= tol``), unit diagonal (``|R_ii - 1| <= tol``),
    ``|R_ij| <= 1 + tol`` and smallest eigenvalue (``numpy.linalg.eigvalsh``)
    ``>= -tol``. The matrix is never corrected; use ``nearest_correlation``
    explicitly for an invalid estimate.
    """
    if (isinstance(tol, bool) or not isinstance(tol, Real)
            or not np.isfinite(tol) or tol < 0):
        raise ValueError("tol must be a finite nonnegative real number")
    matrix = _real_array(correlation, "correlation")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"correlation must be a non-empty square 2-D matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation must be finite")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > tol:
        raise ValueError(f"correlation must be symmetric (max |R - R^T| = {asymmetry:.3g} > {tol:.3g})")
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if diagonal_error > tol:
        raise ValueError(f"correlation diagonal must be 1 (max |R_ii - 1| = {diagonal_error:.3g})")
    largest = float(np.max(np.abs(matrix)))
    if largest > 1.0 + tol:
        raise ValueError(f"correlation entries must satisfy |R_ij| <= 1 (found {largest:.17g})")
    smallest = float(np.linalg.eigvalsh(matrix)[0])
    if smallest < -tol:
        raise ValueError(
            f"correlation must be positive semidefinite (smallest eigenvalue {smallest:.3g} < "
            f"{-tol:.3g}); nearest_correlation offers an explicit repair"
        )
    return matrix


def correlation_factor(correlation: Any) -> tuple[np.ndarray, Factorization]:
    """Return ``(L, method)`` with ``L L^T = R`` for a validated ``R``.

    ``method == "cholesky"``: lower-triangular ``numpy.linalg.cholesky(R)``
    (``R`` positive definite). If Cholesky fails (rank-deficient ``R``, e.g.
    ``rho = +-1``) and ``R`` passed ``validate_correlation``, the fallback is
    ``method == "eigh_clipped"``: ``R = V diag(lam) V^T`` by
    ``numpy.linalg.eigh``, eigenvalues in ``[-1e-12, 0)`` set to 0,
    ``L = V diag(sqrt(lam+))``, and each row of ``L`` divided by its norm so
    that ``(L L^T)_ii = 1`` up to rounding (per-asset variance preserved).
    ``scipy.linalg.ldl`` (Bunch-Kaufman, block pivots) is deliberately not
    used. The signs and the basis of eigenvectors of repeated eigenvalues
    depend on LAPACK, so with the fallback the same seed may give different
    paths across NumPy/LAPACK builds (the law is the same).
    """
    matrix = validate_correlation(correlation)
    try:
        factor = np.linalg.cholesky(matrix)
        if np.isfinite(factor).all():
            return factor, "cholesky"
    except np.linalg.LinAlgError:
        pass
    eigenvalues, vectors = np.linalg.eigh(matrix)
    eigenvalues = np.where((eigenvalues < 0) & (eigenvalues >= -CORRELATION_TOL), 0.0, eigenvalues)
    if np.any(eigenvalues < 0):
        raise ValueError("correlation has an eigenvalue below -1e-12")
    factor = vectors * np.sqrt(eigenvalues)
    norms = np.sqrt(np.sum(factor * factor, axis=1))
    if not np.all(norms > 0):
        raise ValueError("correlation factor has a zero row")
    factor = factor / norms[:, None]
    return factor, "eigh_clipped"


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    """Frobenius projection onto the PSD cone (eigenvalues clipped at 0)."""
    eigenvalues, vectors = np.linalg.eigh(matrix)
    projected = (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T
    return (projected + projected.T) / 2


def nearest_correlation(matrix: Any, *, tol: float = 1e-10,
                        max_iterations: int = 1000) -> dict[str, Any]:
    """Nearest correlation matrix in the Frobenius norm (explicit repair).

    Higham, N. J. (2002), "Computing the nearest correlation matrix -- a
    problem from finance", IMA J. Numer. Anal. 22(3):329-343, Algorithm 3.3
    with ``W = I``: alternating projections onto the PSD cone ``S``
    (``P_S``: clip eigenvalues at 0) and the unit-diagonal set ``U``
    (``P_U``: set the diagonal to 1), with Dykstra's correction ``dS``::

        R_k = Y_{k-1} - dS_{k-1};  X_k = P_S(R_k);  dS_k = X_k - R_k;  Y_k = P_U(X_k)

    Stops when ``max(||X_k - X_{k-1}||/||X_k||, ||Y_k - Y_{k-1}||/||Y_k||,
    ||Y_k - X_k||/||Y_k||) <= tol`` (Frobenius norms). Post-processing so the
    output passes ``validate_correlation``: eigenvalues of ``Y`` clipped at
    0, congruence rescaling ``D^{-1/2} X D^{-1/2}`` with ``D = diag(X)``,
    symmetrization and an exact unit diagonal.

    Args:
        matrix: Real, finite, square, symmetric (``max |A - A^T| <= 1e-12``)
            matrix; its diagonal need not be 1. It is symmetrized
            ``(A + A^T)/2`` before iterating.
        tol: Positive convergence tolerance.
        max_iterations: Positive iteration cap.

    Returns:
        ``{"correlation": ndarray, "iterations": int, "converged": True,
        "frobenius_distance": ||A - correlation||_F}``.

    Raises:
        ValueError: Invalid input, or no convergence within ``max_iterations``
            (never returns an unconverged matrix), or a post-processed result
            that fails ``validate_correlation``.
    """
    if (isinstance(tol, bool) or not isinstance(tol, Real)
            or not np.isfinite(tol) or tol <= 0):
        raise ValueError("tol must be a finite positive real number")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral)
            or max_iterations <= 0):
        raise ValueError("max_iterations must be a positive integer")
    original = _real_array(matrix, "matrix")
    if original.ndim != 2 or original.shape[0] != original.shape[1] or original.shape[0] == 0:
        raise ValueError(f"matrix must be a non-empty square 2-D matrix, got shape {original.shape}")
    if not np.isfinite(original).all():
        raise ValueError("matrix must be finite")
    if float(np.max(np.abs(original - original.T))) > CORRELATION_TOL:
        raise ValueError("matrix must be symmetric (max |A - A^T| <= 1e-12)")
    a = (original + original.T) / 2

    correction = np.zeros_like(a)
    y = a.copy()
    x = a.copy()
    converged = False
    iterations = 0
    while iterations < int(max_iterations):
        iterations += 1
        previous_x, previous_y = x, y
        r = y - correction
        x = _project_psd(r)
        correction = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        norm_x = max(float(np.linalg.norm(x)), np.finfo(float).tiny)
        norm_y = float(np.linalg.norm(y))
        change = max(float(np.linalg.norm(x - previous_x)) / norm_x,
                     float(np.linalg.norm(y - previous_y)) / norm_y,
                     float(np.linalg.norm(y - x)) / norm_y)
        if change <= tol:
            converged = True
            break
    if not converged:
        raise ValueError(f"nearest_correlation did not converge in {max_iterations} iterations")

    clipped = _project_psd(y)
    scale = np.sqrt(np.diag(clipped))
    if not np.all(scale > 0) or not np.isfinite(scale).all():
        raise ValueError("nearest_correlation produced a zero diagonal")
    result = clipped / scale[:, None] / scale[None, :]
    result = (result + result.T) / 2
    np.fill_diagonal(result, 1.0)
    np.clip(result, -1.0, 1.0, out=result)
    result = validate_correlation(result)
    return {
        "correlation": result,
        "iterations": int(iterations),
        "converged": True,
        "frobenius_distance": float(np.linalg.norm(original - result)),
    }


def estimate_path_bytes(steps: int, n_sims: int, n_assets: int,
                        temporary_arrays: int = 2) -> int:
    """Bytes of the float64 arrays held by ``generate_paths`` at its peak.

    ``8 (steps + 1) n_sims n_assets`` for the output, plus
    ``temporary_arrays`` per-step blocks of ``8 n_sims n_assets`` bytes
    (``generate_paths`` uses 2: the shock block ``Z`` and the correlated block
    ``Z L^T``), plus ``2 * 8 n_assets**2`` for the factor ``L`` and the
    contiguous copy of ``L^T`` that ``matmul`` may make, plus
    ``ufunc_buffer_bytes()`` (two NumPy ufunc iteration buffers, 64 KiB each
    with the default ``numpy.getbufsize()``; the in-place broadcasts over the
    asset axis use one, measured with ``tracemalloc``). Excludes Python object
    overhead and caller-held arrays.
    """
    return (8 * (steps + 1) * n_sims * n_assets
            + 8 * temporary_arrays * n_sims * n_assets
            + 2 * 8 * n_assets * n_assets + ufunc_buffer_bytes())


def ufunc_buffer_bytes() -> int:
    """Allowance for two float64 ufunc buffers of ``numpy.getbufsize()`` elements."""
    return 2 * 8 * int(np.getbufsize())


def _vector(value: Any, name: str) -> np.ndarray:
    """Finite float64 1-D array of size >= 1 (scalars and bool rejected)."""
    array = _real_array(value, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


class MultiAssetGbmSimulator:
    """Exact correlated GBM for ``n_assets >= 1`` assets.

    Not a ``BaseStochasticSimulator`` subclass (``s0`` is a vector), but the
    scalar arguments follow its validation and ``MAX_WORKING_BYTES`` is the
    same budget object (``BaseStochasticSimulator.MAX_WORKING_BYTES``).

    Args:
        s0: Initial prices, 1-D, finite and positive (asset currencies).
        horizon: Years, finite positive real.
        steps: Positive integer number of grid intervals; ``dt = horizon/steps``.
        n_sims: Positive integer number of paths.
        seed: Nonnegative integer or None for ``numpy.random.default_rng``.
        mu: Price drifts (year^-1), 1-D, finite.
        sigma: Volatilities (year^-1/2), 1-D, finite, nonnegative.
        correlation: ``(n_assets, n_assets)`` correlation of the Brownian
            drivers; validated by ``validate_correlation``, factored by
            ``correlation_factor`` (``self.factorization``).

    RNG: one local ``numpy.random.Generator`` (``self.rng``, PCG64) is the
    only stream; every call advances it.

    Memory of ``__init__`` (not checked against ``MAX_WORKING_BYTES``):
    validation and factorization of ``R`` hold several float64
    ``n_assets x n_assets`` arrays at once (copy, ``R - R^T``, ``eigvalsh``
    workspace, Cholesky or ``eigh`` factor): about ``4-5 * 8 n_assets**2``
    bytes at peak (``tracemalloc`` at ``n_assets = 1500``: 4.0x for a float64
    array, one more copy for other inputs such as nested lists), plus LAPACK
    workspace not seen by ``tracemalloc``. ``path_bytes``/``terminal_bytes``
    count only the ``16 n_assets**2`` bytes kept for ``generate_*``.

    Known limitation of ``validate_correlation``: the eigenvalue test uses
    the absolute tolerance ``-1e-12`` (plan contract). Rounding of
    ``eigvalsh`` grows roughly like ``eps * n_assets * ||R||``, so a valid
    PSD matrix of large order can be rejected; e.g. ``R = ones((1000, 1000))``
    (rank 1, exactly PSD) gives ``lambda_min ~ -4.9e-12``. Behaviour is
    unchanged; such matrices currently need ``nearest_correlation`` or a
    smaller model.
    """

    MAX_WORKING_BYTES = BaseStochasticSimulator.MAX_WORKING_BYTES

    def __init__(self, s0: Any, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: Any, sigma: Any,
                 correlation: Any) -> None:
        if (isinstance(horizon, bool) or not isinstance(horizon, Real)
                or not np.isfinite(horizon) or horizon <= 0):
            raise ValueError("horizon must be finite and positive")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if isinstance(n_sims, bool) or not isinstance(n_sims, Integral) or n_sims <= 0:
            raise ValueError("n_sims must be a positive integer")
        if seed is not None and (isinstance(seed, bool)
                                 or not isinstance(seed, Integral) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        s0_vec = _vector(s0, "s0")
        mu_vec = _vector(mu, "mu")
        sigma_vec = _vector(sigma, "sigma")
        if np.any(s0_vec <= 0):
            raise ValueError("s0 must be finite and positive")
        if np.any(sigma_vec < 0):
            raise ValueError("sigma must be nonnegative")
        if not (s0_vec.size == mu_vec.size == sigma_vec.size):
            raise ValueError("s0, mu and sigma must have the same length")
        matrix = validate_correlation(correlation)
        if matrix.shape[0] != s0_vec.size:
            raise ValueError(
                f"correlation shape {matrix.shape} does not match n_assets={s0_vec.size}"
            )

        self.s0: np.ndarray = s0_vec
        self.mu: np.ndarray = mu_vec
        self.sigma: np.ndarray = sigma_vec
        self.correlation: np.ndarray = matrix
        self.n_assets: int = int(s0_vec.size)
        self.horizon: float = float(horizon)
        self.steps: int = int(steps)
        self.n_sims: int = int(n_sims)
        self.seed: int | None = None if seed is None else int(seed)
        self.dt: float = self.horizon / self.steps
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("horizon / steps must be finite and positive")
        factor, method = correlation_factor(matrix)
        self.factor: np.ndarray = factor
        self.factorization: Factorization = method
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    def _check_memory(self, required: int, output_bytes: int) -> None:
        if required > self.MAX_WORKING_BYTES:
            raise ValueError(
                f"path arrays require {required} bytes (output {output_bytes}); "
                f"limit is {self.MAX_WORKING_BYTES} bytes"
            )

    def _drift_scale(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-asset ``(mu - sigma**2/2) dt`` and ``sigma sqrt(dt)``.

        Same arithmetic as ``GbmSimulator``; if ``sigma**2`` overflows while
        ``sigma**2 dt`` does not, it falls back to ``mu dt - scale**2/2``.
        """
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scale = self.sigma * np.sqrt(dt)
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            fallback = self.mu * dt - 0.5 * scale**2
            drift = np.where(np.isfinite(drift), drift, fallback)
        if not np.isfinite(drift).all() or not np.isfinite(scale).all():
            raise ValueError("simulation increments exceed finite precision")
        return drift, scale

    def _to_prices(self, log_returns: np.ndarray) -> None:
        """Map cumulative log returns ``X`` to ``s0 * exp(X)`` in place.

        Like ``GbmSimulator``: multiply after ``exp`` unless some ``X`` would
        overflow or become subnormal, then exponentiate ``ln s0 + X``.
        Non-finite or non-positive prices raise ``ValueError``.
        """
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if log_returns.size and (
                    np.max(log_returns) >= np.log(np.finfo(np.float64).max)
                    or np.min(log_returns) <= np.log(np.finfo(np.float64).tiny)):
                log_returns += np.log(self.s0)
                np.exp(log_returns, out=log_returns)
            else:
                np.exp(log_returns, out=log_returns)
                log_returns *= self.s0
        if log_returns.size:
            low, high = np.min(log_returns), np.max(log_returns)
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0:
                raise ValueError("simulated prices exceed finite positive precision")

    def path_bytes(self) -> int:
        """Checked array bytes of ``generate_paths`` (see ``estimate_path_bytes``).

        Excludes the ``O(n_assets**2)`` memory already used by ``__init__`` to
        validate and factor the correlation (see the class docstring)."""
        return estimate_path_bytes(self.steps, self.n_sims, self.n_assets, 2)

    def generate_paths(self) -> np.ndarray:
        """Return ``(steps + 1, n_sims, n_assets)`` float64 prices.

        Row 0 is ``s0`` broadcast. For ``k = 1..steps`` one block
        ``Z_k ~ N(0, I)`` of shape ``(n_sims, n_assets)`` is drawn from
        ``self.rng`` into a reused buffer (the blocks, in order, are the same
        values as one C-order ``(steps, n_sims, n_assets)`` draw), correlated
        as ``Z_k L^T`` into a second buffer, and
        ``X_k = X_{k-1} + (mu - sigma**2/2) dt + sqrt(dt) sigma (.) (Z_k L^T)``
        is written to the output, ``X_0 = 0``; finally ``S = s0 exp(X)``.
        With ``n_assets = 1`` and ``R = [[1]]`` this is the arithmetic of
        ``GbmSimulator.generate_paths(dist_type="normal")`` (same law; equal
        values for the same seed were observed for the Cholesky factor
        ``[[1.0]]``, but no bitwise promise is made).

        Memory, checked before any allocation or draw: output
        ``8 (steps + 1) n_sims n_assets`` + 2 blocks ``8 n_sims n_assets`` +
        ``16 n_assets**2`` bytes + ``ufunc_buffer_bytes()``
        (``estimate_path_bytes``), against
        ``MAX_WORKING_BYTES``. Overflow/underflow to non-positive or
        non-finite prices raises ``ValueError``.
        """
        output_bytes = 8 * (self.steps + 1) * self.n_sims * self.n_assets
        self._check_memory(self.path_bytes(), output_bytes)
        drift, scale = self._drift_scale(self.dt)
        paths = np.empty((self.steps + 1, self.n_sims, self.n_assets), dtype=np.float64)
        shocks = np.empty((self.n_sims, self.n_assets), dtype=np.float64)
        correlated = np.empty_like(shocks)
        factor_t = self.factor.T
        paths[0].fill(0.0)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for k in range(1, self.steps + 1):
                self.rng.standard_normal(out=shocks)
                np.matmul(shocks, factor_t, out=correlated)
                correlated *= scale
                correlated += drift
                np.add(paths[k - 1], correlated, out=paths[k])
        del shocks, correlated
        self._to_prices(paths[1:])
        paths[0] = self.s0
        return paths

    def terminal_bytes(self, batch_size: int) -> int:
        """Checked bytes of ``generate_terminal``: output, one shock batch,
        factors and the ufunc buffer allowance (as ``estimate_path_bytes``)."""
        rows = min(batch_size, self.n_sims)
        return (8 * self.n_sims * self.n_assets + 8 * rows * self.n_assets
                + 2 * 8 * self.n_assets * self.n_assets + ufunc_buffer_bytes())

    def generate_terminal(self, batch_size: int | None = None) -> np.ndarray:
        """Return terminal prices ``S(T)`` shaped ``(n_sims, n_assets)``.

        ``ln S_T = ln s0 + (mu - sigma**2/2) T + sqrt(T) sigma (.) (Z L^T)`` with
        one ``Z ~ N(0, I)`` row per simulation (``steps`` unused: exact law).
        Shocks are drawn in batches of ``batch_size`` rows (default
        ``min(n_sims, 65536)``) into a reused buffer and correlated straight
        into the output. The shocks ``Z`` do not depend on ``batch_size``
        (sequential draws from the same generator: the same values as one C-order
        ``(n_sims, n_assets)`` draw, and the same final generator state). The
        prices agree across batch sizes only up to the rounding of the BLAS
        product ``Z L^T``: the BLAS may use a different kernel for a different
        block shape (e.g. a 1-row block), so bitwise equality is not
        guaranteed (relative differences of a few ``eps`` were observed with
        ``batch_size=1``). The sequence differs from
        ``generate_paths(...)[-1]`` with the same seed; the laws are equal.

        Memory checked before allocation (``terminal_bytes``):
        ``8 n_sims n_assets`` output + ``8 batch n_assets`` shock buffer +
        ``16 n_assets**2`` + ``ufunc_buffer_bytes()``.
        """
        if batch_size is None:
            batch = min(self.n_sims, 65536)
        elif isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")
        else:
            batch = min(int(batch_size), self.n_sims)
        output_bytes = 8 * self.n_sims * self.n_assets
        self._check_memory(self.terminal_bytes(batch), output_bytes)
        drift, scale = self._drift_scale(self.horizon)
        terminal = np.empty((self.n_sims, self.n_assets), dtype=np.float64)
        shocks = np.empty((batch, self.n_assets), dtype=np.float64)
        factor_t = self.factor.T
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for start in range(0, self.n_sims, batch):
                rows = min(batch, self.n_sims - start)
                block = shocks[:rows]
                self.rng.standard_normal(out=block)
                target = terminal[start:start + rows]
                np.matmul(block, factor_t, out=target)
                target *= scale
                target += drift
        del shocks
        self._to_prices(terminal)
        return terminal


def _validate_asset_paths(asset_paths: Any) -> np.ndarray:
    paths = np.asarray(asset_paths)
    if paths.dtype.kind not in "iuf":
        raise ValueError(f"asset_paths must contain real numbers, got dtype {paths.dtype}")
    paths = paths.astype(np.float64, copy=False)
    if paths.ndim != 3 or paths.shape[0] < 2 or paths.shape[1] == 0 or paths.shape[2] == 0:
        raise ValueError(
            "asset_paths must be shaped (steps + 1, n_sims, n_assets) with at least "
            f"2 rows, 1 simulation and 1 asset; got {paths.shape}"
        )
    # Scalar reductions (NaN propagates) avoid full-size boolean masks.
    low, high = np.min(paths), np.max(paths)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0:
        raise ValueError("asset_paths must contain only finite positive prices")
    if not np.all(paths[0] == paths[0, 0]):
        raise ValueError("each asset must share the same initial price across simulations")
    return paths


def portfolio_quantities(initial_prices: Any, weights: Any,
                         kind: PortfolioKind = "quantities", *,
                         initial_value: float = 1.0) -> np.ndarray:
    """Buy-and-hold quantities ``q`` (units of each asset).

    * ``kind="quantities"``: ``q = weights`` (negative = short).
    * ``kind="initial_weights"``: ``weights`` are value fractions at t = 0
      with ``|sum(w) - 1| <= 1e-12``; ``q_i = w_i V0 / S_i0`` with
      ``V0 = initial_value`` (portfolio currency, finite positive; only used
      for this kind). Asset currencies are assumed equal to the portfolio's.

    Raises ``ValueError`` on invalid inputs, or if the initial portfolio
    value ``sum(q_i S_i0)`` is not finite and positive (zero-sum or
    short-dominated portfolios are unsupported by ``risk_summary``).
    """
    prices = _vector(initial_prices, "initial_prices")
    if np.any(prices <= 0):
        raise ValueError("initial_prices must be positive")
    w = _vector(weights, "weights")
    if w.size != prices.size:
        raise ValueError(f"weights has {w.size} entries; expected {prices.size}")
    if kind == "quantities":
        quantities = w
    elif kind == "initial_weights":
        if (isinstance(initial_value, bool) or not isinstance(initial_value, Real)
                or not np.isfinite(initial_value) or initial_value <= 0):
            raise ValueError("initial_value must be finite and positive")
        if abs(float(np.sum(w)) - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError("initial_weights must sum to 1 (tolerance 1e-12)")
        with np.errstate(over="ignore", invalid="ignore"):
            quantities = w * float(initial_value) / prices
    else:
        raise ValueError("kind must be 'quantities' or 'initial_weights'")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(prices @ quantities)
    if not np.isfinite(quantities).all() or not np.isfinite(value) or value <= 0:
        raise ValueError(
            "portfolio initial value must be positive; zero-sum/short-dominated "
            "weights are not supported by risk_summary"
        )
    return quantities


def portfolio_paths(asset_paths: Any, weights: Any,
                    kind: PortfolioKind = "quantities", *,
                    initial_value: float = 1.0) -> np.ndarray:
    """Buy-and-hold portfolio values ``V_t = sum_i q_i S_i(t)``, ``(steps + 1, n_sims)``.

    ``asset_paths`` is a finite positive ``(steps + 1, n_sims, n_assets)``
    array with a common initial price per asset; quantities come from
    ``portfolio_quantities`` (no rebalancing). Row 0 is computed once as
    ``V0 = S_0 . q`` and assigned to every column, so it is exactly equal
    across simulations (``risk_summary`` requires that). Intermediate values
    may be negative for short positions; they must be finite. The output
    allocates ``8 (steps + 1) n_sims`` bytes.
    """
    paths = _validate_asset_paths(asset_paths)
    quantities = portfolio_quantities(paths[0, 0], weights, kind, initial_value=initial_value)
    with np.errstate(over="ignore", invalid="ignore"):
        values = paths @ quantities
        initial = float(paths[0, 0] @ quantities)
    values[0] = initial
    if not np.isfinite(values).all():
        raise ValueError("portfolio values exceed finite precision")
    return values
