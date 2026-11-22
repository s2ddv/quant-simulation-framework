"""Exact simulation of Merton (lognormal) and Kou (double-exponential) jumps.

Both models share the physical-measure dynamics (plan E4, section 2.1)::

    dS/S_- = (mu - lambda*k) dt + sigma dW + (J - 1) dN,

with ``N`` a Poisson process of intensity ``lambda`` (year^-1) independent of
``W`` and of the i.i.d. jump multipliers ``J > 0``; ``k = E[J - 1]``. The
compensator ``-lambda*k`` keeps ``E[S_T] = S0 * exp(mu*T)``, so ``mu`` is the
price drift (year^-1) as in the GBM simulator, and ``sigma`` is in year^-1/2.
Over any interval of length ``h`` the log price moves exactly by::

    d ln S = (mu - sigma^2/2 - lambda*k) h + sigma*sqrt(h)*Z + sum_{i<=N_h} ln J_i,

``N_h ~ Poisson(lambda*h)``. Sampling this law on every grid step (paths) or
once over ``[0, T]`` (terminal) has no time-discretization error at the grid
points. Quantities that depend on the path between grid points are not exact.

* Merton (1976): ``ln J ~ N(m_J, delta^2)``, ``k = expm1(m_J + delta^2/2)``;
  given ``N = n``, the jump sum is ``n*m_J + delta*sqrt(n)*Z'``.
* Kou (2002): ``ln J ~ +Exp(eta_up)`` with probability ``p_up`` and
  ``-Exp(eta_down)`` otherwise, ``eta_up > 1`` so that ``E[J] < inf``;
  ``k = p*eta_up/(eta_up-1) + (1-p)*eta_down/(eta_down+1) - 1``. Given
  ``N = n``: ``U ~ Binomial(n, p)`` upward jumps, whose sum is
  ``Gamma(shape=U, scale=1/eta_up)``, minus a
  ``Gamma(shape=n-U, scale=1/eta_down)`` downward sum (shape 0 gives 0).

References: R. C. Merton (1976), "Option pricing when underlying stock returns
are discontinuous", J. Financial Economics 3:125-144; S. G. Kou (2002), "A jump
diffusion model for option pricing", Management Science 48(8):1086-1101.
The formulas here were derived and checked numerically; equations and pages
were not checked against the originals (source inaccessible in this session).

RNG streams (plan section 2.3): ``np.random.SeedSequence(seed).spawn(3)``
gives, in order, ``[diffusion, counts, sizes]`` generators (NumPy
``default_rng``/PCG64). ``diffusion`` draws the normals ``Z``, ``counts`` the
Poisson ``N``, ``sizes`` the jump sizes (Merton: ``Z'`` only where ``N > 0``;
Kou: the binomial ups, then the up gammas, then the down gammas, only where
``N > 0``). ``seed=None`` uses fresh OS entropy. ``self.rng`` is the
diffusion generator. With ``lambda = 0`` neither the counts nor the sizes
stream is touched, and when ``delta = 0`` (Merton) no sizes are drawn. Calls
advance the instance's generators; a new instance with the same seed and
the same call sequence reproduces the result, but no equality is promised
across NumPy versions, chunk/batch sizes, sample formats, or with
``GbmSimulator`` for the same seed (different streams).
"""

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from simulation.base import BaseStochasticSimulator

# Flat chunk (elements) for path increments. Scratch per chunk is at most
# _SCRATCH_VECTORS float64/int64 vectors of this length (~3 MiB), independent
# of steps * n_sims; it is added to the memory check explicitly.
_PATH_CHUNK = 65_536
# Upper bound on chunk/batch-sized scratch vectors alive at once: counts,
# nonzero index, nonzero counts, and the jump sizes (Kou: ups, up gammas,
# down gammas; Merton uses fewer).
_SCRATCH_VECTORS = 6


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number (bool rejected)")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be representable as a finite float") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class _JumpDiffusionSimulator(BaseStochasticSimulator):
    """Shared exact sampler; subclasses define ``k`` and the jump-sum law."""

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None, *, mu: float, sigma: float,
                 jump_intensity: float) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed)
        self.mu: float = _finite_real("mu", mu)
        self.sigma: float = _finite_real("sigma", sigma)
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        self.jump_intensity: float = _finite_real("jump_intensity", jump_intensity)
        if self.jump_intensity < 0:
            raise ValueError("jump_intensity must be nonnegative")
        diffusion, counts, sizes = np.random.SeedSequence(self.seed).spawn(3)
        self.diffusion_rng = np.random.default_rng(diffusion)
        self.count_rng = np.random.default_rng(counts)
        self.size_rng = np.random.default_rng(sizes)
        self.rng = self.diffusion_rng

    # -- model hooks -------------------------------------------------------
    @property
    def jump_compensator(self) -> float:
        """``k = E[J - 1]`` (dimensionless)."""
        raise NotImplementedError

    def _add_jump_sums(self, counts: np.ndarray, out: np.ndarray) -> None:
        """Add the sum of ``counts[i]`` i.i.d. log-jumps to ``out[i]``."""
        raise NotImplementedError

    # -- shared machinery --------------------------------------------------
    def calibrate(self, historical_prices: pd.Series) -> None:
        raise NotImplementedError(
            "jump-diffusion calibration lives in calibration.jumps (plan E4.3); "
            "fit the parameters there and pass them to the constructor"
        )

    def _drift(self, h: float) -> tuple[float, float]:
        """Return ``((mu - sigma^2/2 - lambda*k)*h, sigma*sqrt(h))``."""
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scale = float(self.sigma * np.sqrt(h))
            # Scale before squaring: sigma**2 may overflow when sigma**2*h won't.
            drift = (self.mu * h - (0.5 * scale) * scale
                     - (self.jump_intensity * h) * self.jump_compensator)
        if not math.isfinite(drift) or not math.isfinite(scale):
            raise ValueError("log-return drift or diffusion scale exceeds finite precision")
        return drift, scale

    def _poisson(self, lam: float, size: int) -> np.ndarray:
        try:
            return self.count_rng.poisson(lam, size=size)
        except ValueError as exc:
            raise ValueError(
                f"jump_intensity * time = {lam} is too large for Poisson sampling"
            ) from exc

    def _fill_log_increments(self, out: np.ndarray, h: float) -> None:
        """Fill the 1-D view ``out`` with i.i.d. exact log increments over ``h``."""
        drift, scale = self._drift(h)
        lam = self.jump_intensity * h
        self.diffusion_rng.standard_normal(size=out.shape, out=out)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            out *= scale
            out += drift
            if lam > 0:
                self._add_jump_sums(self._poisson(lam, out.size), out)

    @staticmethod
    def _to_prices(log_values: np.ndarray, s0: float, what: str) -> None:
        """In place: ``log_values`` (log returns) -> ``s0 * exp(...)``."""
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            # Add ln s0 first: exp of the log price cannot overflow when the
            # final price is representable.
            log_values += math.log(s0)
            np.exp(log_values, out=log_values)
        # Scalar reductions: NaN and +/-inf fail; 0 means underflow.
        low, high = np.min(log_values), np.max(log_values)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0:
            raise ValueError(f"simulated {what} exceed finite positive float64 precision")

    def generate_paths(self) -> np.ndarray:
        """Return ``float64 (steps + 1, n_sims)`` prices; row 0 equals ``s0``.

        Each step adds the exact log increment over ``dt = horizon/steps``.
        Increments are written straight into ``paths[1:]`` (contiguous) in
        flat chunks of ``_PATH_CHUNK`` elements, cumulated in place along
        axis 0 (no copy), shifted by ``ln s0`` and exponentiated in place.
        Full ``(steps, n_sims)`` temporaries: **0**. Per-chunk scratch of at
        most ``_SCRATCH_VECTORS`` vectors of ``_PATH_CHUNK`` elements is added
        to the budget. Non-finite, zero (underflow) or overflowing prices
        raise ``ValueError``.
        """
        # Zero full-size temporaries (see docstring); chunk scratch below.
        self._check_memory(temporary_arrays=0)
        total = self.steps * self.n_sims
        chunk = min(_PATH_CHUNK, total)
        required = 8 * (self.steps + 1) * self.n_sims + 8 * _SCRATCH_VECTORS * chunk
        if required > self.MAX_WORKING_BYTES:
            raise ValueError(
                f"path arrays require {required} bytes; "
                f"limit is {self.MAX_WORKING_BYTES} bytes"
            )
        self._drift(self.dt)  # Validate before allocating.

        paths = np.empty((self.steps + 1, self.n_sims), dtype=np.float64)
        paths[0].fill(self.s0)
        increments = paths[1:]
        flat = increments.reshape(-1)  # View: rows 1..steps are contiguous.
        assert np.shares_memory(flat, paths)
        for start in range(0, total, chunk):
            self._fill_log_increments(flat[start:start + chunk], self.dt)
        with np.errstate(over="ignore", invalid="ignore"):
            np.cumsum(increments, axis=0, out=increments)
        self._to_prices(increments, self.s0, "prices")
        return paths

    def generate_terminal(self, *, batch_size: int = 10_000,
                          max_working_bytes: int = 256 * 1024 * 1024) -> np.ndarray:
        """Return exact terminal prices ``float64 (n_sims,)``.

        ``ln S_T = ln s0 + (mu - sigma^2/2 - lambda*k)T + sigma*sqrt(T)*Z
        + jump sum over N_T ~ Poisson(lambda*T)``; ``steps`` is unused.
        Each batch is a view of the output filled in place. The budget,
        checked before any allocation or sampling, is ``8*n_sims`` for the
        output plus ``8*_SCRATCH_VECTORS*min(batch_size, n_sims)`` for the
        batch scratch (an upper bound). It excludes Python/NumPy overhead
        and is not an RSS limit. Results depend on ``batch_size``.
        """
        for name, value in (("batch_size", batch_size),
                            ("max_working_bytes", max_working_bytes)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        batch = min(int(batch_size), self.n_sims)
        required = 8 * self.n_sims + 8 * _SCRATCH_VECTORS * batch
        if required > max_working_bytes:
            raise ValueError(
                f"terminal array requires {required} bytes; "
                f"limit is {max_working_bytes} bytes"
            )
        self._drift(self.horizon)  # Validate before allocating.
        terminal = np.empty(self.n_sims, dtype=np.float64)
        for start in range(0, self.n_sims, batch):
            view = terminal[start:start + batch]
            self._fill_log_increments(view, self.horizon)
            self._to_prices(view, self.s0, "terminal prices")
        return terminal


class MertonJumpSimulator(_JumpDiffusionSimulator):
    """Merton jump-diffusion with ``ln J ~ N(jump_mean, jump_std**2)``.

    Parameters (keyword-only after ``seed``): ``mu`` price drift (year^-1),
    ``sigma >= 0`` (year^-1/2), ``jump_intensity >= 0`` (jumps per year),
    ``jump_mean`` and ``jump_std >= 0`` of the log-jump (dimensionless).
    ``jump_std = 0`` gives deterministic jumps (no size draws). See the module
    docstring for the exact increments and RNG streams.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float, sigma: float,
                 jump_intensity: float, jump_mean: float, jump_std: float) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed, mu=mu, sigma=sigma,
                         jump_intensity=jump_intensity)
        self.jump_mean: float = _finite_real("jump_mean", jump_mean)
        self.jump_std: float = _finite_real("jump_std", jump_std)
        if self.jump_std < 0:
            raise ValueError("jump_std must be nonnegative")
        try:
            k = math.expm1(self.jump_mean + 0.5 * self.jump_std * self.jump_std)
        except OverflowError:
            k = math.inf
        if not math.isfinite(k):
            raise ValueError("jump compensator expm1(jump_mean + jump_std^2/2) overflows")
        self._k = k

    @property
    def jump_compensator(self) -> float:
        return self._k

    def _add_jump_sums(self, counts: np.ndarray, out: np.ndarray) -> None:
        index = np.flatnonzero(counts)
        if index.size == 0:
            return
        n = counts[index].astype(np.float64)
        jumps = n * self.jump_mean
        if self.jump_std > 0:
            z = self.size_rng.standard_normal(size=index.size)
            np.sqrt(n, out=n)
            n *= z
            n *= self.jump_std
            jumps += n
        out[index] += jumps


class KouJumpSimulator(_JumpDiffusionSimulator):
    """Kou double-exponential jump-diffusion.

    ``ln J = +Exp(eta_up)`` with probability ``p_up`` (mean ``1/eta_up``),
    else ``-Exp(eta_down)``. Requires ``p_up`` in ``[0, 1]``,
    ``eta_up > 1`` (``E[J]`` is infinite for ``eta_up <= 1``; rejected even
    when ``p_up = 0`` to keep one contract) and ``eta_down > 0``. Other
    parameters and units as in ``MertonJumpSimulator``.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float, sigma: float,
                 jump_intensity: float, p_up: float, eta_up: float,
                 eta_down: float) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed, mu=mu, sigma=sigma,
                         jump_intensity=jump_intensity)
        self.p_up: float = _finite_real("p_up", p_up)
        self.eta_up: float = _finite_real("eta_up", eta_up)
        self.eta_down: float = _finite_real("eta_down", eta_down)
        if not 0 <= self.p_up <= 1:
            raise ValueError("p_up must lie in [0, 1]")
        if self.eta_up <= 1:
            raise ValueError("eta_up must exceed 1 (E[J] is infinite otherwise)")
        if self.eta_down <= 0:
            raise ValueError("eta_down must be positive")
        # Algebraically equal to p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1,
        # without the cancellation of the "- 1".
        k = self.p_up / (self.eta_up - 1) - (1 - self.p_up) / (self.eta_down + 1)
        if not math.isfinite(k):
            raise ValueError("jump compensator exceeds finite precision")
        self._k = k

    @property
    def jump_compensator(self) -> float:
        return self._k

    def _add_jump_sums(self, counts: np.ndarray, out: np.ndarray) -> None:
        index = np.flatnonzero(counts)
        if index.size == 0:
            return
        n = counts[index]
        ups = self.size_rng.binomial(n, self.p_up)
        up_sum = self.size_rng.gamma(ups, 1.0 / self.eta_up)
        n -= ups  # Downward jump counts.
        down_sum = self.size_rng.gamma(n, 1.0 / self.eta_down)
        up_sum -= down_sum
        out[index] += up_sum
