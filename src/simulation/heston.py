"""Heston (1993) stochastic-volatility paths: Andersen QE or full truncation.

Dynamics (constant parameters)::

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,      d<W1, W2> = rho dt

Units: price in the asset's currency; time in years; ``mu`` (price drift)
and ``kappa`` in year^-1; ``v``, ``v0`` and ``theta`` (instantaneous variance)
in year^-1; ``xi`` (volatility of variance) in year^-1; ``rho``
dimensionless. ``mu`` is the price drift of the physical measure P: with the
martingale schemes below ``E[S_T] = s0 exp(mu T)``. There is no ``r``/``q``:
for risk-neutral pricing pass ``mu = r - q`` (a reparametrisation; the
volatility risk premium is then assumed absorbed in the supplied
``kappa``/``theta``) and discount outside. The Feller condition
``2 kappa theta >= xi^2`` is reported (``feller``, ``model_info``), never
required.

**THERE IS DISCRETIZATION ERROR.** Unlike the GBM engine, neither scheme
samples the exact law of ``(S, v)`` on the grid (``exact_on_grid: False``).
The bias shrinks with ``dt = horizon / steps``; it is not removed by more
paths.

Schemes (``scheme`` is explicit):

* ``"andersen_qe"`` (Andersen 2008, "Simple and efficient simulation of the
  Heston stochastic volatility model", J. Computational Finance 11(3):1-42,
  sections "QE scheme", "log-price discretization" and "martingale
  correction"). With ``E = e^{-kappa dt}``::

      m   = theta (1 - E) + v E
      s2  = v xi^2 E (1 - E)/kappa + theta xi^2 (1 - E)^2/(2 kappa)
      psi = s2 / m^2
      psi <= psi_c = 1.5:  b^2 = 2/psi - 1 + sqrt(2/psi) sqrt(2/psi - 1),
                           a = m / (1 + b^2),   v' = a (b + Z_v)^2
      psi >  psi_c:        p = (psi - 1)/(psi + 1),  beta = (1 - p)/m,
                           v' = 0 if U <= p else ln((1 - p)/(1 - U)) / beta

  The conditional mean and variance of ``v'`` match the CIR ones, so
  ``E[v_t]`` and ``Var[v_t]`` are exact on the grid (other features of the
  law are not). Log price with ``gamma1 = gamma2 = 1/2``::

      K0 = -rho kappa theta dt/xi
      K1 = gamma1 dt (kappa rho/xi - 1/2) - rho/xi
      K2 = gamma2 dt (kappa rho/xi - 1/2) + rho/xi
      K3 = gamma1 dt (1 - rho^2),   K4 = gamma2 dt (1 - rho^2)
      ln S' = ln S + mu dt + K0 + K1 v + K2 v' + sqrt(K3 v + K4 v') Z

  ``martingale_correction=True`` (QE-M) replaces ``K0`` per path by ``K0*``
  so that ``E[S' | S, v] = S e^{mu dt}`` exactly. With ``A = K2 + K4/2``::

      quadratic:   K0* = -A b^2 a/(1 - 2 A a) + ln(1 - 2 A a)/2 - (K1 + K3/2) v,  needs A < 1/(2a)
      exponential: K0* = -ln(p + beta (1 - p)/(beta - A)) - (K1 + K3/2) v,          needs A < beta

  ``rho <= 0`` gives ``A <= 0`` and the conditions always hold. If they fail
  on any path at any step a ``ValueError`` names the step and suggests a
  smaller ``dt`` or ``martingale_correction=False``; there is no silent
  fallback. Without the correction ``E[S_T]`` is only approximately
  ``s0 e^{mu T}``.

* ``"full_truncation"`` (Lord, Koekkoek and van Dijk 2010, "A comparison of
  biased simulation schemes for stochastic volatility models", Quantitative
  Finance 10(2):177-194), Euler for ``v`` and log-Euler for ``S``::

      v~' = v~ + kappa (theta - v~+) dt + xi sqrt(v~+) dW_v
      ln S' = ln S + (mu - v~+/2) dt + sqrt(v~+) dW_S,
      dW_S = rho dW_v + sqrt(1 - rho^2) dW_perp,   v~+ = max(v~, 0)

  The auxiliary ``v~`` (possibly negative) is propagated; the variance matrix
  returns ``v~+``. Log-Euler is a discrete martingale step by step
  (``E[S'|S] = S e^{mu dt}``), so ``martingale_correction`` does not apply
  (recorded as ``"not_applicable"``). The bias is much larger than QE-M's.

* ``xi == 0`` (both schemes): exact deterministic branch. ``v`` follows the
  ODE ``v' = theta + (v - theta) E`` and ``ln S`` moves by a Gaussian with
  the step's exact integrated variance ``I = theta dt + (v - theta)(1 - E)/kappa``:
  ``ln S' = ln S + mu dt - I/2 + sqrt(I) Z``. This branch is exact on the
  grid. ``"full_truncation"`` accepts any ``xi > 0``. ``"andersen_qe"``
  rejects ``0 < xi < QE_XI_MIN = 1e-4`` with ``ValueError``: ``K1`` and
  ``K2`` contain ``-/+ rho/xi`` and the log-price step cancels
  ``(rho/xi)(v' - v)`` against the ``K0`` terms, so a tiny ``xi`` silently
  loses the martingale property (measured: ``xi = 1e-14``, ``rho = 0.9``
  gave ``E[S_T] - S0 = +0.33``, 5 SE). Use ``xi = 0`` (exact) instead. The
  threshold equals ``validation.heston_theory.XI_MIN`` (replicated here so
  that ``simulation`` does not import ``validation``; a test checks that
  the two agree).

RNG streams (plan section 2.3): ``np.random.SeedSequence(seed).spawn(3)``
gives NumPy ``default_rng``/PCG64 generators ``[0] variance_rng`` (``Z_v`` of
QE, ``dW_v / sqrt(dt)`` of full truncation), ``[1] uniform_rng`` (``U`` of the
QE exponential branch, ``random()`` in ``[0, 1)``), ``[2] price_rng`` (``Z`` of
the QE log price, ``dW_perp / sqrt(dt)`` of full truncation). Per step, QE
draws ``Z_v`` and ``U`` for **all** paths of the batch (even though each path
uses only one of them) and then ``Z``; full truncation draws ``Z_v`` then
``Z`` and never touches ``uniform_rng``; ``xi == 0`` draws only ``Z``. The
generators are created in ``__init__`` and advance between calls. The base
class ``self.rng`` is not used by this simulator. Same seed does not promise
equality across NumPy versions, batch sizes or sample formats
(``generate_paths`` vs ``generate_terminal``).
"""

from __future__ import annotations

import math
import platform
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from simulation.base import BaseStochasticSimulator

SCHEMES = ("andersen_qe", "full_truncation")
PSI_C = 1.5
#: Smallest positive ``xi`` accepted by the QE scheme (module docstring);
#: equal to ``validation.heston_theory.XI_MIN``.
QE_XI_MIN = 1e-4
GAMMA1 = 0.5
GAMMA2 = 0.5
# Bound on float64-equivalent vectors of batch length alive at once: state
# (log price, v), the draws (Z_v, U, Z), the QE temporaries (m, s2, psi, b2,
# a, p, beta, K0*, branch results) and the boolean masks (quad, u <= p, the
# per-step finiteness check). tracemalloc measured at most ~25.1 for QE
# (``generate_paths`` with steps = 1 and ``generate_terminal``) and ~9 for
# full truncation (NumPy 2.x, CPython 3.13); 32 leaves ~7 vectors of slack.
# No (steps + 1, n_sims) boolean matrix is ever allocated: finiteness is
# checked per step on the state and by max/min reductions on the output.
_WORK_VECTORS = 32
_DEFAULT_MAX_BATCH = 100_000
_RNG_CONVENTION = ("NumPy default_rng/PCG64; SeedSequence(seed).spawn(3): "
                   "[0] variance normals, [1] QE uniforms, [2] price normals")
_LOG_MAX = math.log(np.finfo(np.float64).max)
_LOG_TINY = math.log(np.finfo(np.float64).tiny)


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


class HestonSimulator(BaseStochasticSimulator):
    """Heston price paths ``(steps + 1, n_sims)``; row zero is ``s0``.

    See the module docstring for the dynamics, units, schemes, RNG streams
    and the discretization-error caveat. Parameters are supplied by the
    user; ``calibrate`` is not implemented.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float, v0: float, kappa: float,
                 theta: float, xi: float, rho: float, scheme: str = "andersen_qe",
                 martingale_correction: bool = True) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed)
        self.mu: float = _finite_real("mu", mu)
        self.v0: float = _finite_real("v0", v0)
        self.kappa: float = _finite_real("kappa", kappa)
        self.theta: float = _finite_real("theta", theta)
        self.xi: float = _finite_real("xi", xi)
        self.rho: float = _finite_real("rho", rho)
        if self.v0 < 0:
            raise ValueError("v0 must be nonnegative")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.xi < 0:
            raise ValueError("xi must be nonnegative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if not isinstance(scheme, str) or scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if not isinstance(martingale_correction, (bool, np.bool_)):
            raise ValueError("martingale_correction must be a bool")
        if scheme == "andersen_qe" and 0.0 < self.xi < QE_XI_MIN:
            raise ValueError(
                f"0 < xi < {QE_XI_MIN:g} is ill conditioned for the andersen_qe "
                "scheme (K1, K2 contain rho/xi and cancel; the martingale property "
                f"is lost); use xi = 0 (exact deterministic variance), xi >= "
                f"{QE_XI_MIN:g} or scheme='full_truncation'")
        self.scheme: str = scheme
        self.martingale_correction: bool = bool(martingale_correction)

        dt = self.dt
        with np.errstate(over="ignore", under="ignore"):
            self._e = math.exp(-self.kappa * dt)
            self._one_minus_e = -math.expm1(-self.kappa * dt)
            drift = self.mu * dt
        if not (math.isfinite(self._one_minus_e) and math.isfinite(drift)
                and self._one_minus_e > 0.0):
            raise ValueError("kappa * dt or mu * dt is outside the representable range")
        self._drift = drift
        if self.xi > 0.0:
            k = self.kappa
            x = self.xi
            r = self.rho
            ratio = self._one_minus_e / k  # (1 - E)/kappa
            self._s2_v = x * x * self._e * ratio
            self._s2_c = self.theta * x * x * self._one_minus_e * ratio / 2.0
            self._k0 = -r * k * self.theta * dt / x
            self._k1 = GAMMA1 * dt * (k * r / x - 0.5) - r / x
            self._k2 = GAMMA2 * dt * (k * r / x - 0.5) + r / x
            self._k3 = GAMMA1 * dt * (1.0 - r * r)
            self._k4 = GAMMA2 * dt * (1.0 - r * r)
            self._a_mg = self._k2 + self._k4 / 2.0
            consts = (self._s2_v, self._s2_c, self._k0, self._k1, self._k2,
                      self._k3, self._k4, self._a_mg, x * math.sqrt(dt))
            if not all(math.isfinite(c) for c in consts):
                raise ValueError("Heston scheme constants exceed finite precision")

        variance_seq, uniform_seq, price_seq = np.random.SeedSequence(self.seed).spawn(3)
        self.variance_rng: np.random.Generator = np.random.default_rng(variance_seq)
        self.uniform_rng: np.random.Generator = np.random.default_rng(uniform_seq)
        self.price_rng: np.random.Generator = np.random.default_rng(price_seq)

    # -------------------------------------------------------------- metadata
    @property
    def feller(self) -> dict[str, Any]:
        """``{"satisfied", "two_kappa_theta", "xi_squared", "margin"}``.

        Feller condition ``2 kappa theta >= xi^2``; reported only. Same
        result as ``validation.heston_theory.feller_condition``.
        """
        two_kt = 2.0 * self.kappa * self.theta
        xi2 = self.xi * self.xi
        return {"satisfied": bool(two_kt >= xi2), "two_kappa_theta": two_kt,
                "xi_squared": xi2, "margin": two_kt - xi2}

    def calibrate(self, historical_prices: pd.Series) -> None:
        raise NotImplementedError(
            "Heston calibration is not implemented (plan E5.2 is a research "
            "opinion only: the variance is latent); supply mu, v0, kappa, "
            "theta, xi and rho to the constructor"
        )

    def model_info(self) -> dict[str, Any]:
        """JSON-serializable description (``json.dumps(..., allow_nan=False)``)."""
        exact = self.xi == 0.0
        if self.scheme == "andersen_qe":
            variance_output = "v"
            correction: bool | str = self.martingale_correction
        else:
            variance_output = "max(v_tilde,0)"
            correction = "not_applicable"
        return {
            "model": "heston",
            "s0": self.s0,
            "horizon_years": self.horizon,
            "steps": self.steps,
            "dt_years": self.dt,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "mu": self.mu,
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "scheme": self.scheme,
            "discretization": "exact_deterministic_variance" if exact else self.scheme,
            "exact_on_grid": exact,
            "martingale_correction": correction,
            "psi_c": PSI_C,
            "gamma1": GAMMA1,
            "gamma2": GAMMA2,
            "feller": self.feller,
            "mu_meaning": "annual price drift; pass r - q for risk-neutral use",
            "units": {"time": "years", "mu": "1/year", "v0": "1/year",
                      "theta": "1/year", "kappa": "1/year", "xi": "1/year",
                      "rho": "dimensionless"},
            "rng": _RNG_CONVENTION,
            "variance_output": variance_output,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "references": [
                "Heston (1993) Review of Financial Studies 6(2):327-343",
                "Andersen (2008) J. Computational Finance 11(3):1-42 (QE, "
                "log-price discretization, martingale correction)",
                "Lord, Koekkoek, van Dijk (2010) Quantitative Finance 10(2):177-194",
            ],
        }

    # -------------------------------------------------------------- one step
    def _step(self, step: int, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Advance ``x`` (log price relative to s0) in place; return new state.

        ``v`` is the variance state (QE: ``v``; full truncation: ``v~``).
        Returns the new variance state array (a new array). ``step`` is the
        0-based step index, used only in error messages.
        """
        n = x.shape[0]
        dt = self.dt
        if self.xi == 0.0:
            iv = self.theta * dt + (v - self.theta) * (self._one_minus_e / self.kappa)
            np.maximum(iv, 0.0, out=iv)
            z = self.price_rng.standard_normal(n)
            x += self._drift - 0.5 * iv + np.sqrt(iv) * z
            return self.theta + (v - self.theta) * self._e
        if self.scheme == "full_truncation":
            zv = self.variance_rng.standard_normal(n)
            zp = self.price_rng.standard_normal(n)
            vp = np.maximum(v, 0.0)
            sq = np.sqrt(vp * dt)
            x += (self._drift - 0.5 * vp * dt
                  + sq * (self.rho * zv + math.sqrt(1.0 - self.rho * self.rho) * zp))
            return v + self.kappa * (self.theta - vp) * dt + self.xi * sq * zv
        return self._qe_step(step, x, v)

    def _qe_step(self, step: int, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        zv = self.variance_rng.standard_normal(n)
        u = self.uniform_rng.random(n)
        z = self.price_rng.standard_normal(n)
        # m = theta (1 - E) + v E: a sum of nonnegative terms, so m >= theta
        # (1 - E) > 0 without cancellation (theta + (v - theta) E rounds to 0
        # for v = 0 when kappa dt is so small that E == 1.0).
        m = self.theta * self._one_minus_e + v * self._e
        s2 = v * self._s2_v + self._s2_c
        psi = s2 / (m * m)
        quad = psi <= PSI_C
        # Quadratic branch (psi in (0, 1.5]); placeholders where unused.
        psi_q = np.where(quad, psi, 1.0)
        inv = 2.0 / psi_q
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = m / (1.0 + b2)
        # Exponential branch (psi > 1.5).
        psi_e = np.where(quad, 3.0, psi)
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / m
        with np.errstate(divide="ignore"):
            v_exp = np.where(u <= p, 0.0, np.log((1.0 - p) / (1.0 - u)) / beta)
        v_new = np.where(quad, a * (np.sqrt(b2) + zv) ** 2, v_exp)

        if self.martingale_correction:
            big_a = self._a_mg
            if big_a > 0.0:
                bad = np.where(quad, big_a >= 1.0 / (2.0 * a), big_a >= beta)
                if bad.any():
                    raise ValueError(
                        f"QE martingale correction undefined at step {step + 1} of "
                        f"{self.steps} (A = K2 + K4/2 = {big_a:g} >= 1/(2a) or beta "
                        f"on {int(bad.sum())} paths); use a smaller dt or "
                        "martingale_correction=False"
                    )
            two_aa = np.where(quad, 2.0 * big_a * a, 0.0)
            k0_q = -big_a * b2 * a / (1.0 - two_aa) + 0.5 * np.log1p(-two_aa)
            denom = np.where(quad, 1.0, beta - big_a)
            k0_e = -np.log(p + beta * (1.0 - p) / denom)
            k0: float | np.ndarray = (np.where(quad, k0_q, k0_e)
                                      - (self._k1 + 0.5 * self._k3) * v)
        else:
            k0 = self._k0
        var_term = np.maximum(self._k3 * v + self._k4 * v_new, 0.0)
        x += self._drift + k0 + self._k1 * v + self._k2 * v_new + np.sqrt(var_term) * z
        return v_new

    # -------------------------------------------------------------- memory
    def _check_bytes(self, required: int, what: str) -> None:
        if required > self.MAX_WORKING_BYTES:
            raise ValueError(
                f"{what} require {required} bytes; limit is "
                f"{self.MAX_WORKING_BYTES} bytes"
            )

    def required_bytes(self, return_variance: bool = False) -> int:
        """Estimated peak bytes of ``generate_paths``(``_and_variance``).

        Output matrix ``8 (steps + 1) n_sims`` (+ the same for the variance
        matrix) plus ``_WORK_VECTORS`` float64 vectors of length ``n_sims``
        (measured bound, including the boolean masks). No ``(steps, n_sims)``
        temporary of any dtype is allocated (finiteness is checked per step
        and by reductions).
        """
        rows = (self.steps + 1) * self.n_sims
        return 8 * rows * (1 + int(return_variance)) + 8 * _WORK_VECTORS * self.n_sims

    # -------------------------------------------------------------- sampling
    def _simulate(self, return_variance: bool) -> tuple[np.ndarray, np.ndarray | None]:
        self._check_bytes(self.required_bytes(return_variance), "Heston path arrays")
        n = self.n_sims
        out = np.empty((self.steps + 1, n), dtype=np.float64)
        out[0].fill(0.0)
        var = np.empty((self.steps + 1, n), dtype=np.float64) if return_variance else None
        if var is not None:
            var[0].fill(self.v0)
        x = np.zeros(n)
        v = np.full(n, self.v0)
        truncate = self.scheme == "full_truncation" and self.xi > 0.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for step in range(self.steps):
                v = self._step(step, x, v)
                # Row-wise checks (an n_sims bool vector, counted in
                # _WORK_VECTORS) instead of a (steps + 1, n_sims) mask.
                if not np.isfinite(x).all():
                    raise ValueError("Heston log price is not finite (overflow or "
                                     "invalid operation); reduce xi, v0, theta or "
                                     "the horizon")
                out[step + 1] = x
                if var is not None:
                    row = var[step + 1]
                    if truncate:
                        np.maximum(v, 0.0, out=row)
                    else:
                        row[...] = v
                    if not np.isfinite(row).all():
                        raise ValueError("Heston variance is not finite (overflow)")
                    if np.min(row) < 0.0:  # cannot happen: QE >= 0, FT returns v+
                        raise ValueError("Heston variance output is negative")
        return out, var

    def _prices(self, return_variance: bool) -> tuple[np.ndarray, np.ndarray | None]:
        cum, var = self._simulate(return_variance)
        increments = cum[1:]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if np.max(increments) >= _LOG_MAX or np.min(increments) <= _LOG_TINY:
                increments += math.log(self.s0)
                np.exp(increments, out=increments)
            else:
                np.exp(increments, out=increments)
                increments *= self.s0
            # max/min reductions allocate nothing; nan propagates through max.
            top = float(np.max(increments))
            bottom = float(np.min(increments))
        if not math.isfinite(top) or not bottom > 0.0:
            raise ValueError("Heston prices overflow or underflow float64")
        cum[0].fill(self.s0)
        return cum, var

    def generate_paths(self) -> np.ndarray:
        """Return float64 prices ``(steps + 1, n_sims)`` with row zero ``s0``.

        Prices are ``s0 * exp(ln(S_k/s0))`` with the log price propagated by
        the selected scheme (module docstring; there is discretization
        error). Peak memory: the output plus about ``_WORK_VECTORS`` vectors
        of length ``n_sims`` (``required_bytes``), checked against
        ``MAX_WORKING_BYTES``. Non-finite values, overflow or underflow to
        zero raise ``ValueError``. Calls advance the three generators.
        """
        paths, _ = self._prices(False)
        return paths

    def generate_paths_and_variance(self) -> tuple[np.ndarray, np.ndarray]:
        """Prices as ``generate_paths`` plus the variance matrix.

        Both float64 ``(steps + 1, n_sims)``; variance row 0 is ``v0``. Row
        ``k`` is the simulated variance at ``t_k``: QE's ``v`` (``>= 0``), or
        ``max(v~, 0)`` for full truncation (``model_info()["variance_output"]``).
        The variance matrix is counted in the memory check. A separate method
        (not ``return_variance=True``) keeps ``generate_paths`` typed as
        returning an ndarray, as in ``GarchTSimulator``. Draws are identical
        to ``generate_paths`` for the same generator states.
        """
        paths, var = self._prices(True)
        assert var is not None
        return paths, var

    def generate_terminal(self, *, batch_size: int | None = None) -> np.ndarray:
        """Return terminal prices ``float64 (n_sims,)`` without the path matrix.

        Same scheme and law as ``generate_paths`` (same ``dt`` grid), streamed
        over batches of paths: only the state ``(ln S, v)`` and the per-step
        work vectors of one batch are alive. Batch ``b`` runs all steps before
        batch ``b + 1``, so results are not bit-identical to
        ``generate_paths(...)[-1]`` for the same seed (different draw order).
        ``batch_size=None`` picks ``min(n_sims, 100_000)`` reduced to fit
        ``MAX_WORKING_BYTES``; the output ``8 n_sims`` bytes plus
        ``_WORK_VECTORS`` vectors of ``batch_size`` must fit the budget.
        """
        n = self.n_sims
        output_bytes = 8 * n
        if batch_size is None:
            room = (self.MAX_WORKING_BYTES - output_bytes) // (8 * _WORK_VECTORS)
            batch = int(min(n, _DEFAULT_MAX_BATCH, max(room, 1)))
        else:
            if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral)
                    or batch_size <= 0):
                raise ValueError("batch_size must be a positive integer or None")
            batch = int(min(batch_size, n))
        self._check_bytes(output_bytes + 8 * _WORK_VECTORS * batch,
                          "Heston terminal arrays")
        out = np.empty(n, dtype=np.float64)
        log_s0 = math.log(self.s0)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for start in range(0, n, batch):
                stop = min(start + batch, n)
                x = np.zeros(stop - start)
                v = np.full(stop - start, self.v0)
                for step in range(self.steps):
                    v = self._step(step, x, v)
                if not np.isfinite(x).all():
                    raise ValueError("Heston log price is not finite (overflow or "
                                     "invalid operation); reduce xi, v0, theta or "
                                     "the horizon")
                x += log_s0
                np.exp(x, out=out[start:stop])
        top = float(np.max(out))
        bottom = float(np.min(out))
        if not math.isfinite(top) or not bottom > 0.0:
            raise ValueError("Heston terminal prices overflow or underflow float64")
        return out
