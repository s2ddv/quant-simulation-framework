"""GARCH(1,1) with standardized Student-t innovations as a path simulator.

Model (discrete time, one step = one trading day, decimal units)::

    r_t     = mu + a_t + e_t           (daily log return, ln S_t - ln S_{t-1})
    e_t     = sqrt(h_t) * z_t
    z_t     = T_nu * sqrt((nu - 2) / nu),  T_nu ~ Student-t(nu)  (unit variance)
    h_{t+1} = omega + alpha * e_t**2 + beta * h_t
    h_1     = h0                       (variance of the first simulated day)

``a_t`` is an optional per-step drift adjustment (``_drift_adjustment``),
zero by default; it is the extension point for a future risk-neutral rule
(plan E3.9.0/E3.9.1), which is **not implemented** here. The simulated law is
the physical measure P of the fitted model.

Units and conventions:
    * ``mu`` is the **daily log-return location** (decimal per trading day).
      It is NOT the annual price drift of the GBM contract
      (``E[S_T] != s0 * exp(mu * T)``); with t innovations ``E[S_T]`` is in
      fact infinite, because the t distribution has no moment generating
      function.
    * ``omega`` and ``h`` are daily variances (decimal^2 per trading day);
      ``alpha``, ``beta`` are dimensionless; ``nu`` > 2 is the t degrees of
      freedom.
    * ``steps`` is a number of trading days and ``horizon`` (years) must equal
      ``steps / TRADING_DAYS`` (252). The model is defined on the daily grid,
      so the simulation is exact on that grid (no discretization error);
      intraday quantities are not modelled.
    * Stationarity requires ``alpha + beta < 1``; the unconditional daily
      variance is ``omega / (1 - alpha - beta)``.

The recursion, parameter units and draw order are those of
``backtest.garch`` (``simulate_cumulative``), which is imported, never
modified: for the same ``Generator`` state the cumulative log returns are
bit-identical to ``simulate_cumulative`` (see ``generate_paths``).

References:
    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    Bollerslev, T. (1987). A conditionally heteroskedastic time series model
    for speculative prices and rates of return. Review of Economics and
    Statistics 69(3), 542-547 (Student-t innovations).
    Internal reference: ``src/backtest/garch.py`` (MLE and cumulative sampler).
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from backtest.garch import fit_garch_t
from calibration.estimation import TRADING_DAYS, log_returns
from simulation.base import BaseStochasticSimulator

# Work vectors of length n_sims alive in one step of generate_paths: h, z, e,
# the increment, the running log sum and the e*e product, plus about two
# vectors allocated inside Generator.standard_t (normal and gamma draws).
_WORK_VECTORS = 8
_RNG_CONVENTION = ("NumPy default_rng/PCG64; single stream; standard_t per step "
                   "(same order as backtest.garch.simulate_cumulative)")
_LOG_MAX = math.log(np.finfo(np.float64).max)
_LOG_TINY = math.log(np.finfo(np.float64).tiny)


def _real(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, rejecting bool and non-reals."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def validate_garch_params(mu: Any, omega: Any, alpha: Any, beta: Any,
                          nu: Any) -> tuple[float, float, float, float, float]:
    """Validate GARCH(1,1)-t parameters in daily decimal units.

    Requires finite reals with ``omega > 0``, ``alpha >= 0``, ``beta >= 0``,
    ``alpha + beta < 1`` (covariance stationarity) and ``nu > 2`` (finite
    innovation variance). Returns the values as floats.
    """
    mu_f = _real(mu, "mu")
    omega_f = _real(omega, "omega")
    alpha_f = _real(alpha, "alpha")
    beta_f = _real(beta, "beta")
    nu_f = _real(nu, "nu")
    if omega_f <= 0:
        raise ValueError("omega must be positive (daily variance intercept)")
    if alpha_f < 0:
        raise ValueError("alpha must be nonnegative")
    if beta_f < 0:
        raise ValueError("beta must be nonnegative")
    if alpha_f + beta_f >= 1:
        raise ValueError(
            "alpha + beta must be below 1; alpha + beta >= 1 is not covariance "
            "stationary (no finite unconditional variance)"
        )
    if nu_f <= 2:
        raise ValueError("nu must be greater than 2; nu <= 2 has infinite variance")
    return mu_f, omega_f, alpha_f, beta_f, nu_f


def _validate_h0(h0: Any) -> float:
    h0_f = _real(h0, "h0")
    if h0_f <= 0:
        raise ValueError("h0 must be positive (daily conditional variance)")
    return h0_f


class GarchTSimulator(BaseStochasticSimulator):
    """Daily GARCH(1,1)-t price paths ``(steps + 1, n_sims)``; row zero is ``s0``.

    Parameters are in daily decimal units (see the module docstring); ``mu``
    is the daily log-return location, not an annual price drift. ``omega``,
    ``alpha``, ``beta`` and ``nu`` are supplied together or left ``None``
    until ``calibrate``. ``h0`` is the conditional variance of the first
    simulated day; when ``None`` the unconditional variance
    ``omega / (1 - alpha - beta)`` is used and recorded as its origin.
    ``calibrate`` replaces all parameters and sets ``h0`` to the filtered
    one-step-ahead variance after the sample.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float = 0.0,
                 omega: float | None = None, alpha: float | None = None,
                 beta: float | None = None, nu: float | None = None,
                 h0: float | None = None) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed)
        expected = self.steps / TRADING_DAYS
        if abs(self.horizon - expected) > 1e-12 * expected:
            raise ValueError(
                f"GARCH steps are trading days: horizon must equal steps / "
                f"{TRADING_DAYS} = {expected!r} years, got {self.horizon!r} "
                f"(use GarchTSimulator.from_days)"
            )
        self.mu: float = _real(mu, "mu")
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None
        self.nu: float | None = None
        self.fit_result: dict[str, Any] | None = None
        given = [p is not None for p in (omega, alpha, beta, nu)]
        if any(given) and not all(given):
            raise ValueError("omega, alpha, beta and nu must be supplied together")
        if all(given):
            (self.mu, self.omega, self.alpha, self.beta,
             self.nu) = validate_garch_params(self.mu, omega, alpha, beta, nu)
        self._h0: float | None = None if h0 is None else _validate_h0(h0)
        self._h0_source: str | None = None if h0 is None else "user_supplied"

    @classmethod
    def from_days(cls, s0: float, days: int, n_sims: int, seed: int | None = None,
                  **params: Any) -> GarchTSimulator:
        """Build with ``horizon = days / TRADING_DAYS`` and ``steps = days``."""
        if isinstance(days, bool) or not isinstance(days, Integral) or days <= 0:
            raise ValueError("days must be a positive integer")
        return cls(s0, int(days) / TRADING_DAYS, int(days), n_sims, seed, **params)

    # ------------------------------------------------------------------ state
    def _params(self) -> tuple[float, float, float, float, float]:
        if self.omega is None or self.alpha is None or self.beta is None or self.nu is None:
            raise ValueError("calibrate or supply omega, alpha, beta and nu before simulating")
        return validate_garch_params(self.mu, self.omega, self.alpha, self.beta, self.nu)

    def unconditional_variance(self) -> float:
        """Daily unconditional variance ``omega / (1 - alpha - beta)``."""
        _, omega, alpha, beta, _ = self._params()
        value = omega / (1.0 - alpha - beta)
        if not math.isfinite(value):
            raise ValueError("unconditional variance exceeds finite precision")
        return value

    @property
    def h0(self) -> float:
        """Variance of the first simulated day (unconditional if not set)."""
        if self._h0 is not None:
            return self._h0
        return self.unconditional_variance()

    @property
    def h0_source(self) -> str:
        """``"user_supplied"``, ``"calibrated_h_next"`` or ``"unconditional"``."""
        return self._h0_source if self._h0 is not None and self._h0_source else "unconditional"

    # ------------------------------------------------------------ calibration
    def calibrate(self, historical_prices: pd.Series) -> None:
        """Fit by ``backtest.garch.fit_garch_t`` on daily log returns.

        Needs at least 251 prices (250 returns). Sets ``mu``, ``omega``,
        ``alpha``, ``beta``, ``nu`` and ``h0 = fit["h_next"]`` (filtered
        variance of the day after the sample). The fit summary is stored in
        ``fit_result`` (``converged``, ``message``, ``nll``, ``n_obs``).
        Non-convergence is not hidden: the best parameters are kept, the flag
        is recorded and a ``RuntimeWarning`` is issued. Fitted values that
        violate the validation rules raise ``ValueError``.
        """
        returns = log_returns(historical_prices).to_numpy(dtype=float)
        if not np.isfinite(returns).all():
            raise ValueError("log returns exceed finite precision")
        fit = fit_garch_t(returns)
        params = validate_garch_params(fit["mu"], fit["omega"], fit["alpha"],
                                       fit["beta"], fit["nu"])
        h_next = _validate_h0(fit["h_next"])
        self.mu, self.omega, self.alpha, self.beta, self.nu = params
        self._h0, self._h0_source = h_next, "calibrated_h_next"
        self.fit_result = {
            "converged": bool(fit["converged"]), "message": str(fit["message"]),
            "nll": float(fit["nll"]), "n_obs": int(fit["n_obs"]),
        }
        if not self.fit_result["converged"]:
            warnings.warn(
                f"GARCH(1,1)-t MLE did not converge ({fit['message']}); "
                "best parameters kept, see fit_result['converged']",
                RuntimeWarning, stacklevel=2,
            )

    # ------------------------------------------------------------- extension
    def _drift_adjustment(self, step: int, h: np.ndarray) -> float | np.ndarray:
        """Per-day additive adjustment ``a_t`` to the log return (default 0.0).

        ``step`` is the 0-based day index (the return of day ``step + 1``);
        ``h`` is the read-only ``(n_sims,)`` conditional variance of that day,
        known before the shock is drawn, so any rule built on it is
        predictable. Return a scalar or an ``(n_sims,)`` array of finite
        values. Subclasses implementing a risk-neutral rule (E3.9.1) must
        not draw from ``self.rng`` here, or the stream alignment with
        ``simulate_cumulative`` is lost. No Q rule is implemented here.
        """
        return 0.0

    # -------------------------------------------------------------- sampling
    def _required_bytes(self, return_variance: bool) -> int:
        rows = (self.steps + 1) * self.n_sims
        return 8 * rows * (1 + int(return_variance)) + 8 * _WORK_VECTORS * self.n_sims

    def _check_garch_memory(self, return_variance: bool) -> None:
        """Output matrix (+ variance matrix) and ``n_sims``-length work vectors.

        The recursion is sequential in time, so no ``(steps, n_sims)``
        temporary is allocated; ``_check_memory`` counts temporaries of that
        size, hence this specialised estimate.
        """
        output_bytes = 8 * (self.steps + 1) * self.n_sims
        required = self._required_bytes(return_variance)
        if required > self.MAX_WORKING_BYTES:
            raise ValueError(
                f"path arrays require {required} bytes (output {output_bytes}); "
                f"limit is {self.MAX_WORKING_BYTES} bytes"
            )

    def _simulate_log(self, return_variance: bool = False
                      ) -> tuple[np.ndarray, np.ndarray | None]:
        """Cumulative log returns ``(steps + 1, n_sims)`` (row 0 zero) and h.

        Row ``k`` is ``ln(S_k / s0)``, accumulated in float64 with the same
        arithmetic as ``simulate_cumulative``: ``total += mu + e`` per day.
        The optional variance matrix has row ``k`` = conditional variance of
        the return of day ``k + 1`` (row 0 = ``h0``, last row = variance of
        the day after the horizon).
        """
        mu, omega, alpha, beta, nu = self._params()
        h0 = self.h0
        self._check_garch_memory(return_variance)
        sd = np.sqrt((nu - 2) / nu)
        n = self.n_sims
        out = np.empty((self.steps + 1, n), dtype=np.float64)
        out[0].fill(0.0)
        hvar = np.empty((self.steps + 1, n), dtype=np.float64) if return_variance else None
        h = np.full(n, h0)
        total = np.zeros(n)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for step in range(self.steps):
                if hvar is not None:
                    hvar[step] = h
                h_view = h.view()
                h_view.flags.writeable = False
                adj = self._drift_adjustment(step, h_view)
                z = self.rng.standard_t(nu, size=n) * sd
                e = np.sqrt(h) * z
                if isinstance(adj, (int, float)) and not isinstance(adj, bool) and adj == 0.0:
                    total += mu + e
                else:
                    adj_arr = np.asarray(adj, dtype=np.float64)
                    if adj_arr.shape not in ((), (n,)) or not np.isfinite(adj_arr).all():
                        raise ValueError(
                            "_drift_adjustment must return a finite scalar or (n_sims,) array"
                        )
                    total += mu + e + adj_arr
                out[step + 1] = total
                h = omega + alpha * e * e + beta * h
            if hvar is not None:
                hvar[self.steps] = h
        if not np.isfinite(out).all() or (hvar is not None and not np.isfinite(hvar).all()):
            raise ValueError(
                "GARCH path exceeds finite precision (conditional variance or "
                "log price overflowed); reduce omega, h0 or the horizon"
            )
        return out, hvar

    def _prices(self, return_variance: bool) -> tuple[np.ndarray, np.ndarray | None]:
        cum, hvar = self._simulate_log(return_variance)
        increments = cum[1:]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if (np.max(increments) >= _LOG_MAX or np.min(increments) <= _LOG_TINY):
                increments += math.log(self.s0)
                np.exp(increments, out=increments)
            else:
                np.exp(increments, out=increments)
                increments *= self.s0
        if not np.isfinite(increments).all() or np.min(increments) <= 0:
            raise ValueError("GARCH prices overflow or underflow float64")
        cum[0].fill(self.s0)
        return cum, hvar

    def generate_paths(self) -> np.ndarray:
        """Return float64 prices ``(steps + 1, n_sims)`` with row zero ``s0``.

        One ``self.rng`` stream; each day draws
        ``rng.standard_t(nu, size=n_sims) * sqrt((nu - 2) / nu)`` in the same
        order as ``backtest.garch.simulate_cumulative``. With the default
        (zero) drift adjustment the cumulative log returns are bit-identical
        to ``simulate_cumulative(params, h0, steps, n_sims, default_rng(seed))``;
        ``log(paths[-1] / s0)`` matches it only up to the rounding of
        ``exp``/``log`` (a few ulp), because prices are ``s0 * exp(cum)``.
        Repeated calls advance ``self.rng``.

        Peak memory is the output and about ``8 * n_sims`` float64 work
        vectors; no ``(steps, n_sims)`` temporary is created. Non-finite
        variances or prices, or prices that underflow to zero, raise
        ``ValueError``.
        """
        paths, _ = self._prices(False)
        return paths

    def generate_paths_and_variance(self) -> tuple[np.ndarray, np.ndarray]:
        """Prices as ``generate_paths`` plus the conditional variance matrix.

        Both are float64 ``(steps + 1, n_sims)``. Variance row ``k`` is the
        daily conditional variance of the return of day ``k + 1``: row 0 is
        ``h0`` and the last row is the variance of the day after the horizon.
        The variance matrix is included in the memory estimate. Draws are
        identical to ``generate_paths`` for the same generator state.
        """
        paths, hvar = self._prices(True)
        assert hvar is not None
        return paths, hvar

    # -------------------------------------------------------------- metadata
    def model_info(self) -> dict[str, Any]:
        """JSON-serializable description (``json.dumps(..., allow_nan=False)``)."""
        calibrated = None not in (self.omega, self.alpha, self.beta, self.nu)
        info: dict[str, Any] = {
            "model": "garch11_t",
            "mu": self.mu,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "nu": self.nu,
            "mu_meaning": "daily log-return location (decimal per trading day), "
                          "not annual price drift",
            "variance_unit": "decimal^2 per trading day",
            "time_unit": "trading_day",
            "trading_days_per_year": TRADING_DAYS,
            "steps": self.steps,
            "horizon_years": self.horizon,
            "s0": self.s0,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "h0": self.h0 if calibrated or self._h0 is not None else None,
            "h0_source": self.h0_source if calibrated or self._h0 is not None else None,
            "unconditional_variance_daily": (self.unconditional_variance()
                                             if calibrated else None),
            "exact_on_grid": True,
            "rng": _RNG_CONVENTION,
            "numpy_version": np.__version__,
            "measure": "physical",
            "risk_neutralization": "not_implemented",
            "fit": None if self.fit_result is None else dict(self.fit_result),
        }
        return info
