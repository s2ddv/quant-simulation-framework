"""Maximum-likelihood calibration of the Merton jump-diffusion to daily returns.

Model (physical measure P, constant parameters, one trading day
``dt = 1/TRADING_DAYS`` years)::

    dS/S_- = (mu - lambda*k) dt + sigma dW + (J - 1) dN,   ln J ~ N(m_J, delta^2)
    k = E[J - 1] = exp(m_J + delta^2/2) - 1

so the daily log-return is exactly::

    x = a*dt + sigma*sqrt(dt)*Z + sum_{i <= N} Y_i,
    N ~ Poisson(lambda*dt),  Y_i ~ N(m_J, delta^2),  a = mu - sigma^2/2 - lambda*k

and its density is a Poisson mixture of normals::

    f(x) = sum_n Pois(n; lambda*dt) * phi(x; a*dt + n*m_J, sigma^2*dt + n*delta^2)

truncated at ``n_max = poisson.isf(1e-14, LAMBDA_MAX*dt)`` (16 for
``LAMBDA_MAX = 252``), the same ``n_max`` for every parameter value so that the
objective is a smooth function of the parameters. The neglected Poisson mass
is below 1e-14 for every admissible ``lambda <= LAMBDA_MAX``.

Units: ``mu`` (price drift, ``E[S_T] = S_0 e^{mu T}``) and ``lambda``
(jumps per year) in year^-1, ``sigma`` in year^-1/2, ``m_J`` and ``delta`` in
log-price units per jump (dimensionless). ``mu`` is the drift of the PRICE,
not the mean of the log-returns (``mean(x)/dt = a + lambda*m_J``).

Identifiability and inference caveats (documented, not corrected):

* The likelihood is unbounded as ``sigma -> 0``: the ``n = 0`` component can
  collapse onto a single observation. A floor ``sigma >= SIGMA_FLOOR``
  (1e-3 year^-1/2) is therefore imposed; multi-start does not remove the
  singularity. Likewise ``delta >= DELTA_FLOOR`` (1e-4).
* ``lambda -> 0``: ``m_J`` and ``delta`` are not identified. ``lambda*dt``
  large with small jumps: the mixture approaches a normal and essentially only
  the total variance ``sigma^2 + lambda*(m_J^2 + delta^2)`` is identified.
* The likelihood-ratio test of GBM (``lambda = 0``) against Merton is NOT
  chi-square: ``lambda = 0`` lies on the boundary of the parameter space and
  ``m_J``, ``delta`` are absent under the null (Davies' problem). AIC/BIC are
  reported only as heuristics.
* Standard errors from the inverse numerical Hessian (observed Fisher
  information) are asymptotic and invalid at or near a bound; they are then
  reported as ``None``.
* Parameters estimated under P (historical) are neither risk-neutral nor
  implied; using them for pricing requires an explicit measure assumption
  (see :mod:`pricing.merton`).

No network access and no downloads happen here; results are deterministic
given ``seed`` (NumPy ``default_rng(SeedSequence(seed))``/PCG64 draws the
random starting points; the optimizer itself is deterministic).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy
from scipy.stats import poisson

from calibration.estimation import TRADING_DAYS, log_returns
from calibration.prices import validate_prices

FloatArray = npt.NDArray[np.float64]

LAMBDA_MAX = float(TRADING_DAYS)  # year^-1: at most one jump per day on average
SIGMA_FLOOR = 1e-3                # year^-1/2, see module docstring
DELTA_FLOOR = 1e-4                # log-price units
TRUNCATION_TAIL = 1e-14           # Poisson mass neglected beyond n_max
MAX_WORKING_BYTES = 512 * 1024**2  # memory budget of one likelihood evaluation
_WORK_ARRAYS = 8                  # (n_obs, n_max + 1) float64 arrays alive at once
_LAMBDA_SCALE = 25.2              # optimizer variable u2 = lambda / _LAMBDA_SCALE
_NLL_CAP = 1e300                   # returned for a non-finite nll
_GRAD_CAP = 1e30                   # cap on |gradient| components (scaled variables)
PARAM_NAMES = ("mu", "sigma", "jump_intensity", "jump_mean", "jump_std")
_LOG_2PI = math.log(2.0 * math.pi)


def truncation_n_max(dt: float, lambda_max: float = LAMBDA_MAX,
                     tail: float = TRUNCATION_TAIL) -> int:
    """Largest jump count kept: ``P(N > n_max) <= tail`` for ``N ~ Pois(lambda_max*dt)``."""
    return int(poisson.isf(tail, lambda_max * dt))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _real(name: str, value: Any, *, positive: bool = False,
          nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number (bool is rejected)")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _returns_array(returns: Any) -> FloatArray:
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()
    arr = np.asarray(returns)
    if arr.dtype == np.bool_ or arr.dtype.kind not in "iuf":
        raise ValueError("log returns must be real numbers (bool and complex are rejected)")
    if arr.ndim != 1:
        raise ValueError("log returns must be one-dimensional")
    x = arr.astype(np.float64)
    if x.size < 2:
        raise ValueError("at least two log returns are required")
    if not np.all(np.isfinite(x)):
        raise ValueError("log returns must be finite")
    return x


def _check_memory(n_obs: int, n_max: int) -> None:
    need = 8 * n_obs * (n_max + 1) * _WORK_ARRAYS
    if need > MAX_WORKING_BYTES:
        raise ValueError(
            f"likelihood needs about {need / 1024**2:.0f} MiB "
            f"({n_obs} observations x {n_max + 1} mixture terms x {_WORK_ARRAYS} "
            f"float64 arrays), above MAX_WORKING_BYTES; use fewer observations")


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def _loglik_terms(x: FloatArray, a: float, sigma: float, lam: float, m_j: float,
                  delta: float, dt: float, n_max: int, *,
                  gradient: bool = False) -> tuple[float, FloatArray | None]:
    """Total log-likelihood and, optionally, its gradient.

    The gradient is with respect to ``(a, ln sigma, lambda, m_J, ln delta)``.
    Requires ``sigma > 0``, ``delta > 0`` and ``lambda >= 0``.
    """
    n = np.arange(n_max + 1, dtype=np.float64)
    lam_dt = lam * dt
    log_pois = xlogy(n, lam_dt) - lam_dt - gammaln(n + 1.0)       # (n_max+1,)
    var = sigma * sigma * dt + n * delta * delta                  # (n_max+1,)
    mean = a * dt + n * m_j
    resid = x[:, None] - mean[None, :]                            # (n_obs, n_max+1)
    z2 = resid * resid / var[None, :]
    with np.errstate(divide="ignore"):
        comp = (log_pois - 0.5 * (_LOG_2PI + np.log(var)))[None, :] - 0.5 * z2
    peak = comp.max(axis=1)
    weights = np.exp(comp - peak[:, None])
    total = weights.sum(axis=1)
    ell = peak + np.log(total)                                    # (n_obs,)
    loglik = float(ell.sum())
    if not gradient:
        return loglik, None
    post = weights / total[:, None]                               # posterior of N
    r_over_v = resid / var[None, :]
    d_var = 0.5 * (r_over_v * r_over_v - 1.0 / var[None, :])      # d log phi / d var
    g_a = dt * float((post * r_over_v).sum())
    g_lnsigma = 2.0 * sigma * sigma * dt * float((post * d_var).sum())
    g_m = float((post * r_over_v * n[None, :]).sum())
    g_lndelta = 2.0 * delta * delta * float((post * d_var * n[None, :]).sum())
    # d/d lambda of log Pois = n/lambda - dt. sum_n post_n * n / lambda is
    # computed as exp(comp_n - ln lambda - ell) so that it stays finite at
    # lambda = 0 (only n = 1 contributes there).
    if n_max >= 1:
        n1 = n[1:]
        shifted = (xlogy(n1 - 1.0, lam_dt) + math.log(dt) - lam_dt
                   - gammaln(n1 + 1.0) - 0.5 * (_LOG_2PI + np.log(var[1:])))
        shifted = shifted[None, :] - 0.5 * z2[:, 1:] - ell[:, None]
        with np.errstate(over="ignore"):
            # May be +inf when sigma is tiny and lambda = 0 (true slope).
            g_lam = float((np.exp(shifted) * n1[None, :]).sum()) - dt * x.size
    else:
        g_lam = -dt * x.size
    return loglik, np.array([g_a, g_lnsigma, g_lam, g_m, g_lndelta])


def _natural_to_a(mu: float, sigma: float, lam: float, m_j: float,
                  delta: float) -> float:
    with np.errstate(over="ignore"):
        k = math.expm1(m_j + 0.5 * delta * delta)
    return mu - 0.5 * sigma * sigma - lam * k


def merton_daily_loglik(params: Sequence[float], log_returns: Any, *,
                        dt: float = 1.0 / TRADING_DAYS,
                        n_max: int | None = None) -> float:
    """Exact (up to truncation) Merton log-likelihood of log-returns.

    ``params = (mu, sigma, jump_intensity, jump_mean, jump_std)`` in the
    module's units; ``log_returns`` is a 1-D array of ``ln S_t - ln S_{t-1}``
    over steps of ``dt`` years. ``n_max`` defaults to
    ``truncation_n_max(dt)`` (neglected mass < 1e-14 for ``lambda <= 252``);
    for a larger ``lambda*dt`` pass a larger ``n_max``.
    """
    if not isinstance(params, Sequence) or isinstance(params, str) or len(params) != 5:
        raise ValueError("params must be (mu, sigma, jump_intensity, jump_mean, jump_std)")
    mu = _real("mu", params[0])
    sigma = _real("sigma", params[1], positive=True)
    lam = _real("jump_intensity", params[2], nonnegative=True)
    m_j = _real("jump_mean", params[3])
    delta = _real("jump_std", params[4], positive=True)
    step = _real("dt", dt, positive=True)
    x = _returns_array(log_returns)
    nm = truncation_n_max(step) if n_max is None else n_max
    if isinstance(nm, bool) or not isinstance(nm, Integral) or nm < 0:
        raise ValueError("n_max must be a nonnegative integer")
    nm = int(nm)
    _check_memory(x.size, nm)
    a = _natural_to_a(mu, sigma, lam, m_j, delta)
    if not math.isfinite(a):
        raise ValueError("drift a = mu - sigma^2/2 - lambda*k is not finite")
    value, _ = _loglik_terms(x, a, sigma, lam, m_j, delta, step, nm)
    if not math.isfinite(value):
        raise ValueError("log-likelihood is not finite for these parameters")
    return value


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class _Problem:
    """Scaled optimisation problem ``u -> nll``.

    ``u = (a*dt/s, ln sigma, lambda/25.2, m_J/s, ln delta)`` with ``s`` the
    daily sample standard deviation, so that every coordinate is O(1).
    """

    def __init__(self, x: FloatArray, dt: float) -> None:
        self.x = x
        self.dt = dt
        self.n_max = truncation_n_max(dt)
        _check_memory(x.size, self.n_max)
        self.s = float(np.std(x))
        self.mean = float(np.mean(x))
        x_abs = float(np.max(np.abs(x)))
        sd_ann = self.s / math.sqrt(dt)
        sigma_hi = max(10.0 * sd_ann, 2.0 * SIGMA_FLOOR)
        jump_hi = max(2.0 * x_abs, 10.0 * self.s)
        delta_hi = max(jump_hi, 2.0 * DELTA_FLOOR)
        # Natural-unit bounds, reported in the result.
        self.bounds_natural = {
            "a_dt": (-10.0 * self.s + self.mean, 10.0 * self.s + self.mean),
            "sigma": (SIGMA_FLOOR, sigma_hi),
            "jump_intensity": (0.0, LAMBDA_MAX),
            "jump_mean": (-jump_hi, jump_hi),
            "jump_std": (DELTA_FLOOR, delta_hi),
        }
        lo_a, hi_a = self.bounds_natural["a_dt"]
        self.bounds_u = [
            (lo_a / self.s, hi_a / self.s),
            (math.log(SIGMA_FLOOR), math.log(sigma_hi)),
            (0.0, LAMBDA_MAX / _LAMBDA_SCALE),
            (-jump_hi / self.s, jump_hi / self.s),
            (math.log(DELTA_FLOOR), math.log(delta_hi)),
        ]

    def decode(self, u: FloatArray) -> tuple[float, float, float, float, float]:
        return (float(u[0]) * self.s / self.dt, math.exp(u[1]),
                float(u[2]) * _LAMBDA_SCALE, float(u[3]) * self.s, math.exp(u[4]))

    def encode(self, a: float, sigma: float, lam: float, m_j: float,
               delta: float) -> FloatArray:
        u = np.array([a * self.dt / self.s, math.log(sigma), lam / _LAMBDA_SCALE,
                      m_j / self.s, math.log(delta)])
        lo = np.array([b[0] for b in self.bounds_u])
        hi = np.array([b[1] for b in self.bounds_u])
        return np.clip(u, lo, hi)

    def nll_and_grad(self, u: FloatArray) -> tuple[float, FloatArray]:
        a, sigma, lam, m_j, delta = self.decode(u)
        ll, g = _loglik_terms(self.x, a, sigma, lam, m_j, delta, self.dt,
                              self.n_max, gradient=True)
        assert g is not None
        scale = np.array([self.s / self.dt, 1.0, _LAMBDA_SCALE, self.s, 1.0])
        nll = -ll
        if not math.isfinite(nll):
            return _NLL_CAP, np.zeros(5)
        # An infinite slope (lambda = 0 with a tiny sigma: the jump component
        # explains an outlier infinitely better) is capped so that the line
        # search sees a consistent, finite descent direction and backtracks.
        grad = np.nan_to_num(-g * scale, nan=0.0, posinf=_GRAD_CAP, neginf=-_GRAD_CAP)
        return nll, np.clip(grad, -_GRAD_CAP, _GRAD_CAP)

    def nll_natural(self, theta: FloatArray) -> float:
        """nll at natural parameters ``(mu, sigma, lambda, m_J, delta)``."""
        mu, sigma, lam, m_j, delta = (float(v) for v in theta)
        a = _natural_to_a(mu, sigma, lam, m_j, delta)
        ll, _ = _loglik_terms(self.x, a, sigma, lam, m_j, delta, self.dt, self.n_max)
        return -ll


def _starting_points(prob: _Problem, n_starts: int,
                     rng: np.random.Generator) -> list[FloatArray]:
    dt, s = prob.dt, prob.s
    sd_ann = s / math.sqrt(dt)
    a0 = prob.mean / dt
    # Start 0: GBM-like (few, wide, centred jumps; most variance in sigma).
    starts = [prob.encode(a0, max(0.95 * sd_ann, SIGMA_FLOOR), 0.5, 0.0,
                          max(3.0 * s, DELTA_FLOOR))]
    for _ in range(n_starts - 1):
        lam = math.exp(rng.uniform(math.log(1.0), math.log(100.0)))
        delta = math.exp(rng.uniform(math.log(s), math.log(6.0 * s)))
        m_j = rng.uniform(-2.0 * s, 2.0 * s)
        sigma = max(sd_ann * rng.uniform(0.4, 1.0), SIGMA_FLOOR)
        a = a0 - lam * m_j
        starts.append(prob.encode(a, sigma, lam, m_j, max(delta, DELTA_FLOOR)))
    return starts


def _hessian(f: Any, theta: FloatArray, steps: FloatArray) -> FloatArray:
    """Central finite-difference Hessian of a scalar function."""
    k = theta.size
    f0 = f(theta)
    h = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k); ei[i] = steps[i]
        h[i, i] = (f(theta + ei) - 2.0 * f0 + f(theta - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros(k); ej[j] = steps[j]
            h[i, j] = h[j, i] = (f(theta + ei + ej) - f(theta + ei - ej)
                                 - f(theta - ei + ej) + f(theta - ei - ej)) / (
                4.0 * steps[i] * steps[j])
    return h


_HESS_FLOORS = np.array([0.1, 0.01, 1.0, 0.01, 0.01])  # typical scale per parameter
_HESS_REL_STEP = 1e-4                                  # ~ eps**(1/4)


def fit_merton_jumps_returns(returns: Any, *, n_starts: int = 8, seed: int | None = 0,
                             dt: float = 1.0 / TRADING_DAYS,
                             maxiter: int = 1000) -> dict[str, Any]:
    """Merton jump-diffusion MLE from daily log-returns (see module docstring).

    Parameters
    ----------
    returns : 1-D real array (or Series) of log-returns ``ln S_t - ln S_{t-1}``.
    n_starts : number of L-BFGS-B starts (>= 1). Start 0 is GBM-like
        (``lambda = 0.5``); the others are drawn from
        ``default_rng(SeedSequence(seed))``.
    seed : nonnegative integer or ``None`` (``None`` is not reproducible).
    dt : step in years, default ``1/252``.
    maxiter : L-BFGS-B iteration limit per start.

    Returns a JSON-safe dict:

    * ``mu, sigma, jump_intensity, jump_mean, jump_std`` (annual units;
      ``mu = a + sigma^2/2 + lambda*k`` is the PRICE drift);
    * ``converged`` (optimizer success of the best start), ``nll``,
      ``n_starts``, ``n_converged``, ``n_obs``, ``n_params`` (5), ``aic``,
      ``bic`` (``2k + 2 nll``, ``k ln n + 2 nll``);
    * ``gbm_nll``, ``gbm_aic``, ``gbm_bic`` for the normal MLE of the same
      returns with ``ddof = 0`` (2 parameters). This differs from
      :func:`calibration.estimation.estimate_gbm_params`, which uses
      ``ddof = 1``; ddof = 0 is the maximiser of the Gaussian likelihood;
    * ``std_errors``: dict of the five parameters' asymptotic standard errors
      from the inverse of the central finite-difference Hessian of the nll in
      ``(mu, sigma, lambda, m_J, delta)``, or ``None`` when an estimate is at a
      bound or the Hessian is not positive definite;
    * ``warnings``: identifiability and convergence warnings (strings);
    * ``dt``, ``n_max``, ``seed``, ``bounds``, ``optimizer``, ``rng``,
      ``model_comparison_note``.
    """
    x = _returns_array(returns)
    if isinstance(n_starts, bool) or not isinstance(n_starts, Integral) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)
                             or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    step = _real("dt", dt, positive=True)
    n_obs = int(x.size)
    if not float(np.std(x)) > 0.0:
        raise ValueError("log returns have zero variance; the likelihood is degenerate")

    prob = _Problem(x, step)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    best: Any = None
    n_converged = 0
    for u0 in _starting_points(prob, int(n_starts), rng):
        res = minimize(prob.nll_and_grad, u0, jac=True, method="L-BFGS-B",
                       bounds=prob.bounds_u,
                       options={"maxiter": int(maxiter), "ftol": 1e-13, "gtol": 1e-8})
        if res.success:
            n_converged += 1
        if math.isfinite(res.fun) and res.fun < _NLL_CAP and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("no start produced a finite likelihood")

    a, sigma, lam, m_j, delta = prob.decode(best.x)
    k = math.expm1(m_j + 0.5 * delta * delta)
    mu = a + 0.5 * sigma * sigma + lam * k
    nll = float(prob.nll_natural(np.array([mu, sigma, lam, m_j, delta])))

    # GBM (normal) MLE with ddof = 0.
    var0 = float(np.var(x))
    gbm_nll = 0.5 * n_obs * (math.log(2.0 * math.pi * var0) + 1.0)
    log_n = math.log(n_obs)

    warnings: list[str] = []
    if not best.success:
        warnings.append(f"best start did not report convergence: {best.message}")
    if n_obs < 250:
        warnings.append(f"only {n_obs} returns: jump parameters are weakly identified")
    tol_b = 1e-6
    at_bound = []
    if lam <= tol_b * LAMBDA_MAX:
        at_bound.append("jump_intensity at lower bound 0")
    if lam >= LAMBDA_MAX * (1.0 - tol_b):
        at_bound.append(f"jump_intensity at upper bound {LAMBDA_MAX:g}")
    if sigma <= SIGMA_FLOOR * (1.0 + tol_b):
        at_bound.append(
            f"sigma at floor {SIGMA_FLOOR:g}: the likelihood is unbounded as sigma -> 0, "
            "the estimate is an artefact of the floor")
    if delta <= DELTA_FLOOR * (1.0 + tol_b):
        at_bound.append(f"jump_std at floor {DELTA_FLOOR:g}")
    for name, (lo, hi), val in (
            ("a*dt", prob.bounds_u[0], best.x[0]), ("jump_mean", prob.bounds_u[3], best.x[3]),
            ("sigma", (-math.inf, prob.bounds_u[1][1]), best.x[1]),
            ("jump_std", (-math.inf, prob.bounds_u[4][1]), best.x[4])):
        if val <= lo + tol_b or val >= hi - tol_b:
            at_bound.append(f"{name} at a data-dependent optimizer bound")
    warnings.extend(at_bound)
    expected_jumps = lam * step * n_obs
    if expected_jumps < 1.0:
        warnings.append(
            f"lambda*T = {expected_jumps:.3g} expected jumps in the sample: "
            "jump_mean and jump_std are not identified (lambda ~ 0)")
    jump_size = math.sqrt(m_j * m_j + delta * delta)
    diff_sd = sigma * math.sqrt(step)
    if lam * step >= 0.5 and delta <= 2.0 * diff_sd:
        warnings.append(
            f"lambda*dt = {lam * step:.3g} with small jumps (delta = {delta:.3g} vs daily "
            f"diffusion sd {diff_sd:.3g}): jumps resemble diffusion, essentially only "
            "sigma^2 + lambda*(m_J^2 + delta^2) is identified")
    elif expected_jumps >= 1.0 and jump_size <= diff_sd:
        warnings.append(
            f"jump size sqrt(m_J^2 + delta^2) = {jump_size:.3g} is below the daily "
            f"diffusion sd {diff_sd:.3g}: jumps and diffusion are weakly separated")

    std_errors: dict[str, float | None] | None = None
    theta = np.array([mu, sigma, lam, m_j, delta])
    if at_bound:
        warnings.append("estimate at a bound: Fisher standard errors are invalid and omitted")
    else:
        steps = _HESS_REL_STEP * np.maximum(np.abs(theta), _HESS_FLOORS)
        if lam - steps[2] < 0.0:
            steps[2] = 0.5 * lam
        hess = _hessian(prob.nll_natural, theta, steps)
        hess = 0.5 * (hess + hess.T)
        eig_ok = bool(np.all(np.isfinite(hess)))
        if eig_ok:
            eig_ok = bool(np.min(np.linalg.eigvalsh(hess)) > 0.0)
        if not eig_ok:
            warnings.append("numerical Hessian is not positive definite: "
                            "standard errors omitted (flat or non-identified direction)")
        else:
            cov = np.linalg.inv(hess)
            diag = np.diag(cov)
            std_errors = {
                name: (float(math.sqrt(v)) if math.isfinite(v) and v > 0.0 else None)
                for name, v in zip(PARAM_NAMES, diag)}

    return {
        "model": "merton_jump_diffusion",
        "measure": "physical",
        "mu": float(mu),
        "sigma": float(sigma),
        "jump_intensity": float(lam),
        "jump_mean": float(m_j),
        "jump_std": float(delta),
        "jump_compensator_k": float(k),
        "converged": bool(best.success),
        "nll": nll,
        "n_starts": int(n_starts),
        "n_converged": int(n_converged),
        "n_obs": n_obs,
        "n_params": 5,
        "aic": 2.0 * 5 + 2.0 * nll,
        "bic": 5 * log_n + 2.0 * nll,
        "gbm_nll": float(gbm_nll),
        "gbm_aic": 2.0 * 2 + 2.0 * gbm_nll,
        "gbm_bic": 2 * log_n + 2.0 * gbm_nll,
        "std_errors": std_errors,
        "warnings": warnings,
        "dt": step,
        "n_max": int(prob.n_max),
        "seed": seed,
        "rng": "NumPy default_rng(SeedSequence(seed))/PCG64 (random starts only)",
        "optimizer": "scipy.optimize.minimize L-BFGS-B, analytic gradient, multi-start",
        "bounds": {key: [float(lo), float(hi)] for key, (lo, hi)
                   in prob.bounds_natural.items()},
        "model_comparison_note": (
            "GBM vs Merton LRT is not chi-square (lambda = 0 on the boundary, "
            "jump_mean/jump_std absent under the null: Davies' problem); "
            "AIC/BIC are heuristics"),
    }


def fit_merton_jumps(prices: pd.Series, *, n_starts: int = 8, seed: int | None = 0,
                     maxiter: int = 1000) -> dict[str, Any]:
    """Merton MLE from daily closing prices (at least three).

    Prices are validated with :func:`calibration.prices.validate_prices` and
    converted with :func:`calibration.estimation.log_returns`; the step is
    ``1/TRADING_DAYS``. See :func:`fit_merton_jumps_returns` for the output.
    """
    validated = validate_prices(prices, min_observations=3)
    return fit_merton_jumps_returns(log_returns(validated).to_numpy(),
                                    n_starts=n_starts, seed=seed, maxiter=maxiter)
