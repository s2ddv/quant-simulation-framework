"""GARCH(1,1) with standardized Student-t innovations, fitted by scipy MLE.

Daily log returns r_t (in percent internally for conditioning):
    r_t = mu + e_t,  e_t = sqrt(h_t) * z_t,  z_t ~ t_nu scaled to unit variance
    h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
Stationarity: alpha + beta < 1. nu > 2 so the variance exists.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

NU_MIN, NU_MAX = 2.1, 200.0


def _variance_path(e2: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """h_t for t=0..n (h[n] is the one-step-ahead variance after the sample)."""
    drive = np.empty(e2.size + 1)
    drive[0] = h0
    drive[1:] = omega + alpha * e2
    # h_{k} = beta*h_{k-1} + drive_k, with h_0 = drive[0]
    return lfilter([1.0], [1.0, -beta], drive)


def _nll(theta, r, h0):
    mu, omega, alpha, beta, nu = theta
    e = r - mu
    h = _variance_path(e * e, omega, alpha, beta, h0)[:-1]
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e12
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(h) - 0.5 * (nu + 1) * np.log1p(e * e / (h * (nu - 2))))
    return float(-np.sum(ll))


def fit_garch_t(returns: np.ndarray, start: np.ndarray | None = None) -> dict:
    """Fit by constrained MLE. ``returns`` are daily log returns (decimals).

    Returns a dict with parameters in decimal units (mu, omega), the flag
    ``converged`` and optimizer message; failures are reported, never hidden.
    """
    r = np.asarray(returns, dtype=float) * 100.0
    if r.size < 250 or not np.isfinite(r).all():
        raise ValueError("need at least 250 finite returns")
    var = float(np.var(r))
    h0 = var
    if start is None:
        start = np.array([float(np.mean(r)), var * 0.05, 0.08, 0.90, 8.0])
    bounds = [(-1.0, 1.0), (1e-8, 10 * var), (0.0, 0.5), (0.0, 0.9999), (NU_MIN, NU_MAX)]
    cons = [{"type": "ineq", "fun": lambda t: 0.9999 - t[2] - t[3]}]
    best = None
    for x0 in (start, np.array([float(np.mean(r)), var * 0.05, 0.08, 0.90, 8.0])):
        res = minimize(_nll, x0, args=(r, h0), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
        if res.success:
            break
    # The loop body runs at least once, so ``best`` is always set here.
    assert best is not None
    mu, omega, alpha, beta, nu = best.x
    e = r - mu
    h = _variance_path(e * e, omega, alpha, beta, h0)
    return {"mu": mu / 100.0, "omega": omega / 1e4, "alpha": alpha, "beta": beta, "nu": nu,
            "converged": bool(best.success), "message": str(best.message),
            "nll": float(best.fun), "n_obs": int(r.size),
            "h_next": float(h[-1]) / 1e4,        # variance of day t+1 given data to t
            "theta_pct": best.x.copy()}


def _nll_normal(theta, r, h0):
    mu, omega, alpha, beta = theta
    e = r - mu
    h = _variance_path(e * e, omega, alpha, beta, h0)[:-1]
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e12
    return float(0.5 * np.sum(np.log(2 * np.pi * h) + e * e / h))


def fit_garch_normal(returns: np.ndarray, start: np.ndarray | None = None) -> dict:
    """GARCH(1,1) with Gaussian innovations (ablation: same recursion, no fat-tailed shocks).

    Same conventions, bounds and failure reporting as ``fit_garch_t``; ``nu`` is None.
    """
    r = np.asarray(returns, dtype=float) * 100.0
    if r.size < 250 or not np.isfinite(r).all():
        raise ValueError("need at least 250 finite returns")
    var = float(np.var(r)); h0 = var
    default = np.array([float(np.mean(r)), var * 0.05, 0.08, 0.90])
    bounds = [(-1.0, 1.0), (1e-8, 10 * var), (0.0, 0.5), (0.0, 0.9999)]
    cons = [{"type": "ineq", "fun": lambda t: 0.9999 - t[2] - t[3]}]
    best = None
    for x0 in (default if start is None else start, default):
        res = minimize(_nll_normal, x0, args=(r, h0), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
        if res.success:
            break
    mu, omega, alpha, beta = best.x
    h = _variance_path((r - mu) ** 2, omega, alpha, beta, h0)
    return {"mu": mu / 100.0, "omega": omega / 1e4, "alpha": alpha, "beta": beta, "nu": None,
            "converged": bool(best.success), "message": str(best.message), "nll": float(best.fun),
            "n_obs": int(r.size), "h_next": float(h[-1]) / 1e4, "theta_pct": best.x.copy()}


def next_variance(returns: np.ndarray, params: dict) -> float:
    """One-step-ahead variance using frozen parameters and returns up to t."""
    r = np.asarray(returns, dtype=float) * 100.0
    e = r - params["mu"] * 100.0
    h0 = float(np.var(r))
    h = _variance_path(e * e, params["omega"] * 1e4, params["alpha"], params["beta"], h0)
    return float(h[-1]) / 1e4


def simulate_cumulative(params: dict, h_next: float, horizon: int, n: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` cumulative log returns over ``horizon`` days from the filtered state."""
    mu, omega, a, b = (params[k] for k in ("mu", "omega", "alpha", "beta"))
    nu = params.get("nu")                      # None -> Gaussian innovations
    sd = None if nu is None else np.sqrt((nu - 2) / nu)
    h = np.full(n, h_next)
    total = np.zeros(n)
    for _ in range(horizon):
        z = rng.standard_normal(n) if nu is None else rng.standard_t(nu, size=n) * sd
        e = np.sqrt(h) * z
        total += mu + e
        h = omega + a * e * e + b * h
    return total
