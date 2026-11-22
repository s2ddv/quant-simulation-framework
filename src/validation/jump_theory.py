"""Analytical terminal benchmarks for Merton and Kou jump-diffusions.

Model and units as in ``simulation.jump_diffusion``: price in asset currency,
horizon ``T`` in years, ``mu`` (price drift) and ``jump_intensity`` (lambda) in
year^-1, ``sigma`` in year^-1/2, log-jump parameters dimensionless. With
``k = E[J - 1]`` and the compensated drift, ``ln S_T`` equals
``ln S0 + (mu - sigma^2/2 - lambda*k)T + sigma*W_T + sum_{i<=N_T} ln J_i``.

Moments (independence of W, N and the J_i; Poisson moment generating function
``E[x^N] = exp(lambda*T*(x - 1))``)::

    E[S_T]   = S0 * exp(mu*T)
    E[S_T^2] = S0^2 * exp((2*mu + sigma^2)*T + lambda*T*(E[J^2] - 1 - 2k))
    Var[S_T] = E[S_T]^2 * expm1(sigma^2*T + lambda*T*E[(J - 1)^2])

since ``E[J^2] - 1 - 2k = E[(J - 1)^2]``. Merton:
``E[(J-1)^2] = exp(2m+delta^2)*expm1(delta^2) + expm1(m+delta^2/2)^2``.
Kou: ``E[(J-1)^2] = 2p/((eta1-1)(eta1-2)) + 2(1-p)/((eta2+1)(eta2+2))``,
finite only for ``eta1 > 2`` when ``p > 0``.

References: R. C. Merton (1976), J. Financial Economics 3:125-144;
S. G. Kou (2002), Management Science 48(8):1086-1101. Formulas derived and
verified numerically; equations/pages not checked against the originals
(source inaccessible in this session).
"""

import math
from numbers import Real
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import poisson

#: Poisson right-tail mass left out of the mixture, ``P(N_T > n_max)``.
TRUNCATION_TAIL = 1e-14
#: Largest number of mixture components accepted (memory/time guard).
MAX_COMPONENTS = 1_000_000
_OVERFLOW = "jump-diffusion terminal statistics exceed finite float64 precision"


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


def _common(s0: object, mu: object, sigma: object, horizon: object,
            jump_intensity: object) -> tuple[float, float, float, float, float]:
    values = (_finite_real("s0", s0), _finite_real("mu", mu),
              _finite_real("sigma", sigma), _finite_real("horizon", horizon),
              _finite_real("jump_intensity", jump_intensity))
    if values[0] <= 0:
        raise ValueError("s0 must be positive")
    if values[2] < 0 or values[3] < 0 or values[4] < 0:
        raise ValueError("sigma, horizon and jump_intensity must be nonnegative")
    return values


def _merton_jump(jump_mean: object, jump_std: object) -> tuple[float, float]:
    m = _finite_real("jump_mean", jump_mean)
    d = _finite_real("jump_std", jump_std)
    if d < 0:
        raise ValueError("jump_std must be nonnegative")
    return m, d


def _kou_jump(p_up: object, eta_up: object, eta_down: object) -> tuple[float, float, float]:
    p = _finite_real("p_up", p_up)
    e1 = _finite_real("eta_up", eta_up)
    e2 = _finite_real("eta_down", eta_down)
    if not 0 <= p <= 1:
        raise ValueError("p_up must lie in [0, 1]")
    if e1 <= 1:
        raise ValueError("eta_up must exceed 1 (E[J] is infinite otherwise)")
    if e2 <= 0:
        raise ValueError("eta_down must be positive")
    return p, e1, e2


def _exp(log_value: float) -> float:
    try:
        value = math.exp(log_value)
    except OverflowError:
        raise ValueError(_OVERFLOW) from None
    if not math.isfinite(value) or value == 0:
        raise ValueError(_OVERFLOW)
    return value


def merton_jump_compensator(jump_mean: float, jump_std: float) -> float:
    """``k = E[J - 1] = expm1(m_J + delta^2/2)`` for ``ln J ~ N(m_J, delta^2)``."""
    m, d = _merton_jump(jump_mean, jump_std)
    try:
        k = math.expm1(m + 0.5 * d * d)
    except OverflowError:
        raise ValueError(_OVERFLOW) from None
    if not math.isfinite(k):
        raise ValueError(_OVERFLOW)
    return k


def kou_jump_compensator(p_up: float, eta_up: float, eta_down: float) -> float:
    """``k = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1`` (requires ``eta1 > 1``).

    Evaluated as ``p/(eta1-1) - (1-p)/(eta2+1)``, the same value without
    cancellation.
    """
    p, e1, e2 = _kou_jump(p_up, eta_up, eta_down)
    k = p / (e1 - 1) - (1 - p) / (e2 + 1)
    if not math.isfinite(k):
        raise ValueError(_OVERFLOW)
    return k


def _moments(s0: float, mu: float, sigma: float, horizon: float,
             jump_intensity: float, jump_second: float | None) -> dict[str, float | None]:
    """Moments from ``E[(J-1)^2]`` (``None``: infinite second moment)."""
    log_mean = math.log(s0) + mu * horizon
    mean = s0 if mu == 0 or horizon == 0 else _exp(log_mean)
    if jump_second is None:
        return {"mean": mean, "second_moment": None, "variance": None}
    with np.errstate(over="ignore", under="ignore"):
        scale = sigma * math.sqrt(horizon)
        a = float(scale * scale + (jump_intensity * horizon) * jump_second)
    if not math.isfinite(a):
        raise ValueError(_OVERFLOW)
    second = _exp(2 * log_mean + a)
    if a == 0:
        variance = 0.0
    else:
        # log(expm1(a)) without overflow for large a, accurate for small a.
        log_expm1 = math.log(math.expm1(a)) if a <= 1 else a + math.log1p(-math.exp(-a))
        variance = _exp(2 * log_mean + log_expm1)
    return {"mean": mean, "second_moment": second, "variance": variance}


def merton_terminal_moments(s0: float, mu: float, sigma: float, horizon: float,
                            jump_intensity: float, jump_mean: float,
                            jump_std: float) -> dict[str, float | None]:
    """Return ``{"mean", "second_moment", "variance"}`` of the Merton ``S_T``.

    ``E[S_T^2] = S0^2 exp((2mu + sigma^2)T + lambda*T*(e^{2m+2delta^2} - 1 - 2k))``,
    computed in log space. ``ValueError`` on invalid input or if a value is
    not representable as a positive finite float64 (a zero variance is
    returned only when sigma*sqrt(T) and lambda*T*E[(J-1)^2] are exactly 0).
    """
    s0, mu, sigma, horizon, lam = _common(s0, mu, sigma, horizon, jump_intensity)
    m, d = _merton_jump(jump_mean, jump_std)
    k = merton_jump_compensator(m, d)
    try:
        jump_second = math.exp(2 * m + d * d) * math.expm1(d * d) + k * k
    except OverflowError:
        raise ValueError(_OVERFLOW) from None
    if not math.isfinite(jump_second):
        raise ValueError(_OVERFLOW)
    return _moments(s0, mu, sigma, horizon, lam, jump_second)


def kou_terminal_moments(s0: float, mu: float, sigma: float, horizon: float,
                         jump_intensity: float, p_up: float, eta_up: float,
                         eta_down: float) -> dict[str, float | None]:
    """Return ``{"mean", "second_moment", "variance"}`` of the Kou ``S_T``.

    ``E[J^2] = p*eta1/(eta1-2) + (1-p)*eta2/(eta2+2)``. When ``p_up > 0``
    and ``eta_up <= 2`` the second moment of ``J`` (hence of ``S_T`` when
    ``lambda*T > 0``) is infinite: ``"second_moment"`` and ``"variance"``
    are then ``None`` while ``"mean"`` is still returned. Note: the sample
    variance has finite variance only for ``eta_up > 4`` (fourth moment),
    which tests comparing variances should respect.
    """
    s0, mu, sigma, horizon, lam = _common(s0, mu, sigma, horizon, jump_intensity)
    p, e1, e2 = _kou_jump(p_up, eta_up, eta_down)
    down = 2 * (1 - p) / ((e2 + 1) * (e2 + 2))
    if p > 0 and e1 <= 2:
        if lam * horizon > 0:
            return _moments(s0, mu, sigma, horizon, lam, None)
        jump_second = 0.0  # No jumps occur; the jump law is irrelevant.
    else:
        jump_second = (2 * p / ((e1 - 1) * (e1 - 2)) if p > 0 else 0.0) + down
    if not math.isfinite(jump_second):
        raise ValueError(_OVERFLOW)
    return _moments(s0, mu, sigma, horizon, lam, jump_second)


def merton_log_mixture(s0: float, mu: float, sigma: float, horizon: float,
                       jump_intensity: float, jump_mean: float,
                       jump_std: float) -> dict[str, Any]:
    """Poisson mixture of normals for ``ln S_T`` (Merton), truncated.

    Component ``n = 0..n_max`` has weight ``P(N_T = n)``, mean
    ``ln S0 + (mu - sigma^2/2 - lambda*k)T + n*m_J`` and variance
    ``sigma^2*T + n*delta^2``; a zero variance is an atom. ``n_max =
    poisson.isf(TRUNCATION_TAIL, lambda*T)`` (0 when ``lambda*T = 0``);
    weights are not renormalized and ``truncated_mass = P(N_T > n_max)``
    bounds the absolute CDF error. Returns a dict with ``weights``,
    ``means``, ``variances`` (float64 arrays), ``n_max`` and
    ``truncated_mass``.
    """
    s0, mu, sigma, horizon, lam = _common(s0, mu, sigma, horizon, jump_intensity)
    m, d = _merton_jump(jump_mean, jump_std)
    k = merton_jump_compensator(m, d)
    lam_t = lam * horizon
    if not math.isfinite(lam_t):
        raise ValueError(_OVERFLOW)
    if lam_t > 0:
        n_max = int(poisson.isf(TRUNCATION_TAIL, lam_t))
        if n_max + 1 > MAX_COMPONENTS:
            raise ValueError(
                f"lambda*T = {lam_t} needs {n_max + 1} mixture components; "
                f"limit is {MAX_COMPONENTS}"
            )
        n = np.arange(n_max + 1, dtype=np.float64)
        weights = poisson.pmf(n, lam_t).astype(np.float64)
        truncated = float(poisson.sf(n_max, lam_t))
    else:
        n_max = 0
        n = np.zeros(1)
        weights = np.ones(1)
        truncated = 0.0
    scale = sigma * math.sqrt(horizon)
    base = math.log(s0) + mu * horizon - (0.5 * scale) * scale - (lam_t * k if lam_t else 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        means = base + n * m
        variances = scale * scale + n * (d * d)
    if not (math.isfinite(base) and np.isfinite(means).all()
            and np.isfinite(variances).all()):
        raise ValueError(_OVERFLOW)
    return {"weights": weights, "means": means, "variances": variances,
            "n_max": n_max, "truncated_mass": truncated}


def _log_cdf(y: np.ndarray, mixture: dict[str, Any]) -> np.ndarray:
    """Mixture CDF of ``ln S_T`` at ``y`` (1-D), chunked over ``y``."""
    keep = mixture["weights"] > 0
    w = mixture["weights"][keep]
    mean = mixture["means"][keep]
    sd = np.sqrt(mixture["variances"][keep])
    atom = sd == 0
    safe_sd = np.where(atom, 1.0, sd)
    result = np.empty(y.size, dtype=np.float64)
    chunk = max(1, (1 << 20) // max(1, w.size))
    for start in range(0, y.size, chunk):
        block = y[start:start + chunk, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = ndtr((block - mean) / safe_sd)
        values = np.where(atom, (block >= mean).astype(np.float64), values)
        result[start:start + chunk] = values @ w
    return result


def merton_terminal_cdf(x: Any, s0: float, mu: float, sigma: float, horizon: float,
                        jump_intensity: float, jump_mean: float,
                        jump_std: float) -> Any:
    """``P(S_T <= x)`` for the Merton model, vectorized over ``x``.

    ``x`` must be finite and positive. Absolute error is at most
    ``truncated_mass`` (< 1e-14) plus floating-point rounding. Returns a
    float for scalar ``x`` and an array of ``x``'s shape otherwise.
    """
    mixture = merton_log_mixture(s0, mu, sigma, horizon, jump_intensity,
                                 jump_mean, jump_std)
    values = np.asarray(x)
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number) \
            or np.iscomplexobj(values):
        raise ValueError("x must contain real numbers (bool rejected)")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("x must be finite and positive")
    out = _log_cdf(np.log(values).reshape(-1), mixture).reshape(values.shape)
    return float(out) if out.ndim == 0 else out


def merton_terminal_quantile(p: float, s0: float, mu: float, sigma: float,
                             horizon: float, jump_intensity: float,
                             jump_mean: float, jump_std: float) -> float:
    """``p``-quantile of the Merton ``S_T`` via ``brentq`` on the log CDF.

    Solves ``F_{ln S_T}(y) = p`` on a bracket spanning each positive-weight
    component's mean +/- 40 standard deviations (atoms widened by 1), with
    ``xtol = 1e-13`` in log price, then returns ``exp(y)``. For a mixture
    with atoms, the root is the location of the step (``inf{y: F(y) >= p}``
    up to ``xtol``). With ``jump_intensity = 0`` this matches
    ``validation.statistics.terminal_quantile``. ``p`` must lie in
    ``(0, 1 - truncated_mass)``.
    """
    prob = _finite_real("p", p)
    if not 0 < prob < 1:
        raise ValueError("p must lie strictly inside (0, 1)")
    mixture = merton_log_mixture(s0, mu, sigma, horizon, jump_intensity,
                                 jump_mean, jump_std)
    keep = mixture["weights"] > 0
    if prob >= float(mixture["weights"][keep].sum()):
        raise ValueError("p exceeds the probability mass kept by the truncation")
    mean = mixture["means"][keep]
    sd = np.sqrt(mixture["variances"][keep])
    width = np.where(sd == 0, 1.0, 40 * sd)
    low, high = float(np.min(mean - width)), float(np.max(mean + width))
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(_OVERFLOW)

    def objective(y: float) -> float:
        return float(_log_cdf(np.array([y]), mixture)[0]) - prob

    root = brentq(objective, low, high, xtol=1e-13, rtol=4 * np.finfo(float).eps,
                  maxiter=500)
    return _exp(float(root))
