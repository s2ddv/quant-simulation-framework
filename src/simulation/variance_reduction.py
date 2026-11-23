"""Variance reduction for Monte Carlo expectations under the normal GBM.

Antithetic variates (E2.1)
--------------------------
For ``n_sims = 2m`` the sampler draws ``Z`` with shape ``(steps, m)`` from the
instance's own ``np.random.Generator`` and uses ``[Z, -Z]``. Columns
``0..m-1`` hold the originals and columns ``m..2m-1`` their mirrors, so
column ``i`` and column ``i + m`` form pair ``i``. Terminal vectors use the
same order ``[originals..., mirrors...]``. An odd ``n_sims`` is rejected:
a lone unpaired path would carry a different weight in the pair estimator.
Antithetic sampling is defined only for the normal (exact GBM) mode; the
discrete Student t scenario is outside this scope and is rejected.

Each sample is still a draw from the exact GBM law, because ``-Z`` has the
same law as ``Z``. The two members of a pair are, however, *dependent*.
For an integrand ``f`` the estimator and its standard error use the pair as
the independent unit::

    Y_i = (f(S+_i) + f(S-_i)) / 2,        i = 1..m
    estimate = mean(Y),   SE = std(Y, ddof=1) / sqrt(m)

For the terminal price (``v = sigma**2 * T``, ``E S_T = S0*exp(mu*T)``)::

    Var(S_T)       = S0**2 * exp(2*mu*T) * expm1(v)
    Cov(S+, S-)    = S0**2 * exp(2*(mu - sigma**2/2)*T) - (E S_T)**2
                   = S0**2 * exp(2*mu*T) * expm1(-v)      (negative)
    Var(Y)         = (Var + Cov) / 2
    VRF            = Var / (2*Var(Y)) = 1 / (-expm1(-v))

The VRF compares, at equal cost ``2m`` exponentials, the plain mean of
``2m`` independent draws with the mean of ``m`` antithetic pairs. Examples
with ``T = 1``: sigma 0.2 -> 25.50, 0.25 -> 16.51, 0.5 -> 4.521, 1 -> 1.582.

When it does not help: the gain needs ``f`` to be monotone in ``Z``
(negative pair covariance). For even or non-monotone payoffs the pair can
be positively correlated and the VRF falls below 1; e.g. a straddle
``|S_T - K|`` with ``K`` at the median and sigma 0.2 gives VRF about 0.54
(limit 0.5 as the payoff becomes even in ``Z``).

Interval caveat: the i.i.d. intervals of ``risk.intervals`` (Wilson for the
loss probability, order-statistic VaR interval, bootstrap CVaR) assume
independent columns and are **not** valid on antithetic samples as they
stand. Use the pair as the unit (``antithetic_mean``) for expectations; the
antithetic sample remains a valid sample of the law for point estimates.

``"control"`` (E2.2) is an *estimator* technique applied after sampling,
not a sampling mode, so it is not a value of ``VarianceReduction``.

Control variate (E2.2)
----------------------
For a target ``Y = g(S)`` and a control ``X`` with known mean ``E X`` (for
the GBM, ``X = S_T`` and ``E X = S0*exp(mu*T)`` under the physical measure P,
see ``gbm_terminal_control_mean``), ``control_variate_mean`` returns::

    theta = mean(Y) - beta * (mean(X) - E X)
    e_i   = Y_i - beta * (X_i - E X)              (residuals)
    beta_hat = Cov(X, Y) / Var(X)                 (sample moments)

with three sources for ``beta``:

* ``"same_sample"`` (default): ``beta_hat`` from the same ``N`` draws.
  ``theta`` is biased by ``O(1/N)``; ``SE = std(e, ddof=2)/sqrt(N)`` (residual
  of a regression with intercept), ``N >= 3``.
* ``"pilot"`` (``pilot_fraction = f`` in ``(0, 1)``): the first
  ``n_pilot = floor(f*N)`` observations, in the given order, estimate
  ``beta`` and are then discarded; ``theta`` and ``SE = std(e, ddof=1)/
  sqrt(N - n_pilot)`` use only the remaining ones. The partition is a
  deterministic prefix with no RNG: for independent, identically
  distributed draws in sampling order (plain ``"none"`` GBM columns) the
  prefix is independent of the rest, so ``theta`` is unbiased conditional
  on ``beta``. Do not pass sorted or otherwise reordered samples.
* ``"fixed"`` (``beta`` given): no estimation, unbiased, ``SE`` with
  ``ddof=1`` (useful in E3 with a known optimal or reused coefficient).

``rho2`` is the squared sample correlation of ``(X, Y)`` on the
observations that form ``theta`` (equal to ``beta_hat**2*Var(X)/Var(Y)``)
and ``vrf = Var(Y)/Var(e)`` on the same observations (``ddof=1`` for both)
is the achieved variance-reduction factor estimate. In ``"same_sample"``
mode it equals ``1/(1 - rho2)``; with a pilot or fixed ``beta`` it is at
most that value and can fall below 1 when ``beta`` is poor. Both are
estimates, not population values.

Example: ``Y = (S_T - 100)+`` under P with ``S0 = 100``, ``mu = 0.05``,
``sigma = 0.2``, ``T = 1`` and ``X = S_T`` has a theoretical VRF near 6.9.

When it does not help: the gain is ``1/(1 - rho**2)``, so it vanishes when
``Y`` is weakly correlated with ``S_T``. With the parameters above: a
far out-of-the-money call ``K = 160`` gives VRF about 1.11, a put ``K = 60``
about 1.01 and a digital ``1{S_T < 80}`` about 1.36. The control only
estimates *expectations*: it does not apply to quantiles, VaR or CVaR,
whose estimators and intervals (``risk``) are unchanged.

Antithetic samples are **not** supported by ``control_variate_mean``: the
residual SE and the pilot split assume independent observations, while an
antithetic sample has dependent pairs. Combining the two would require the
pair as the unit (averaging ``X`` and ``Y`` over each pair first), which is
outside E2.2.

Degenerate cases never produce ``nan`` or ``inf``; undefined quantities are
``None`` so the result is JSON-safe with ``allow_nan=False``. A constant
control (``Var(X) = 0``, e.g. ``sigma = 0``) gives ``beta = 0`` in the
estimated modes, a ``warning`` and ``rho2 = vrf = None``. A constant target
gives ``rho2 = None``. Perfect correlation (``Y = X``, or ``Y`` an exact
affine function of ``X``) gives ``rho2 = 1`` and a finite ``SE`` at or near
0. ``vrf`` is ``None`` when the residual variance is exactly zero (as for
``Y = X``, thanks to power-of-two rescaling); for other affine relations
rounding can leave a residual variance near machine precision and ``vrf``
is then a very large finite number with no statistical meaning.

Randomized quasi-Monte Carlo (E2.3)
-----------------------------------
``variance_reduction="qmc"`` replaces pseudo-random normals by
``Z = ndtri(u)``, with ``u`` the points of ``R`` independently scrambled
Sobol sequences (``scipy.stats.qmc.Sobol(d, scramble=True, bits=30,
rng=Generator)``; SciPy scrambles with a linear matrix scramble plus a
digital random shift, see ``QMC_RNG_CONVENTION``). ``n_sims = R*N`` with
``N = 2**k`` points per scramble (``random_base2(k)`` keeps the balance
properties of the net); scramble ``r`` occupies columns (or terminal
entries) ``[r*N, (r+1)*N)``. ``u`` is clipped to ``[2**-53, 1 - 2**-53]``
before ``ndtri`` because a 30-bit scrambled point can be exactly 0.

* Terminal prices: ``d = 1`` and ``S_T = S0*exp((mu - sigma**2/2)*T +
  sigma*sqrt(T)*Z)``, exact.
* Paths: ``d = steps`` and ``W`` on the grid ``t_k = k*T/steps`` is built by
  a Brownian bridge (``brownian_bridge``): dimension 0 sets
  ``W_T = sqrt(T)*Z_0`` and the following dimensions fill midpoints in
  breadth-first bisection order, conditionally on the two endpoints::

      W_m = ((t_r - t_m)*W_l + (t_m - t_l)*W_r)/(t_r - t_l)
            + sqrt((t_m - t_l)*(t_r - t_m)/(t_r - t_l)) * Z_k

  so the coarse, high-variance features of the path use the first and best
  distributed Sobol coordinates (Glasserman, Monte Carlo Methods in
  Financial Engineering, 2004, sec. 5.5). The map ``Z -> W`` is linear with
  ``A A^T = min(t_i, t_j)``, so ``W`` has the exact Brownian law for i.i.d.
  ``Z`` and ``log S_t = log S0 + (mu - sigma**2/2)*t + sigma*W_t`` is the
  exact GBM law at the grid points. ``steps`` above ``QMC_MAX_DIMENSION``
  (``Sobol.MAXDIM``, 21201 in SciPy 1.18) is rejected.

Error estimate: the points inside one scramble are dependent by design, so
the scramble is the independent unit; each scramble mean is unbiased up to
the ``2**-30`` quantization of the Sobol points (``bits=30``), negligible
(~1e-9 in ``E[Z]``). ``qmc_mean`` averages each block and
returns ``estimate = mean(block_means)`` with ``SE = std(block_means,
ddof=1)/sqrt(R)``. With ``R = 1`` there is no estimable standard error
(``None`` plus a note). The i.i.d. intervals of ``risk.intervals`` (Wilson,
order-statistic VaR, bootstrap CVaR) are **not** valid on QMC samples;
quantiles, VaR and CVaR need per-scramble estimates combined across the
``R`` replicates, which is not implemented here. Combining QMC with
antithetic sampling or with ``control_variate_mean`` is not supported. The
Student t scenario is rejected.

When it does not help: QMC gains rely on a smooth integrand with low
effective dimension. Discontinuous payoffs in several dimensions
(barriers), integrands with high effective dimension, small ``N`` and long
grids without the bridge reordering show much smaller gains, possibly none;
the ``R``-replicate SE stays valid in every case. Measured over 50 seeds
(``S0 = 100``, ``mu = 0.07``, ``sigma = 0.3``, ``T = 1``, ``R = 16``,
``N = 2**12`` against plain MC with ``R*N`` draws; each figure is the ratio
of the empirical MSEs against the analytic value over the 50 seeds, with a
relative uncertainty of roughly +-25%): terminal
mean about 5950, 252-step arithmetic-average mean about 14000 with the
bridge but about 580 with sequential increments, a one-dimensional digital
``1{S_T < 80}`` still about 9700, and a 252-step barrier indicator
``1{min S < 80}`` only about 13. These are single-configuration
measurements, not guarantees.

RNG streams and reproducibility: QMC mode does **not** consume ``self.rng``
(a later ``"none"`` call returns what it would have returned without the
QMC call). Scramble ``r`` of the ``c``-th QMC call (``c = 0, 1, ...``,
counted per simulator instance) uses
``default_rng(SeedSequence(entropy, spawn_key=(QMC_SCRAMBLE_STREAM, c, r)))``
where ``entropy`` is the instance seed, or, with ``seed=None``, fresh OS
entropy drawn once per instance. Repeated calls therefore give new
independent replicates, and a fresh instance with the same seed reproduces
the same sequence of calls. Equality across SciPy/NumPy versions is not
promised.

RNG streams: antithetic sampling consumes the instance's existing
``self.rng`` (PCG64 via ``default_rng``) and opens no new stream. It draws
``steps * m`` normals instead of ``steps * n_sims``, so its prices are not
elementwise comparable with ``"none"`` for the same seed.
"""

from __future__ import annotations

import math
import zlib
from numbers import Integral, Real
from typing import Literal, TypedDict

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc as _scipy_qmc

VarianceReduction = Literal["none", "antithetic", "qmc"]
VARIANCE_REDUCTION_METHODS: tuple[str, ...] = ("none", "antithetic", "qmc")


class AntitheticMean(TypedDict):
    estimate: float
    standard_error: float | None
    pairs: int


def validate_variance_reduction(value: object) -> str:
    """Return a supported sampling mode or raise ``ValueError``."""
    if not isinstance(value, str) or value not in VARIANCE_REDUCTION_METHODS:
        raise ValueError(
            "variance_reduction must be one of "
            + ", ".join(repr(name) for name in VARIANCE_REDUCTION_METHODS)
        )
    return value


def antithetic_mean(values: object) -> AntitheticMean:
    """Pair-based mean and standard error of an antithetic sample.

    ``values`` is a finite real 1-D array of length ``2m >= 2`` ordered as
    ``[originals..., mirrors...]`` (``values[i]`` pairs with
    ``values[i + m]``). Returns ``estimate = mean(Y)`` with
    ``Y_i = (values[i] + values[i + m]) / 2``, ``standard_error =
    std(Y, ddof=1) / sqrt(m)`` and ``pairs = m``. With one pair the sample
    standard deviation is undefined and ``standard_error`` is ``None``.

    Halving before adding keeps each ``Y_i`` finite for any finite inputs.
    The mean and standard deviation fall back to scaling by ``max |Y|``
    when the plain reductions would overflow.
    """
    array = np.asarray(values)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("values must be a real numeric array")
    array = array.astype(np.float64, copy=False)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size < 2 or array.size % 2:
        raise ValueError("values must have an even length of at least 2")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")

    m = array.size // 2
    with np.errstate(under="ignore"):
        pairs = 0.5 * array[:m] + 0.5 * array[m:]

    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(np.mean(pairs))
        spread = float(np.std(pairs, ddof=1)) if m > 1 else 0.0
    if not np.isfinite(estimate) or not np.isfinite(spread):
        scale = float(np.max(np.abs(pairs)))
        scaled = pairs / scale
        estimate = float(np.mean(scaled)) * scale
        spread = (float(np.std(scaled, ddof=1)) * scale) if m > 1 else 0.0
    if not np.isfinite(estimate) or not np.isfinite(spread):
        raise ValueError("antithetic mean exceeds finite precision")
    standard_error = spread / math.sqrt(m) if m > 1 else None
    return {"estimate": estimate, "standard_error": standard_error, "pairs": m}


def _nonnegative(name: str, value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _total_variance(sigma: float, horizon: float) -> float:
    # Scale before squaring: sigma**2 may overflow when sigma**2*T is finite.
    scale = sigma * math.sqrt(horizon)
    try:
        return scale * scale
    except OverflowError:  # pragma: no cover - float multiply returns inf
        return math.inf


def antithetic_terminal_vrf(sigma: float, horizon: float) -> float:
    """Theoretical VRF ``1 / (-expm1(-sigma**2*T))`` of the terminal mean.

    Ratio of the plain ``2m``-draw variance to the ``m``-pair antithetic
    variance of the GBM terminal-price mean (independent of ``s0`` and
    ``mu``). ``sigma`` in year^-1/2, ``horizon`` in years. The value tends to
    1 as ``sigma**2*T`` grows. When ``sigma*sqrt(T) == 0`` (or the variance
    is so small that the ratio overflows) both estimators have zero
    variance; the function returns ``math.inf``, the limit as
    ``sigma**2*T -> 0+``, and never ``nan``.
    """
    sigma = _nonnegative("sigma", sigma)
    horizon = _nonnegative("horizon", horizon)
    v = _total_variance(sigma, horizon)
    denominator = -math.expm1(-v)
    if denominator == 0:
        return math.inf
    try:
        return 1.0 / denominator
    except (OverflowError, ZeroDivisionError):  # pragma: no cover
        return math.inf


def antithetic_pair_covariance(s0: float, mu: float, sigma: float,
                               horizon: float) -> float:
    """Exact ``Cov(S+_T, S-_T) = S0**2*exp(2*mu*T)*expm1(-sigma**2*T)``.

    Equal to ``S0**2*exp(2*(mu - sigma**2/2)*T) - (S0*exp(mu*T))**2`` but
    evaluated in log space without cancellation. Price units squared; it is
    ``<= 0`` and zero only when ``sigma*sqrt(T) == 0``. A result that
    overflows, or underflows to zero although ``sigma**2*T > 0``, raises
    ``ValueError``.
    """
    if (isinstance(s0, bool) or not isinstance(s0, Real)
            or not math.isfinite(s0) or s0 <= 0):
        raise ValueError("s0 must be finite and positive")
    if isinstance(mu, bool) or not isinstance(mu, Real) or not math.isfinite(mu):
        raise ValueError("mu must be finite")
    sigma = _nonnegative("sigma", sigma)
    horizon = _nonnegative("horizon", horizon)
    v = _total_variance(sigma, horizon)
    if v == 0:
        return 0.0
    log_magnitude = (2.0 * math.log(s0) + 2.0 * float(mu) * horizon
                     + math.log(-math.expm1(-v)))
    if not math.isfinite(log_magnitude):
        raise ValueError("pair covariance exceeds finite precision")
    try:
        magnitude = math.exp(log_magnitude)
    except OverflowError:
        raise ValueError("pair covariance exceeds finite precision") from None
    if magnitude == 0:
        raise ValueError("pair covariance underflows to zero")
    return -magnitude


# --- Control variate (E2.2) -------------------------------------------------

ControlBetaSource = Literal["same_sample", "pilot", "fixed"]


class ControlVariateMean(TypedDict):
    estimate: float
    standard_error: float
    beta: float
    rho2: float | None
    vrf: float | None
    n: int
    n_pilot: int
    beta_source: ControlBetaSource
    bias: Literal["O(1/N)", "none"]
    warning: str | None


_CONSTANT_CONTROL_WARNING = (
    "control has zero sample variance; beta set to 0 and no variance "
    "reduction applied"
)


def _real_vector(name: str, values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must be a real numeric array")
    array = array.astype(np.float64, copy=False)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _finite_real(name: str, value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _power_of_two_scale(magnitude: float) -> float:
    # Largest power of two <= magnitude (1 for 0): dividing by it is exact,
    # keeps scaled values below 2 and preserves exact affine relations.
    if magnitude == 0:
        return 1.0
    return math.ldexp(1.0, math.frexp(magnitude)[1] - 1)


def _moments(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Sample ``Var(a)``, ``Var(b)``, ``Cov(a, b)`` with ``ddof=1``.

    A constant vector has variance exactly 0 (``np.mean`` of equal values
    may round away from the common value).
    """
    n = a.size
    da = a - np.mean(a) if np.ptp(a) > 0 else np.zeros_like(a)
    db = b - np.mean(b) if np.ptp(b) > 0 else np.zeros_like(b)
    with np.errstate(under="ignore"):
        return (float(da @ da) / (n - 1), float(db @ db) / (n - 1),
                float(da @ db) / (n - 1))


def control_variate_mean(values: object, control: object,
                         control_mean: float, *, beta: float | None = None,
                         pilot_fraction: float | None = None,
                         ) -> ControlVariateMean:
    """Control-variate estimate of ``E[Y]`` with a control of known mean.

    ``values`` (``Y``) and ``control`` (``X``) are finite real 1-D arrays of
    equal length ``N`` holding independent, identically distributed draws in
    sampling order; ``control_mean`` is the exact ``E[X]`` in the units of
    ``X``. The returned ``estimate``, ``standard_error`` and ``beta`` are in
    the units of ``Y`` (``beta`` in units of ``Y`` per unit of ``X``).

    ``beta`` and ``pilot_fraction`` are mutually exclusive:

    * neither: ``beta_source = "same_sample"``, ``bias = "O(1/N)"``,
      ``SE = std(e, ddof=2)/sqrt(N)``, requires ``N >= 3``;
    * ``pilot_fraction = f`` in ``(0, 1)``: ``beta_source = "pilot"``, the
      first ``floor(f*N)`` draws estimate ``beta`` and the rest form the
      estimate with ``SE = std(e, ddof=1)/sqrt(n)``; requires at least 2
      draws in each part; ``bias = "none"``;
    * ``beta`` given: ``beta_source = "fixed"``, ``SE`` with ``ddof=1``,
      requires ``N >= 2``; ``bias = "none"``.

    ``n`` is the number of draws in ``estimate`` and ``standard_error`` and
    ``n_pilot`` the number used only for ``beta`` (``n + n_pilot = N``).
    ``rho2`` and ``vrf`` are defined in the module docstring; they are
    ``None`` when undefined (constant ``X`` or ``Y``, or zero residual
    variance). ``warning`` reports a constant control (``beta = 0`` in the
    estimated modes) and is otherwise ``None``. Every float in the result is
    finite.

    Only expectations: do not use it for quantiles, VaR or CVaR. Antithetic
    samples are not supported (dependent pairs). It does not help when ``Y``
    is weakly correlated with ``X`` (VRF ``1/(1 - rho**2)`` near 1), e.g.
    deep out-of-the-money calls, far puts or digitals on ``S_T``.

    Computations run on power-of-two rescaled copies of ``X`` and ``Y`` so
    sums, variances and covariances do not overflow; a result that still
    exceeds finite precision raises ``ValueError``.

    The standard error and any normal interval built from it are asymptotic
    and under-cover at small ``N`` (e.g. for an at-the-money call under P a
    nominal 95% interval covered about 0.76 at N=10 and 0.89 at N=30 with the
    same-sample beta, approaching 0.95 only near N=1000).
    """
    y = _real_vector("values", values)
    x = _real_vector("control", control)
    if x.size != y.size:
        raise ValueError("values and control must have the same length")
    mean_x = _finite_real("control_mean", control_mean)
    if beta is not None and pilot_fraction is not None:
        raise ValueError("beta and pilot_fraction are mutually exclusive")

    size = y.size
    source: ControlBetaSource
    if beta is not None:
        fixed_beta = _finite_real("beta", beta)
        if size < 2:
            raise ValueError("a fixed beta needs at least 2 observations")
        source, n_pilot, ddof = "fixed", 0, 1
    elif pilot_fraction is not None:
        fraction = _finite_real("pilot_fraction", pilot_fraction)
        if not 0 < fraction < 1:
            raise ValueError("pilot_fraction must lie in (0, 1)")
        n_pilot = int(math.floor(fraction * size))
        if n_pilot < 2 or size - n_pilot < 2:
            raise ValueError(
                "pilot_fraction must leave at least 2 observations in the "
                "pilot and 2 in the main sample"
            )
        source, ddof = "pilot", 1
    else:
        if size < 3:
            raise ValueError("a same-sample beta needs at least 3 observations")
        source, n_pilot, ddof = "same_sample", 0, 2

    scale_x = _power_of_two_scale(max(float(np.max(np.abs(x))), abs(mean_x)))
    scale_y = _power_of_two_scale(float(np.max(np.abs(y))))
    a = x / scale_x
    b = y / scale_y
    centre = mean_x / scale_x

    warning: str | None = None
    if source == "fixed":
        with np.errstate(over="ignore", under="ignore"):
            beta_scaled = fixed_beta * (scale_x / scale_y)
        if not math.isfinite(beta_scaled):
            raise ValueError("beta exceeds finite precision for these data")
    else:
        fit_a, fit_b = (a[:n_pilot], b[:n_pilot]) if source == "pilot" else (a, b)
        var_fit, _, cov_fit = _moments(fit_a, fit_b)
        if var_fit == 0:
            beta_scaled = 0.0
            warning = _CONSTANT_CONTROL_WARNING
        else:
            beta_scaled = cov_fit / var_fit

    main_a, main_b = a[n_pilot:], b[n_pilot:]
    n = main_b.size
    var_a, var_b, cov_ab = _moments(main_a, main_b)
    if var_a == 0 and warning is None:
        warning = _CONSTANT_CONTROL_WARNING if source != "fixed" else (
            "control has zero sample variance; the correction is a constant "
            "shift and rho2 is undefined"
        )

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        estimate_scaled = (float(np.mean(main_b))
                           - beta_scaled * (float(np.mean(main_a)) - centre))
        residual = main_b - beta_scaled * main_a
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                "control-variate residual exceeds finite precision"
            )
        if np.ptp(residual) > 0:
            centred = residual - np.mean(residual)
            residual_ss = float(centred @ centred)
        else:
            residual_ss = 0.0
        estimate = estimate_scaled * scale_y
        standard_error = (math.sqrt(residual_ss / (n - ddof) / n) * scale_y)
        beta_value = beta_scaled * (scale_y / scale_x)
    if source == "fixed":
        beta_value = fixed_beta
    if not all(math.isfinite(v) for v in (estimate, standard_error, beta_value,
                                          residual_ss)):
        raise ValueError("control-variate estimate exceeds finite precision")

    rho2: float | None = None
    if var_a > 0 and var_b > 0:
        rho2 = min(1.0, max(0.0, (cov_ab / var_a) * (cov_ab / var_b)))
    residual_var = residual_ss / (n - 1)
    vrf: float | None = None
    if var_a > 0:
        if residual_var > 0:
            ratio = var_b / residual_var
            vrf = ratio if math.isfinite(ratio) else None
        elif var_b > 0:
            rho2 = 1.0  # exact affine relation: zero residual variance

    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "beta": beta_value,
        "rho2": rho2,
        "vrf": vrf,
        "n": n,
        "n_pilot": n_pilot,
        "beta_source": source,
        "bias": "O(1/N)" if source == "same_sample" else "none",
        "warning": warning,
    }


def gbm_terminal_control_mean(s0: float, mu: float, horizon: float) -> float:
    """Exact ``E[S_T] = S0*exp(mu*T)`` of the GBM, for ``X = S_T``.

    Physical measure P: ``mu`` is the price drift in year^-1, ``horizon``
    ``T`` in years, ``s0`` and the result in the asset's currency. Under the
    risk-neutral measure Q (E3) the control would be the discounted
    ``exp(-r*T)*S_T`` with mean ``S0*exp(-q*T)`` instead. Evaluated in log
    space; a result that overflows or underflows to zero raises
    ``ValueError``.
    """
    if (isinstance(s0, bool) or not isinstance(s0, Real)
            or not math.isfinite(s0) or s0 <= 0):
        raise ValueError("s0 must be finite and positive")
    mu = _finite_real("mu", mu)
    horizon = _nonnegative("horizon", horizon)
    log_mean = math.log(s0) + mu * horizon
    if not math.isfinite(log_mean):
        raise ValueError("terminal mean exceeds finite precision")
    try:
        mean = math.exp(log_mean)
    except OverflowError:
        raise ValueError("terminal mean exceeds finite precision") from None
    if mean == 0:
        raise ValueError("terminal mean underflows to zero")
    return mean


# --- Randomized quasi-Monte Carlo (E2.3) ------------------------------------

# Integer label of the scramble stream: ``SeedSequence`` rejects strings in
# ``spawn_key``, so the label "qmc_scramble" is hashed as in risk.intervals.
QMC_SCRAMBLE_STREAM = zlib.crc32(b"qmc_scramble")
QMC_SOBOL_BITS = 30
# Sobol dimension limit of the installed SciPy (21201 in SciPy 1.18).
QMC_MAX_DIMENSION: int = int(getattr(_scipy_qmc.Sobol, "MAXDIM", 21201))
QMC_CLIP_EPSILON = 2.0 ** -53
QMC_RNG_CONVENTION = (
    "scipy.stats.qmc.Sobol(scramble=True: linear matrix scramble + digital "
    f"random shift, bits={QMC_SOBOL_BITS}, random_base2); scramble r of call "
    "c uses default_rng(SeedSequence(entropy, spawn_key=(QMC_SCRAMBLE_STREAM, "
    "c, r))) PCG64; u clipped to [2**-53, 1-2**-53]; Z = scipy.special.ndtri(u); "
    "paths by Brownian bridge (dimension 0 -> W_T, then BFS midpoints)"
)


class QmcMean(TypedDict):
    estimate: float
    standard_error: float | None
    scrambles: int
    points_per_scramble: int
    note: str | None


_SINGLE_SCRAMBLE_NOTE = (
    "a single scramble has no estimable standard error; use at least 2 "
    "independent scrambles"
)


def validate_qmc_scrambles(value: object) -> int:
    """Return the number of scrambles ``R >= 1`` or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("qmc_scrambles must be a positive integer")
    return int(value)


def qmc_points_per_scramble(n_sims: int, scrambles: int) -> int:
    """Return ``N = n_sims / R`` after checking it is a power of two."""
    scrambles = validate_qmc_scrambles(scrambles)
    if n_sims % scrambles:
        raise ValueError(
            f"qmc requires n_sims divisible by qmc_scrambles; got n_sims="
            f"{n_sims}, qmc_scrambles={scrambles}"
        )
    points = n_sims // scrambles
    if points < 1 or points & (points - 1):
        raise ValueError(
            f"qmc requires n_sims / qmc_scrambles to be a power of two; got "
            f"{points} points per scramble"
        )
    if points.bit_length() - 1 > QMC_SOBOL_BITS:
        raise ValueError(
            f"qmc supports at most 2**{QMC_SOBOL_BITS} points per scramble"
        )
    return points


def qmc_scramble_rng(entropy: int | list[int] | tuple[int, ...], call: int,
                     scramble: int) -> np.random.Generator:
    """PCG64 generator of scramble ``scramble`` in QMC call ``call``."""
    return np.random.default_rng(np.random.SeedSequence(
        entropy, spawn_key=(QMC_SCRAMBLE_STREAM, int(call), int(scramble))))


def sobol_normals(dimension: int, log2_points: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Scrambled Sobol normals, shape ``(2**log2_points, dimension)``.

    Draws ``random_base2(log2_points)`` from a fresh ``Sobol(dimension,
    scramble=True, bits=QMC_SOBOL_BITS, rng=rng)``, clips ``u`` to
    ``[2**-53, 1 - 2**-53]`` and applies ``ndtri`` in place, so every value
    is finite (``|Z| <= 8.3``). SciPy's peak storage is the larger of its
    scrambling set-up and two arrays of the returned size; see
    ``qmc_sobol_bytes``.
    """
    if not 1 <= dimension <= QMC_MAX_DIMENSION:
        raise ValueError(
            f"Sobol dimension must lie in [1, {QMC_MAX_DIMENSION}]; got {dimension}"
        )
    sampler = _scipy_qmc.Sobol(dimension, scramble=True, bits=QMC_SOBOL_BITS,
                               rng=rng)
    u = sampler.random_base2(log2_points)
    np.clip(u, QMC_CLIP_EPSILON, 1.0 - QMC_CLIP_EPSILON, out=u)
    ndtri(u, out=u)
    return u


def qmc_sobol_bytes(dimension: int, points: int) -> int:
    """Upper estimate of SciPy's peak storage in one ``sobol_normals`` call.

    ``Sobol(scramble=True, bits=QMC_SOBOL_BITS)`` first builds ``dimension``
    lower-triangular ``bits x bits`` int64 scrambling matrices
    (``8*bits**2`` bytes per dimension, plus temporaries: 7.3-7.5 KB per
    dimension traced with SciPy 1.18) and frees them before drawing; the
    draw then holds two ``(points, dimension)`` float64 arrays plus about
    140 bytes of state per dimension. The peak is the larger phase. A
    fixed overhead of a few KiB per call and SciPy's one-time load of its
    direction-number table (~6 MiB transient, once per process) are not
    included.
    """
    setup = 8 * QMC_SOBOL_BITS ** 2 + 512
    draw = 2 * 8 * int(points) + 256
    return int(dimension) * max(setup, draw)


BridgePlan = tuple[tuple[int, int, int, float, float, float], ...]


def brownian_bridge_plan(steps: int, horizon: float) -> BridgePlan:
    """Breadth-first bisection plan on the grid ``t_k = k*horizon/steps``.

    Entry ``j`` (dimension ``j + 1``) is ``(m, l, r, w_l, w_r, sd)``: grid
    point ``m`` is filled from the already built endpoints ``l < m < r`` as
    ``W_m = w_l*W_l + w_r*W_r + sd*Z_{j+1}``. Point ``steps`` is built from
    dimension 0 and point 0 is ``W_0 = 0``. The plan has ``steps - 1``
    entries. ``horizon`` is in years.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (isinstance(horizon, bool) or not isinstance(horizon, Real)
            or not math.isfinite(horizon) or horizon <= 0):
        raise ValueError("horizon must be finite and positive")
    steps, horizon = int(steps), float(horizon)
    dt = horizon / steps
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("horizon / steps must be finite and positive")

    def time(k: int) -> float:
        return horizon if k == steps else k * dt

    plan = []
    queue = [(0, steps)]
    head = 0
    while head < len(queue):
        left, right = queue[head]
        head += 1
        if right - left < 2:
            continue
        mid = (left + right) // 2
        t_l, t_m, t_r = time(left), time(mid), time(right)
        span = t_r - t_l
        plan.append((mid, left, right, (t_r - t_m) / span, (t_m - t_l) / span,
                     math.sqrt((t_m - t_l) * (t_r - t_m) / span)))
        queue.append((left, mid))
        queue.append((mid, right))
    return tuple(plan)


def brownian_bridge(z: object, horizon: float, *, out: np.ndarray | None = None,
                    plan: BridgePlan | None = None) -> np.ndarray:
    """Map standard normals to Brownian values on ``t_k = k*horizon/steps``.

    ``z`` is a finite real array of shape ``(steps, N)`` whose row ``j`` is
    Sobol dimension ``j``; the result ``W`` has the same shape and row
    ``k - 1`` holds ``W_{t_k}`` for ``k = 1..steps`` (``W_0 = 0`` is not
    stored). Row 0 of ``z`` gives ``W_T = sqrt(horizon)*z[0]`` and the other
    rows fill midpoints as in ``brownian_bridge_plan``. The map is linear
    with ``Cov(W_{t_i}, W_{t_j}) = min(t_i, t_j)`` for i.i.d. ``N(0, 1)``
    rows, so the law is exact Brownian motion on the grid; time in years,
    ``W`` in year^1/2.

    ``out`` (shape ``(steps, N)``, float64, may be a non-contiguous view and
    must not overlap ``z``) receives the result without another full-size
    array. Memory: the input plus the output, ``O(steps*N)``.
    """
    array = np.asarray(z)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("z must be a real numeric array")
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("z must have shape (steps, N) with steps >= 1")
    steps = array.shape[0]
    if plan is None:
        plan = brownian_bridge_plan(steps, horizon)
    elif len(plan) != steps - 1:
        raise ValueError("plan does not match the number of steps")
    if out is None:
        out = np.empty(array.shape, dtype=np.float64)
    elif out.shape != array.shape or out.dtype != np.float64:
        raise ValueError("out must be float64 with the shape of z")
    if np.shares_memory(out, array):
        raise ValueError("out must not overlap z")

    np.multiply(array[0], math.sqrt(float(horizon)), out=out[steps - 1])
    for row, (mid, left, right, w_left, w_right, sd) in enumerate(plan, start=1):
        target = out[mid - 1]
        np.multiply(array[row], sd, out=target)
        if left:
            target += w_left * out[left - 1]
        target += w_right * out[right - 1]
    return out


def qmc_mean(values: object, scrambles: int) -> QmcMean:
    """Randomized-QMC mean and its standard error across scrambles.

    ``values`` is a finite real 1-D array of length ``R*N`` in the QMC
    layout: scramble ``r`` occupies ``values[r*N:(r+1)*N]`` (``N`` points
    per scramble, any ``N >= 1`` is accepted here). Each block mean
    ``m_r`` is an unbiased estimate of the expectation, up to the ``2**-30``
    quantization of the Sobol points (``bits=30``), negligible (~1e-9 in
    ``E[Z]``), and the blocks are independent; ``estimate = mean(m_r)`` and
    ``standard_error = std(m_r, ddof=1)/sqrt(R)``. With ``R = 1`` the
    standard error is ``None`` and ``note`` explains why; otherwise ``note``
    is ``None``. ``standard_error**2`` is unbiased, but a ``t_{R-1}``
    interval can under-cover because the block means of terminal prices
    are skewed and heavy-tailed: a nominal 95% interval covered 0.929 with
    ``R = 16, N = 64`` and 0.885 with ``R = 8, N = 256`` (4000 seeds).

    Only expectations: the i.i.d. intervals of ``risk.intervals`` (Wilson,
    order-statistic VaR, bootstrap CVaR) do **not** hold on QMC points.
    Quantiles, VaR and CVaR require per-scramble estimates combined over
    the ``R`` replicates, not implemented here. Antithetic or control
    variates combined with QMC are not supported. The gain over plain Monte
    Carlo is much smaller for discontinuous path functionals (barriers),
    high effective-dimension integrands, small ``N`` or long grids without
    a bridge.
    """
    scrambles = validate_qmc_scrambles(scrambles)
    array = np.asarray(values)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("values must be a real numeric array")
    array = array.astype(np.float64, copy=False)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size < scrambles or array.size % scrambles:
        raise ValueError("values length must be a positive multiple of scrambles")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")

    points = array.size // scrambles
    blocks = array.reshape(scrambles, points)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        means = np.mean(blocks, axis=1)
        if not np.isfinite(means).all():
            scale = float(np.max(np.abs(array)))
            means = np.mean(blocks / scale, axis=1) * scale
        estimate = float(np.mean(means))
        spread = float(np.std(means, ddof=1)) if scrambles > 1 else 0.0
        if not math.isfinite(estimate) or not math.isfinite(spread):
            scale = float(np.max(np.abs(means)))
            estimate = float(np.mean(means / scale)) * scale
            spread = (float(np.std(means / scale, ddof=1)) * scale
                      if scrambles > 1 else 0.0)
    if not math.isfinite(estimate) or not math.isfinite(spread):
        raise ValueError("qmc mean exceeds finite precision")
    if scrambles == 1:
        return {"estimate": estimate, "standard_error": None, "scrambles": 1,
                "points_per_scramble": points, "note": _SINGLE_SCRAMBLE_NOTE}
    return {"estimate": estimate, "standard_error": spread / math.sqrt(scrambles),
            "scrambles": scrambles, "points_per_scramble": points, "note": None}
