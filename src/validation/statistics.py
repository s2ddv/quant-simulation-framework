"""Analytical terminal GBM benchmarks and nested-sample convergence."""

import math
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from simulation.variance_reduction import (antithetic_mean, control_variate_mean,
                                           qmc_mean)

QUANTILES = (0.05, 0.5, 0.95)


def terminal_theory(s0, mu, sigma, horizon):
    """GBM moments and quantiles, calculated in log space.

    Price is in asset currency, horizon in years, mu in year^-1 and sigma
    in year^-1/2. Positive moments/quantiles must remain representable as
    float64; a genuinely zero variance is reserved for sigma=0 or T=0.
    Intermediate squares and exponentials must not restrict that range.
    """
    if (not np.isfinite([s0, mu, sigma, horizon]).all()
            or s0 <= 0 or sigma < 0 or horizon < 0):
        raise ValueError("Require finite parameters, s0 > 0, sigma/horizon >= 0.")
    s0, mu, sigma, horizon = map(float, (s0, mu, sigma, horizon))
    message = "Theoretical terminal statistics exceed finite precision."

    def positive_exp(log_value):
        try:
            value = math.exp(log_value)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(value) or value == 0:
            raise ValueError(message)
        return value

    log_mean = math.log(s0) + mu * horizon
    mean = s0 if mu == 0 or horizon == 0 else positive_exp(log_mean)
    if sigma == 0 or horizon == 0:
        return {"mean": mean, "variance": 0.0,
                "quantiles": {str(p): mean for p in QUANTILES}}

    # v = sigma^2*T may underflow even when mean^2*expm1(v) is representable.
    log_v = 2 * math.log(sigma) + math.log(horizon)
    try:
        v = math.exp(log_v)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if log_v < -36:
        # log(expm1(v)) = log(v) + O(v), below float64 relative precision.
        log_expm1_v = log_v
    elif v <= 1:
        log_expm1_v = math.log(math.expm1(v))
    else:
        log_expm1_v = v + math.log1p(-math.exp(-v))
    variance = positive_exp(2 * log_mean + log_expm1_v)
    standard_deviation_log = math.exp(log_v / 2)
    quantiles = {
        str(p): positive_exp(log_mean - v / 2
                            + standard_deviation_log * NormalDist().inv_cdf(p))
        for p in QUANTILES
    }
    return {"mean": mean, "variance": variance, "quantiles": quantiles}


def _comparison(estimate, theoretical):
    error = None if estimate is None else abs(estimate - theoretical)
    return {"estimate": estimate, "theoretical": theoretical,
            "absolute_error": error,
            "relative_error": (error / abs(theoretical)
                               if error is not None and theoretical != 0 else None)}


def compare_terminal(terminal, theory):
    """Use unbiased sample variance (ddof=1) and linear empirical quantiles."""
    terminal = np.asarray(terminal, dtype=float)
    if (terminal.ndim != 1 or terminal.size == 0
            or not np.isfinite(terminal).all() or np.any(terminal <= 0)):
        raise ValueError("terminal must be a nonempty vector of finite positive prices.")
    return {
        "mean": _comparison(float(terminal.mean()), theory["mean"]),
        "variance": _comparison(float(terminal.var(ddof=1))
                                if terminal.size > 1 else None, theory["variance"]),
        "quantiles": {
            str(p): _comparison(float(np.quantile(terminal, p, method="linear")),
                                theory["quantiles"][str(p)]) for p in QUANTILES
        },
    }


def convergence_analysis(terminal, theory):
    """Compare increasing prefixes of ONE run, not separately reseeded runs.

    The reference SE is the exact RMSE of the sample mean under fixed GBM
    parameters, not an envelope or a guarantee of monotonic observed error.
    """
    terminal = np.asarray(terminal, dtype=float)
    compare_terminal(terminal, theory)  # Validate before constructing checkpoints.
    counts = np.unique(np.geomspace(min(2, terminal.size), terminal.size,
                                   num=min(12, terminal.size)).astype(int))
    counts = np.unique(np.append(counts, terminal.size))
    return [{"n_sims": int(n), **compare_terminal(terminal[:n], theory),
             "mean_standard_error": float(np.sqrt(theory["variance"] / n))}
            for n in counts]


def terminal_quantile(s0, mu, sigma, horizon, p):
    """Analytical ``p``-quantile of the GBM terminal price ``S_T``.

    ``Q_p(S_T) = exp(ln s0 + (mu - sigma^2/2)*T + sigma*sqrt(T)*Phi^-1(p))``,
    evaluated in log space with the same expression and guards as
    ``terminal_theory`` (so p in ``QUANTILES`` reproduces its values). Price in
    asset currency, horizon T in years, mu in year^-1, sigma in year^-1/2;
    ``p`` is a real probability strictly inside (0, 1). With sigma=0 or T=0
    the law is degenerate and every quantile equals the deterministic mean.
    Unlike ``terminal_theory``, the variance is not formed, so a quantile can
    be returned even when the variance itself is not representable.

    Raises:
        ValueError: On invalid parameters or ``p`` (bool rejected), or if the
            quantile overflows or underflows to zero in float64.
    """
    for name, value in (("s0", s0), ("mu", mu), ("sigma", sigma),
                        ("horizon", horizon)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                f"{name} must be a real number (bool rejected), "
                f"got {type(value).__name__}."
            )
    try:
        s0, mu, sigma, horizon = map(float, (s0, mu, sigma, horizon))
    except OverflowError:
        for name, value in (("s0", s0), ("mu", mu), ("sigma", sigma),
                            ("horizon", horizon)):
            try:
                float(value)
            except OverflowError:
                raise ValueError(
                    f"{name} must be representable as a finite float."
                ) from None
        raise
    if (not np.isfinite([s0, mu, sigma, horizon]).all()
            or s0 <= 0 or sigma < 0 or horizon < 0):
        raise ValueError("Require finite parameters, s0 > 0, sigma/horizon >= 0.")
    if (isinstance(p, (bool, np.bool_)) or not isinstance(p, Real)
            or not 0 < p < 1):
        raise ValueError("p must be a real number strictly between 0 and 1.")
    s0, mu, sigma, horizon, p = map(float, (s0, mu, sigma, horizon, p))
    message = "Theoretical terminal quantile exceeds finite precision."

    def positive_exp(log_value):
        try:
            value = math.exp(log_value)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(value) or value == 0:
            raise ValueError(message)
        return value

    log_mean = math.log(s0) + mu * horizon
    if sigma == 0 or horizon == 0:
        return s0 if mu == 0 or horizon == 0 else positive_exp(log_mean)
    log_v = 2 * math.log(sigma) + math.log(horizon)
    try:
        v = math.exp(log_v)
    except OverflowError as exc:
        raise ValueError(message) from exc
    standard_deviation_log = math.exp(log_v / 2)
    return positive_exp(log_mean - v / 2
                        + standard_deviation_log * NormalDist().inv_cdf(p))


# --- Convergence with an external estimator (E2.4) ---------------------------

CONVERGENCE_SCHEDULES = ("geometric", "power2")


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer (bool rejected).")
    return int(value)


def _real_units(name, values, ndim=None):
    array = np.asarray(values)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.integer)
                                       or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must be a real numeric array.")
    array = array.astype(np.float64, copy=False)
    if array.ndim < 1 or (ndim is not None and array.ndim != ndim):
        expected = "at least one dimension" if ndim is None else f"{ndim} dimension(s)"
        raise ValueError(f"{name} must have {expected}.")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must hold at least one unit along axis 0.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return array


def _checkpoint_counts(n_units, counts, schedule, min_units):
    if min_units > n_units:
        raise ValueError(f"min_units={min_units} exceeds the {n_units} available units.")
    power2 = schedule == "power2"
    if power2 and n_units & (n_units - 1):
        raise ValueError(
            f"schedule='power2' requires a power-of-two number of units; got {n_units}.")
    if counts is None:
        if power2:
            generated = [1 << k for k in range(n_units.bit_length())]
            return [n for n in generated if n >= min_units]
        start = max(min(2, n_units), min_units)
        grid = np.geomspace(start, n_units, num=min(12, n_units)).astype(int)
        return [int(n) for n in np.unique(np.append(grid, n_units))]
    # np.asarray turns a mixed list such as [True, 10] into integers.
    if any(isinstance(n, (bool, np.bool_))
           for n in np.asarray(counts, dtype=object).reshape(-1)):
        raise ValueError("counts must be a nonempty 1-D sequence of integers.")
    array = np.asarray(counts)
    if (array.ndim != 1 or array.size == 0 or array.dtype == np.bool_
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError("counts must be a nonempty 1-D sequence of integers.")
    values = [int(n) for n in array]
    if values[0] < min_units or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(
            f"counts must be strictly increasing integers >= min_units={min_units}.")
    if values[-1] != n_units:
        raise ValueError(f"counts must end at the number of units ({n_units}).")
    if power2 and any(n & (n - 1) for n in values):
        raise ValueError("schedule='power2' requires every count to be a power of two.")
    return values


def _finite_number(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{what} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{what} is not finite ({value}).")
    return value


def convergence_with_estimator(units, estimator, *, theoretical=None, counts=None,
                               schedule="geometric", evaluations_per_unit=1,
                               min_units=1):
    """Convergence of an external estimator over nested prefixes of ONE run.

    Generalizes ``convergence_analysis`` (which stays unchanged) to
    estimators whose independent unit is not a single draw:

    * i.i.d. ("none"): ``units`` is the 1-D vector of values ``f(S_i)``;
      use ``mean_estimator``.
    * antithetic: ``units`` has shape ``(m, 2)`` with row ``i`` the pair
      ``(f(S+_i), f(S-_i))``, i.e. ``antithetic_units(values)`` for values in
      the ``[originals..., mirrors...]`` order of ``simulation.gbm``; use
      ``antithetic_estimator`` and ``evaluations_per_unit=2``. Prefixes are
      always prefixes of *pairs*: a prefix of the flat vector would take
      originals without their mirrors.
    * QMC: ``units`` has shape ``(N, R)``, row ``i`` holding point ``i`` of
      every scramble (``qmc_units(values, R)``); use ``qmc_estimator``,
      ``schedule="power2"`` and ``evaluations_per_unit=R``. Only power-of-two
      prefixes of a scrambled Sobol sequence keep the balance of the
      ``(t, m, d)``-net; other prefixes are valid but lose the QMC rate.
    * control variate: ``units`` has shape ``(N, 2)`` with columns ``Y`` and
      ``X`` (``np.stack([y, x], axis=1)``); use ``control_estimator`` and
      ``min_units=3`` for the same-sample beta. ``beta`` is re-estimated on
      every prefix, so each checkpoint carries its own ``O(1/n)`` bias.

    ``estimator(prefix)`` receives a read-only view ``units[:n]`` and returns
    a mapping with ``"estimate"`` (finite real) and ``"standard_error"``
    (finite real ``>= 0`` or ``None`` when not estimable, e.g. one pair or
    one scramble). ``counts`` lists explicit checkpoints (strictly
    increasing integers ``>= min_units`` ending at ``n_units``); otherwise
    they are generated by ``schedule``: ``"geometric"`` gives up to 12
    log-spaced counts from ``max(min(2, n_units), min_units)`` plus the
    total, the same grid as ``convergence_analysis`` when ``min_units <= 2``;
    ``"power2"`` gives ``1, 2, 4, ..., n_units`` (``>= min_units``) and
    requires ``n_units`` (and any explicit count) to be a power of two.

    Each checkpoint is ``{"n_units", "n_evaluations", "estimate",
    "standard_error", "absolute_error", "relative_error"}``, where
    ``n_evaluations = n_units * evaluations_per_unit`` is the number of
    integrand evaluations, so techniques are compared at equal cost. The
    errors are ``None`` without ``theoretical``; ``relative_error`` is also
    ``None`` when ``theoretical == 0``. The result is JSON-safe with
    ``allow_nan=False``: an estimator that returns ``nan``/``inf`` (or an
    error that overflows) raises ``ValueError`` instead of leaking a
    non-finite value, and a ``None`` standard error is kept as ``None``. In
    particular infinite diagnostic ratios such as
    ``antithetic_terminal_vrf(0, T) = inf`` never reach a report through
    this function, which carries only the estimate and its SE.

    Prefixes are nested, hence correlated samples of the same realization:
    the observed error need not fall monotonically, and comparisons across
    checkpoints are not independent replications. The standard errors are
    those of the chosen estimator on each prefix; the i.i.d. intervals of
    ``risk.intervals`` are not valid for antithetic or QMC units.
    """
    if not callable(estimator):
        raise ValueError("estimator must be callable.")
    if not isinstance(schedule, str) or schedule not in CONVERGENCE_SCHEDULES:
        raise ValueError(f"schedule must be one of {CONVERGENCE_SCHEDULES}.")
    evaluations_per_unit = _positive_int("evaluations_per_unit", evaluations_per_unit)
    min_units = _positive_int("min_units", min_units)
    array = _real_units("units", units)
    if theoretical is not None:
        theoretical = _finite_number(theoretical, "theoretical")
    view = array.view()
    view.flags.writeable = False
    checkpoints = []
    for n in _checkpoint_counts(array.shape[0], counts, schedule, min_units):
        result = estimator(view[:n])
        try:
            raw_estimate, raw_se = result["estimate"], result["standard_error"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "estimator must return a mapping with 'estimate' and "
                "'standard_error'.") from exc
        estimate = _finite_number(raw_estimate, f"estimate at n_units={n}")
        standard_error = None
        if raw_se is not None:
            standard_error = _finite_number(raw_se, f"standard_error at n_units={n}")
            if standard_error < 0:
                raise ValueError(f"standard_error at n_units={n} is negative.")
        absolute_error = relative_error = None
        if theoretical is not None:
            absolute_error = _finite_number(abs(estimate - theoretical),
                                            f"absolute_error at n_units={n}")
            if theoretical != 0:
                relative_error = _finite_number(absolute_error / abs(theoretical),
                                                f"relative_error at n_units={n}")
        checkpoints.append({
            "n_units": n, "n_evaluations": n * evaluations_per_unit,
            "estimate": estimate, "standard_error": standard_error,
            "absolute_error": absolute_error, "relative_error": relative_error,
        })
    return checkpoints


def antithetic_units(values):
    """Reshape ``[originals..., mirrors...]`` (length ``2m``) into ``(m, 2)`` pairs."""
    array = _real_units("values", values, ndim=1)
    if array.size % 2:
        raise ValueError("antithetic values must have an even length.")
    m = array.size // 2
    return np.stack([array[:m], array[m:]], axis=1)


def qmc_units(values, scrambles):
    """Reshape the QMC layout (scramble ``r`` in ``[r*N, (r+1)*N)``) into ``(N, R)``."""
    scrambles = _positive_int("scrambles", scrambles)
    array = _real_units("values", values, ndim=1)
    if array.size % scrambles:
        raise ValueError("values length must be a multiple of scrambles.")
    return array.reshape(scrambles, array.size // scrambles).T


def mean_estimator(prefix):
    """i.i.d. sample mean and ``SE = std(ddof=1)/sqrt(n)`` (``None`` for n=1).

    As in ``antithetic_mean``, large finite values whose plain reductions
    overflow are rescaled by ``max |x|``; a result that is still not finite
    raises ``ValueError``.
    """
    array = _real_units("prefix", prefix, ndim=1)
    root_n = math.sqrt(array.size)
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(np.mean(array))
        se = float(np.std(array, ddof=1)) / root_n if array.size > 1 else None
        if not math.isfinite(estimate) or (se is not None and not math.isfinite(se)):
            scale = float(np.max(np.abs(array)))
            scaled = array / scale
            estimate = float(np.mean(scaled)) * scale
            # Divide by sqrt(n) before rescaling: std itself may exceed float64.
            se = (float(np.std(scaled, ddof=1)) / root_n * scale
                  if array.size > 1 else None)
    if not math.isfinite(estimate) or (se is not None and not math.isfinite(se)):
        raise ValueError("mean_estimator result exceeds finite precision.")
    return {"estimate": estimate, "standard_error": se}


def antithetic_estimator(prefix):
    """``antithetic_mean`` on ``(m, 2)`` pair units (see ``antithetic_units``)."""
    array = _real_units("prefix", prefix, ndim=2)
    if array.shape[1] != 2:
        raise ValueError("antithetic units must have shape (m, 2).")
    return antithetic_mean(np.concatenate([array[:, 0], array[:, 1]]))


def qmc_estimator(prefix):
    """``qmc_mean`` on ``(N, R)`` units: the first ``N`` points of each scramble."""
    array = _real_units("prefix", prefix, ndim=2)
    return qmc_mean(array.T.reshape(-1), array.shape[1])


def control_estimator(control_mean, *, beta=None, pilot_fraction=None):
    """Return an estimator applying ``control_variate_mean`` to ``(N, 2)`` units.

    Columns are ``Y`` (target) and ``X`` (control with exact mean
    ``control_mean``). With the default same-sample beta use
    ``min_units=3``; ``beta`` is refitted on each prefix.
    """
    def estimator(prefix):
        array = _real_units("prefix", prefix, ndim=2)
        if array.shape[1] != 2:
            raise ValueError("control units must have shape (N, 2) with columns Y, X.")
        return control_variate_mean(array[:, 0], array[:, 1], control_mean,
                                    beta=beta, pilot_fraction=pilot_fraction)
    return estimator
