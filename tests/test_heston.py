"""Law-level tests for the Heston path simulator ``simulation.heston`` (E5.1).

Model (``HestonSimulator``)::

    dS = mu S dt + sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,
    d<W1, W2> = rho dt

Units: price in the asset's currency, time in years, ``mu``, ``kappa``, ``v``,
``theta``, ``xi`` in year^-1, ``rho`` dimensionless. Risk-neutral pricing uses
``mu = r - q`` and discounts by ``e^{-rT}`` outside the simulator.

Independent references (written here or in ``tests/test_heston_theory.py``,
which checks ``validation.heston_theory`` against ``scipy.integrate.quad``):

* CIR moments ``E[v_t] = theta + (v0 - theta) e^{-kappa t}`` and
  ``Var[v_t] = v0 xi^2 e^{-kt}(1 - e^{-kt})/k + theta xi^2 (1 - e^{-kt})^2/(2k)``
  written below (``_cir_mean``, ``_cir_var``).
* Moments of the plain Euler recursion of ``v`` (no truncation):
  ``m_{k+1} = m_k + kappa (theta - m_k) dt``,
  ``V_{k+1} = (1 - kappa dt)^2 V_k + xi^2 dt m_k``. Full truncation equals this
  recursion while ``v~`` stays positive (checked through the fraction of
  truncated values, which is reported in the output as exact zeros).
* GBM law of ``ln S_T`` for ``xi -> 0`` with ``v0 = theta``:
  ``N(ln S0 + (mu - theta/2) T, theta T)``.
* ``E[S_T] = S0 e^{mu T}`` (both QE-M and log-Euler full truncation are
  discrete martingales).
* Semi-analytic European prices from ``validation.heston_theory.heston_price``.

Statistical tolerances (plan section 2.5): fixed seeds; ``|estimate - theory|
<= 4 SE`` (false alarm ~6e-5 per assertion under asymptotic normality). The
SE of a mean is the sample std over ``sqrt(N)``; the SE of a sample variance
is ``sqrt((m4 - s^4)/N)`` with the sample fourth central moment (all moments
of ``v_t`` and of ``ln S_T`` are finite for the parameters used). Where the
scheme has discretization error the tolerance adds a **declared bias**:

* QE-M option prices: 0.15 for ``dt = 1/4`` and 0.10 for ``dt <= 1/8``
  (researcher's opinion, measured biases <= 2.4 SE at N = 200k; the largest
  was case III K = 140 at dt = 1/4: -0.13 +/- 0.054).
* Full-truncation moments of ``v``: ``|Euler recursion - CIR|`` computed
  exactly below (the Euler moment bias), plus 4 SE.
* ``xi -> 0`` limit: ``|Var ln S_T - theta T| <= xi theta T^2`` to first order
  in ``xi`` (``Cov(Int v, Int sqrt(v) dW1) = rho xi theta (T - (1 -
  e^{-kT})/k)/k + O(xi^2) <= xi theta T^2/2``), and ``xi sqrt(theta) T`` on
  the mean; both are far below the SE here.

Option prices use a control variate on ``S_T`` (its mean ``F = S0 e^{(r-q)T}``
is exact for QE-M and full truncation): ``y - c (S_T - F)`` with the sample
optimal ``c``; the SE is the std of the controlled values over ``sqrt(N)``
(the estimation of ``c`` adds an O(1/N) bias, negligible here).

The Andersen (2008) cases violate Feller (``2 kappa theta < xi^2``). Their
reference prices come from ``heston_price``, itself checked at 1e-7 against
an independent ``quad`` Gil-Pelaez integral.

Memory (``MAX_WORKING_BYTES`` = 256 MiB): the largest matrix pair of the fast
tests is ``2 * 8 * 101 * 50_000`` B ~ 81 MB (full-truncation moments). Prices
use ``generate_terminal`` (state + ~32 vectors of a 100k batch ~ 26 MB plus
the 8 N B output). Slow tests (``@pytest.mark.slow``; they run by default,
deselect with ``-m "not slow"``) use N = 1e6 in batches of 100k (output 8 MB).
"""

from __future__ import annotations

import json
import math
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import simulation.heston as heston_module
from simulation.heston import QE_XI_MIN, HestonSimulator
from validation.heston_theory import XI_MIN, feller_condition, heston_price

K_SE = 4.0
KS_MIN_P = 1e-3
S0 = 100.0
STRIKES = (70.0, 100.0, 140.0)
# Andersen (2008) cases: (kappa, theta = v0, xi, rho, T); S0 = 100, r = q = 0.
CASES: dict[str, tuple[float, float, float, float, float]] = {
    "I": (0.5, 0.04, 1.0, -0.9, 10.0),
    "II": (0.3, 0.04, 0.9, -0.5, 15.0),
    "III": (1.0, 0.09, 1.0, -0.3, 5.0),
}
QE_BIAS = {0.25: 0.15, 0.125: 0.10, 1.0 / 32.0: 0.10}
CASE_I = dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
SCHEMES = ("andersen_qe", "full_truncation")


def _sim(horizon: float, steps: int, n: int, seed: int | None = 1, *,
         mu: float = 0.0, **kw: Any) -> HestonSimulator:
    params: dict[str, Any] = dict(CASE_I)
    params.update(kw)
    return HestonSimulator(S0, horizon, steps, n, seed, mu=mu, **params)


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1)) / math.sqrt(x.size)


def _var_se(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    c = x - float(np.mean(x))
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c ** 4))
    return m2 * n / (n - 1), math.sqrt(max(m4 - m2 * m2, 0.0) / n)


def _cir_mean(v0: float, kappa: float, theta: float, t: float) -> float:
    return theta + (v0 - theta) * math.exp(-kappa * t)


def _cir_var(v0: float, kappa: float, theta: float, xi: float, t: float) -> float:
    e = math.exp(-kappa * t)
    return (v0 * xi * xi * e * (1 - e) / kappa
            + theta * xi * xi * (1 - e) ** 2 / (2 * kappa))


def _euler_moments(v0: float, kappa: float, theta: float, xi: float, dt: float,
                   k: int) -> tuple[float, float]:
    m, var = v0, 0.0
    for _ in range(k):
        var = (1 - kappa * dt) ** 2 * var + xi * xi * dt * m
        m = m + kappa * (theta - m) * dt
    return m, var


def _cv_price(st: np.ndarray, K: float, fwd: float, disc: float,
              kind: str) -> tuple[float, float]:
    y = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
    c = float(np.cov(y, st)[0, 1] / np.var(st, ddof=1))
    z = y - c * (st - fwd)
    est, se = _mean_se(z)
    return disc * est, disc * se


# ---------------------------------------------------------------------------
# Measurement helpers (also used to report the measured values)
# ---------------------------------------------------------------------------
def measure_case_prices(case: str, dt: float, n: int, seed: int,
                        scheme: str = "andersen_qe",
                        batch_size: int | None = None) -> list[dict[str, Any]]:
    kappa, theta, xi, rho, T = CASES[case]
    steps = int(round(T / dt))
    sim = HestonSimulator(S0, T, steps, n, seed, mu=0.0, v0=theta, kappa=kappa,
                          theta=theta, xi=xi, rho=rho, scheme=scheme)
    st = sim.generate_terminal(batch_size=batch_size)
    out = []
    for K in STRIKES:
        ref = heston_price(S0, K, T, 0.0, 0.0, theta, kappa, theta, xi, rho, "call")
        est, se = _cv_price(st, K, S0, 1.0, "call")
        out.append(dict(case=case, K=K, dt=dt, n=n, seed=seed, est=est, ref=ref,
                        se=se, z=(est - ref) / se, err=est - ref))
    return out


def measure_rates_prices(dt: float, n: int, seed: int) -> list[dict[str, Any]]:
    kappa, theta, xi, rho, T = CASES["III"]
    r, q, v0 = 0.03, 0.01, 0.06
    sim = HestonSimulator(S0, T, int(round(T / dt)), n, seed, mu=r - q, v0=v0,
                          kappa=kappa, theta=theta, xi=xi, rho=rho)
    st = sim.generate_terminal()
    fwd, disc = S0 * math.exp((r - q) * T), math.exp(-r * T)
    out = []
    for K, kind in ((80.0, "put"), (100.0, "call"), (130.0, "call")):
        ref = heston_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, kind)  # type: ignore[arg-type]
        est, se = _cv_price(st, K, fwd, disc, kind)
        out.append(dict(K=K, kind=kind, dt=dt, n=n, seed=seed, est=est, ref=ref,
                        se=se, z=(est - ref) / se))
    return out


def measure_martingale(scheme: str, params: dict[str, float], mu: float, T: float,
                       steps: int, n: int, seed: int) -> dict[str, float]:
    sim = _sim(T, steps, n, seed, mu=mu, scheme=scheme, **params)
    st = sim.generate_terminal()
    est, se = _mean_se(st)
    ref = S0 * math.exp(mu * T)
    return dict(est=est, ref=ref, se=se, z=(est - ref) / se)


def measure_gbm_limit(scheme: str, xi: float, n: int, seed: int,
                      steps: int = 50) -> dict[str, float]:
    theta, mu, T = 0.04, 0.05, 1.0
    sim = _sim(T, steps, n, seed, mu=mu, v0=theta, kappa=1.0, theta=theta, xi=xi,
               rho=-0.5, scheme=scheme)
    x = np.log(sim.generate_terminal() / S0)
    mean_ref, var_ref = (mu - theta / 2) * T, theta * T
    m, se_m = _mean_se(x)
    v, se_v = _var_se(x)
    ks = stats.kstest(x, stats.norm(mean_ref, math.sqrt(var_ref)).cdf)
    return dict(mean=m, mean_ref=mean_ref, se_mean=se_m, z_mean=(m - mean_ref) / se_m,
                var=v, var_ref=var_ref, se_var=se_v, z_var=(v - var_ref) / se_v,
                slack_mean=xi * math.sqrt(theta) * T, slack_var=xi * theta * T * T,
                ks_d=float(ks.statistic), ks_p=float(ks.pvalue))


# ---------------------------------------------------------------------------
# 1. Shape, contract and metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_shape_first_row_and_variance_row(scheme: str) -> None:
    sim = _sim(1.0, 12, 500, 3, scheme=scheme)
    paths = sim.generate_paths()
    assert paths.shape == (13, 500) and paths.dtype == np.float64
    assert np.all(paths[0] == S0)
    paths2, var = _sim(1.0, 12, 500, 3, scheme=scheme).generate_paths_and_variance()
    np.testing.assert_array_equal(paths, paths2)
    assert var.shape == (13, 500) and var.dtype == np.float64
    assert np.all(var[0] == CASE_I["v0"])
    assert np.all(var >= 0.0) and np.all(np.isfinite(var))
    assert np.all(paths > 0.0) and np.all(np.isfinite(paths))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("xi", [1.0, 0.0])
def test_model_info_json(scheme: str, xi: float) -> None:
    sim = _sim(2.0, 8, 10, 4, scheme=scheme, xi=xi)
    info = sim.model_info()
    text = json.dumps(info, allow_nan=False)
    assert json.loads(text) == info
    assert info["model"] == "heston" and info["scheme"] == scheme
    assert info["steps"] == 8 and info["dt_years"] == 0.25 and info["seed"] == 4
    assert info["exact_on_grid"] is (xi == 0.0)
    if xi == 0.0:
        assert info["discretization"] == "exact_deterministic_variance"
    else:
        assert info["discretization"] == scheme
    assert info["feller"] == feller_condition(CASE_I["kappa"], CASE_I["theta"], xi)
    assert info["feller"] == sim.feller
    if scheme == "andersen_qe":
        assert info["martingale_correction"] is True and info["variance_output"] == "v"
    else:
        assert info["martingale_correction"] == "not_applicable"
        assert info["variance_output"] == "max(v_tilde,0)"
    assert "PCG64" in info["rng"]
    assert _sim(1.0, 2, 2, scheme="andersen_qe",
                martingale_correction=False).model_info()["martingale_correction"] is False


def test_feller_reported_not_required() -> None:
    assert _sim(1.0, 4, 10).feller["satisfied"] is False                # case I
    ok = _sim(1.0, 4, 10, kappa=2.0, theta=0.04, xi=0.3).feller
    assert ok["satisfied"] is True and ok["margin"] == pytest.approx(0.07)


def test_calibrate_not_implemented() -> None:
    prices = pd.Series(np.linspace(100, 110, 50),
                       index=pd.bdate_range("2020-01-01", periods=50))
    with pytest.raises(NotImplementedError):
        _sim(1.0, 4, 10).calibrate(prices)


# ---------------------------------------------------------------------------
# 2. Determinism and RNG streams
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_determinism_and_generator_advance(scheme: str) -> None:
    a = _sim(1.0, 10, 300, 77, scheme=scheme)
    b = _sim(1.0, 10, 300, 77, scheme=scheme)
    first = a.generate_paths()
    np.testing.assert_array_equal(first, b.generate_paths())
    assert not np.array_equal(first, a.generate_paths())   # generators advance
    t1 = _sim(1.0, 10, 300, 77, scheme=scheme).generate_terminal(batch_size=64)
    t2 = _sim(1.0, 10, 300, 77, scheme=scheme).generate_terminal(batch_size=64)
    np.testing.assert_array_equal(t1, t2)
    # Single batch covering all paths draws in the same order as generate_paths.
    t3 = _sim(1.0, 10, 300, 77, scheme=scheme).generate_terminal(batch_size=300)
    np.testing.assert_allclose(t3, first[-1], rtol=1e-12, atol=0)
    assert not np.array_equal(_sim(1.0, 10, 300, 78, scheme=scheme).generate_paths(),
                              first)


def test_separate_streams() -> None:
    seed = 2025

    def fresh() -> list[np.random.Generator]:
        return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]

    ft = _sim(1.0, 6, 50, seed, scheme="full_truncation")
    ft.generate_paths()
    # Full truncation never touches the uniform stream.
    assert ft.uniform_rng.random() == fresh()[1].random()
    qe = _sim(1.0, 6, 50, seed, scheme="andersen_qe")
    qe.generate_paths()
    # QE draws one uniform per path per step.
    ref_u = fresh()[1]
    ref_u.random(6 * 50)
    assert qe.uniform_rng.random() == ref_u.random()
    # xi = 0 draws only the price normals.
    det = _sim(1.0, 6, 50, seed, xi=0.0)
    det.generate_paths()
    gens = fresh()
    assert det.variance_rng.standard_normal() == gens[0].standard_normal()
    assert det.uniform_rng.random() == gens[1].random()
    # The base-class generator is unused.
    base = _sim(1.0, 6, 50, seed)
    base.generate_paths()
    assert base.rng.random() == np.random.default_rng(seed).random()


# ---------------------------------------------------------------------------
# 3. Moments of the variance process
# ---------------------------------------------------------------------------
QE_MOMENT_CASES = [
    (dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9), 2.0, 11),   # Feller violated
    (dict(v0=0.09, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5), 1.0, 12),   # v0 != theta
]


@pytest.mark.parametrize("params, T, seed", QE_MOMENT_CASES)
def test_qe_variance_moments_exact_on_grid(params: dict[str, float], T: float,
                                           seed: int) -> None:
    # QE matches the conditional CIR mean and variance, so E[v_t] and Var[v_t]
    # are exact on the grid (no declared bias). N = 50_000, 20 steps.
    steps, n = 20, 50_000
    _, var = _sim(T, steps, n, seed, **params).generate_paths_and_variance()
    p = params
    for k in (5, 10, 20):
        t = k * T / steps
        m, se_m = _mean_se(var[k])
        v, se_v = _var_se(var[k])
        assert abs(m - _cir_mean(p["v0"], p["kappa"], p["theta"], t)) <= K_SE * se_m
        assert abs(v - _cir_var(p["v0"], p["kappa"], p["theta"], p["xi"], t)) <= K_SE * se_v


FT_PARAMS = dict(v0=0.09, kappa=3.0, theta=0.06, xi=0.2, rho=-0.5)  # Feller holds


@pytest.mark.parametrize("steps", [20, 100])
def test_full_truncation_variance_moments(steps: int) -> None:
    # The Feller ratio 2 kappa theta / xi^2 = 9 keeps v~ positive (truncated
    # fraction asserted < 1e-3), so FT follows the Euler moment recursion,
    # which is the exact reference for the scheme at 4 SE. Against CIR the
    # declared bias is |Euler - CIR| (exact O(dt) moment bias of Euler): at
    # dt = 1/20 it is ~8 SE on the mean and ~18 SE on the variance (so the
    # bias is real); at dt = 1/100 it is still up to ~3.4 SE on the variance
    # (t = 1/4), hence the declared |Euler - CIR| term.
    n, T = 50_000, 1.0
    p = FT_PARAMS
    _, var = _sim(T, steps, n, 13, scheme="full_truncation",
                  **p).generate_paths_and_variance()
    assert float(np.mean(var[1:] == 0.0)) < 1e-3
    dt = T / steps
    for k in (steps // 4, steps // 2, steps):
        t = k * dt
        m, se_m = _mean_se(var[k])
        v, se_v = _var_se(var[k])
        em, ev = _euler_moments(p["v0"], p["kappa"], p["theta"], p["xi"], dt, k)
        cm = _cir_mean(p["v0"], p["kappa"], p["theta"], t)
        cv = _cir_var(p["v0"], p["kappa"], p["theta"], p["xi"], t)
        assert abs(m - em) <= K_SE * se_m
        assert abs(v - ev) <= K_SE * se_v
        assert abs(m - cm) <= K_SE * se_m + abs(em - cm)
        assert abs(v - cv) <= K_SE * se_v + abs(ev - cv)


# ---------------------------------------------------------------------------
# 4. xi -> 0: GBM limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("xi", [1e-3, 1e-4, 0.0])
def test_gbm_limit(scheme: str, xi: float) -> None:
    # N = 50_000, 50 steps, seed 2024 (fixed). KS p >= 1e-3 (plan 2.5).
    r = measure_gbm_limit(scheme, xi, 50_000, 2024)
    assert abs(r["mean"] - r["mean_ref"]) <= K_SE * r["se_mean"] + r["slack_mean"]
    assert abs(r["var"] - r["var_ref"]) <= K_SE * r["se_var"] + r["slack_var"]
    assert r["ks_p"] > KS_MIN_P, r


def test_zero_xi_is_exact_deterministic_variance() -> None:
    # xi = 0, v0 != theta: v follows the ODE exactly and ln S_T is Gaussian
    # with variance Int v dt = theta T + (v0 - theta)(1 - e^{-kT})/k.
    v0, kappa, theta, mu, T = 0.09, 2.0, 0.04, 0.05, 1.0
    for scheme in SCHEMES:
        sim = _sim(T, 10, 50_000, 31, mu=mu, v0=v0, kappa=kappa, theta=theta, xi=0.0,
                   rho=0.3, scheme=scheme)
        paths, var = sim.generate_paths_and_variance()
        grid = np.linspace(0.0, T, 11)
        np.testing.assert_allclose(var, np.broadcast_to(
            (theta + (v0 - theta) * np.exp(-kappa * grid))[:, None], var.shape),
            rtol=1e-12)
        iv = theta * T + (v0 - theta) * (1 - math.exp(-kappa * T)) / kappa
        x = np.log(paths[-1] / S0)
        ks = stats.kstest(x, stats.norm(mu * T - iv / 2, math.sqrt(iv)).cdf)
        assert ks.pvalue > KS_MIN_P


# ---------------------------------------------------------------------------
# 5. Martingale: E[S_T] = S0 e^{mu T}
# ---------------------------------------------------------------------------
MARTINGALE_CASES = [
    ("andersen_qe", dict(CASE_I), 101),
    ("full_truncation", dict(CASE_I), 102),
    ("andersen_qe", dict(v0=0.09, kappa=1.0, theta=0.09, xi=1.0, rho=0.5), 103),
    ("full_truncation", dict(v0=0.09, kappa=1.0, theta=0.09, xi=1.0, rho=0.5), 104),
]


@pytest.mark.parametrize("scheme, params, seed", MARTINGALE_CASES)
def test_martingale(scheme: str, params: dict[str, float], seed: int) -> None:
    # Exact for any dt (QE-M by construction, log-Euler step by step):
    # dt = 1/4 (coarse on purpose), T = 2, mu = 0.07, N = 200_000.
    r = measure_martingale(scheme, params, 0.07, 2.0, 8, 200_000, seed)
    assert abs(r["z"]) <= K_SE, r


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_perfect_correlation(scheme: str, rho: float) -> None:
    # rho = +-1: K3 = K4 = 0 (QE) and dW_perp drops (FT). With rho = +1 the
    # QE-M condition A < 1/(2a), A < beta holds for these parameters.
    params = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=rho)
    paths, var = _sim(1.0, 12, 2_000, 41, mu=0.03, scheme=scheme,
                      **params).generate_paths_and_variance()
    assert np.all(np.isfinite(paths)) and np.all(paths > 0) and np.all(var >= 0)
    r = measure_martingale(scheme, params, 0.03, 1.0, 12, 200_000, 42)
    assert abs(r["z"]) <= K_SE, r


def test_martingale_correction_failure_raises() -> None:
    # rho > 0 with large xi and dt: A = K2 + K4/2 >= 1/(2a) on every path.
    sim = HestonSimulator(S0, 9.0, 3, 1000, 1, mu=0.0, v0=0.45, kappa=4.5, theta=0.45,
                          xi=2.75, rho=0.98)
    with pytest.raises(ValueError, match="martingale correction"):
        sim.generate_paths()
    with pytest.raises(ValueError, match="martingale correction"):
        HestonSimulator(S0, 9.0, 3, 1000, 1, mu=0.0, v0=0.45, kappa=4.5, theta=0.45,
                        xi=2.75, rho=0.98).generate_terminal()
    # Without the correction the same parameters simulate.
    paths = HestonSimulator(S0, 9.0, 3, 1000, 1, mu=0.0, v0=0.45, kappa=4.5,
                            theta=0.45, xi=2.75, rho=0.98,
                            martingale_correction=False).generate_paths()
    assert np.all(np.isfinite(paths))


# ---------------------------------------------------------------------------
# 6. Feller violated: no nan/inf, no negative variance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("params", [
    dict(CASE_I),
    dict(v0=0.02, kappa=0.1, theta=0.02, xi=2.0, rho=-0.7),   # 2kθ/ξ² = 0.001
    dict(v0=0.0, kappa=0.3, theta=0.04, xi=1.5, rho=-0.99),
])
def test_feller_violated_is_finite_and_nonnegative(scheme: str,
                                                    params: dict[str, float]) -> None:
    paths, var = _sim(5.0, 20, 20_000, 51, mu=0.02, scheme=scheme,
                      **params).generate_paths_and_variance()
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
    assert np.all(np.isfinite(var)) and float(np.min(var)) >= 0.0
    assert float(np.mean(var[1:] == 0.0)) > 0.0 or scheme == "andersen_qe"


# ---------------------------------------------------------------------------
# 7. European prices under Q vs the semi-analytic benchmark
# ---------------------------------------------------------------------------
_PRICE_SEEDS = {("I", 0.25): 9101, ("I", 0.125): 9102, ("II", 0.25): 9103,
                ("II", 0.125): 9104, ("III", 0.25): 9105, ("III", 0.125): 9106}


@pytest.mark.parametrize("case, dt", list(_PRICE_SEEDS))
def test_qe_m_prices_andersen_cases(case: str, dt: float) -> None:
    # N = 100_000, control variate on S_T; |err| <= 4 SE + declared bias.
    for r in measure_case_prices(case, dt, 100_000, _PRICE_SEEDS[(case, dt)]):
        assert abs(r["err"]) <= K_SE * r["se"] + QE_BIAS[dt], r


@pytest.mark.parametrize("dt, seed", [(0.25, 9201), (0.125, 9202)])
def test_qe_m_prices_with_rates_and_dividends(dt: float, seed: int) -> None:
    # Case III dynamics with r = 3%, q = 1%, v0 = 0.06 != theta; mu = r - q,
    # discounting by e^{-rT}; a put and two calls.
    for r in measure_rates_prices(dt, 100_000, seed):
        assert abs(r["est"] - r["ref"]) <= K_SE * r["se"] + QE_BIAS[dt], r


def test_full_truncation_bias_shrinks() -> None:
    # Case I, K = 100: FT bias ~ +2.0 at dt = 1/4 and ~ +1.0 at dt = 1/8
    # (opinion: +1.99, +1.03). Require a large, positive bias at 1/4 and
    # bias(1/8) < 0.75 bias(1/4) with 4-SE margins on both sides.
    b4 = measure_case_prices("I", 0.25, 100_000, 9301, scheme="full_truncation")[1]
    b8 = measure_case_prices("I", 0.125, 100_000, 9302, scheme="full_truncation")[1]
    assert b4["err"] - K_SE * b4["se"] > 1.0
    assert b8["err"] + K_SE * b8["se"] < 0.75 * (b4["err"] - K_SE * b4["se"])
    assert b8["err"] - K_SE * b8["se"] > 0.0


# ---------------------------------------------------------------------------
# 8. Degenerate cases and memory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_degenerate_sizes(scheme: str) -> None:
    one = _sim(1.0, 1, 1, 5, scheme=scheme).generate_paths()
    assert one.shape == (2, 1) and one[0, 0] == S0 and np.isfinite(one).all()
    step1 = _sim(0.5, 1, 1000, 5, scheme=scheme).generate_paths_and_variance()
    assert step1[0].shape == (2, 1000) and np.all(step1[1] >= 0)
    term = _sim(1.0, 3, 1, 5, scheme=scheme).generate_terminal()
    assert term.shape == (1,) and term[0] > 0
    # Tiny horizon: S_T ~ S0 within a few sqrt(v0 T).
    tiny = _sim(1e-8, 4, 1000, 5, scheme=scheme).generate_paths()
    assert np.all(np.abs(np.log(tiny[-1] / S0)) < 10 * math.sqrt(0.04 * 1e-8))
    # One trading day.
    day = _sim(1 / 252, 1, 10_000, 5, scheme=scheme).generate_terminal()
    assert np.all(np.isfinite(day)) and abs(float(np.mean(day)) / S0 - 1) < 0.01


def test_memory_limit_raises_before_allocating() -> None:
    sim = _sim(1.0, 20_000, 2_000, 1)        # 8 * 20001 * 2000 B ~ 320 MB
    assert sim.required_bytes() > sim.MAX_WORKING_BYTES
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="bytes"):
            sim.generate_paths()
        with pytest.raises(ValueError, match="bytes"):
            _sim(1.0, 10_000, 2_000, 1).generate_paths_and_variance()  # ~320 MB pair
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    # Terminal output alone above the limit.
    with pytest.raises(ValueError, match="bytes"):
        _sim(1.0, 1, 40_000_000, 1).generate_terminal()
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            _sim(1.0, 2, 10, 1).generate_terminal(batch_size=bad)  # type: ignore[arg-type]
    # generate_paths_and_variance counts the variance matrix.
    s = _sim(1.0, 10, 100, 1)
    assert s.required_bytes(True) - s.required_bytes(False) == 8 * 11 * 100


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------
GOOD: dict[str, Any] = dict(mu=0.05, **CASE_I)
BAD = [
    ("mu", True), ("mu", math.nan), ("mu", math.inf), ("mu", "0.05"),
    ("v0", -0.01), ("v0", np.bool_(False)), ("v0", math.nan),
    ("kappa", 0.0), ("kappa", -1.0), ("theta", 0.0), ("theta", -0.04),
    ("xi", -0.1), ("xi", math.inf), ("rho", 1.0000001), ("rho", -2.0), ("rho", True),
    ("scheme", "euler"), ("scheme", None), ("scheme", "QE"),
    ("martingale_correction", 1), ("martingale_correction", "yes"),
    ("martingale_correction", None),
]


@pytest.mark.parametrize("name, value", BAD)
def test_input_validation(name: str, value: Any) -> None:
    kw = dict(GOOD)
    kw[name] = value
    with pytest.raises(ValueError):
        HestonSimulator(S0, 1.0, 4, 10, 1, **kw)


@pytest.mark.parametrize("args", [(0.0, 1.0, 4, 10, 1), (S0, 0.0, 4, 10, 1),
                                  (S0, 1.0, 0, 10, 1), (S0, 1.0, 4, 0, 1),
                                  (S0, 1.0, 4, 10, -1), (True, 1.0, 4, 10, 1)])
def test_base_validation(args: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        HestonSimulator(*args, **GOOD)


def test_keyword_only_parameters() -> None:
    with pytest.raises(TypeError):
        cls: Any = HestonSimulator   # positional model parameters are rejected
        cls(S0, 1.0, 4, 10, 1, 0.05, 0.04, 0.5, 0.04, 1.0, -0.9)
    assert HestonSimulator(S0, 1.0, 4, 10, None, **GOOD).seed is None
    assert _sim(1.0, 4, 10, v0=0.0).generate_paths().shape == (5, 10)   # v0 = 0 valid
    assert _sim(1.0, 4, 10, martingale_correction=np.bool_(True)).martingale_correction is True


# ---------------------------------------------------------------------------
# 9b. Regressions of the independent review of d64c716 (P2-1, P3-1, P3-2, S1-S3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("steps, n", [(1, 200_000), (500, 4_000)])
def test_memory_estimate_is_an_upper_bound(scheme: str, steps: int, n: int) -> None:
    # P2-1: the estimate is 8 (steps + 1) n per matrix plus _WORK_VECTORS
    # vectors of n float64; no (steps + 1, n) bool mask may exist (it added
    # 12.5% of the output for long paths and made peak/estimate 1.12-1.23).
    # (1, 200k) stresses the QE temporaries, (500, 4k) the output-sized mask.
    # Measured margin (NumPy 2.x, CPython 3.13): >= 5.8 vectors for QE, ~22
    # for full truncation, so the peak must stay below the estimate itself.
    assert heston_module._WORK_VECTORS >= 26   # measured QE peak ~25.1 vectors
    for return_variance in (False, True):
        sim = _sim(1.0, steps, n, 3, scheme=scheme)
        assert sim.required_bytes(return_variance) == (
            8 * (steps + 1) * n * (1 + int(return_variance))
            + 8 * heston_module._WORK_VECTORS * n)
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            if return_variance:
                sim.generate_paths_and_variance()
            else:
                sim.generate_paths()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not was_tracing:
                tracemalloc.stop()
        assert peak <= sim.required_bytes(return_variance), (peak, return_variance)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_terminal_memory_estimate_is_an_upper_bound(scheme: str) -> None:
    # P2-1 for generate_terminal: output 8 N plus _WORK_VECTORS vectors of one
    # batch (measured QE ~25.1 vectors per batch).
    n, batch = 200_000, 50_000
    sim = _sim(1.0, 5, n, 3, scheme=scheme)
    budget = 8 * n + 8 * heston_module._WORK_VECTORS * batch
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim.generate_terminal(batch_size=batch)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert peak <= budget, (peak, budget)


@pytest.mark.parametrize("xi", [1e-14, 1e-8, 9.99e-5])
def test_qe_rejects_tiny_positive_xi(xi: float) -> None:
    # P3-1: K1, K2 contain -/+ rho/xi; with xi = 1e-14, rho = 0.9 the QE step
    # lost the martingale silently (E[S_T] - S0 = +0.33, 5 SE).
    for correction in (True, False):
        with pytest.raises(ValueError, match="xi"):
            _sim(1.0, 12, 10, xi=xi, rho=0.9, martingale_correction=correction)
    # Full truncation and the exact xi = 0 branch remain available.
    ft = _sim(1.0, 12, 10, xi=xi, rho=0.9, scheme="full_truncation").generate_paths()
    assert np.all(np.isfinite(ft))
    assert np.all(np.isfinite(_sim(1.0, 12, 10, xi=0.0, rho=0.9).generate_paths()))


def test_qe_xi_threshold_matches_theory() -> None:
    # The simulator replicates the pricer threshold without importing it.
    assert QE_XI_MIN == XI_MIN == 1e-4
    assert np.all(np.isfinite(_sim(1.0, 4, 10, xi=QE_XI_MIN).generate_paths()))


def test_qe_tiny_kappa_is_finite_with_driftless_cir_moments() -> None:
    # P3-2: with kappa = 1e-20, E = e^{-kappa dt} == 1.0 and the old
    # m = theta + (v - theta) E rounded to 0 for v = 0 -> nan. With
    # m = theta (1 - E) + v E the limit kappa -> 0 is the driftless CIR:
    # E[v_T] = v0, Var[v_T] = v0 xi^2 T (theta is irrelevant), and QE matches
    # both exactly on the grid; QE-M keeps E[S_T] = S0 e^{mu T}.
    # N = 200_000, 4 steps, seed 61; 4 SE (SE of the variance from m4).
    v0, xi, T, mu = 0.04, 0.5, 1.0, 0.03
    sim = _sim(T, 4, 200_000, 61, mu=mu, v0=v0, kappa=1e-20, theta=0.09, xi=xi,
               rho=-0.5)
    paths, var = sim.generate_paths_and_variance()
    assert np.all(np.isfinite(paths)) and np.all(np.isfinite(var))
    assert float(np.min(var)) >= 0.0 and float(np.mean(var[-1] == 0.0)) > 0.0
    mean, se = _mean_se(var[-1])
    assert abs(mean - v0) <= K_SE * se
    variance, se_v = _var_se(var[-1])
    assert abs(variance - v0 * xi * xi * T) <= K_SE * se_v
    s_mean, s_se = _mean_se(paths[-1])
    assert abs(s_mean - S0 * math.exp(mu * T)) <= K_SE * s_se


class _FixedDraws:
    """Stand-in for a ``np.random.Generator`` returning fixed draws."""

    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def standard_normal(self, n: int) -> np.ndarray:
        assert n == self.values.size
        return self.values.copy()

    def random(self, n: int) -> np.ndarray:
        assert n == self.values.size
        return self.values.copy()


_ZV = [-1.3, 0.2, 2.1]
_U = [0.2, 0.7, 0.95]
_Z = [0.5, -1.1, 1.7]


def _qe_one_step_reference(v: float, dt: float, mu: float, kappa: float,
                           theta: float, xi: float, rho: float,
                           correction: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Andersen (2008) QE / QE-M step written independently of the simulator.

    Returns ``(ln(S'/S), v', psi)`` for the draws ``_ZV``, ``_U``, ``_Z``.
    """
    e = math.exp(-kappa * dt)
    m = theta + (v - theta) * e
    s2 = v * xi * xi * e * (1 - e) / kappa + theta * xi * xi * (1 - e) ** 2 / (2 * kappa)
    psi = s2 / (m * m)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = k4 = 0.5 * dt * (1 - rho * rho)
    big_a = k2 + k4 / 2
    v_new, x_new = [], []
    for zv, u, z in zip(_ZV, _U, _Z):
        if psi <= 1.5:
            b2 = 2 / psi - 1 + math.sqrt(2 / psi) * math.sqrt(2 / psi - 1)
            a = m / (1 + b2)
            vn = a * (math.sqrt(b2) + zv) ** 2
            k0_star = (-big_a * b2 * a / (1 - 2 * big_a * a)
                       + 0.5 * math.log(1 - 2 * big_a * a) - (k1 + k3 / 2) * v)
        else:
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            vn = 0.0 if u <= p else math.log((1 - p) / (1 - u)) / beta
            k0_star = -math.log(p + beta * (1 - p) / (beta - big_a)) - (k1 + k3 / 2) * v
        kk = k0_star if correction else k0
        v_new.append(vn)
        x_new.append(mu * dt + kk + k1 * v + k2 * vn + math.sqrt(k3 * v + k4 * vn) * z)
    return np.asarray(x_new), np.asarray(v_new), psi


# v0 chosen (kappa = 0.5, theta = 0.04, xi = 0.5, dt = 1/4) so that psi falls
# in (1, 1.5] (quadratic, exponential if psi_c were 1.0), (0.5, 1] (quadratic,
# exponential if psi_c were 0.5), (0, 0.5] and > 1.5 (exponential, with U on
# both sides of p).
_PSI_CASES = [(0.04, 1.0, 1.5), (0.06, 0.5, 1.0), (0.3, 0.0, 0.5), (0.01, 1.5, math.inf)]


@pytest.mark.parametrize("v0, psi_lo, psi_hi", _PSI_CASES)
@pytest.mark.parametrize("rho", [-0.9, 0.5])
@pytest.mark.parametrize("correction", [True, False])
def test_qe_one_step_matches_written_formulas(v0: float, psi_lo: float, psi_hi: float,
                                              rho: float, correction: bool,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # S1-S3: deterministic check of one QE(-M) step (gamma1 = gamma2 = 1/2,
    # psi_c = 1.5) through the public generators. Kills sign errors in K1/K2,
    # an omitted or altered K0*, and a wrong psi_c (the branch changes).
    dt, mu, kappa, theta, xi = 0.25, 0.03, 0.5, 0.04, 0.5
    sim = HestonSimulator(S0, dt, 1, 3, 1, mu=mu, v0=v0, kappa=kappa, theta=theta,
                          xi=xi, rho=rho, martingale_correction=correction)
    monkeypatch.setattr(sim, "variance_rng", _FixedDraws(_ZV))
    monkeypatch.setattr(sim, "uniform_rng", _FixedDraws(_U))
    monkeypatch.setattr(sim, "price_rng", _FixedDraws(_Z))
    paths, var = sim.generate_paths_and_variance()
    x_ref, v_ref, psi = _qe_one_step_reference(v0, dt, mu, kappa, theta, xi, rho,
                                               correction)
    assert psi_lo < psi <= psi_hi
    if psi > 1.5:
        assert np.any(v_ref == 0.0) and np.any(v_ref > 0.0)   # both sides of p
    np.testing.assert_allclose(var[1], v_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.log(paths[1] / S0), x_ref, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------
# 10. Large samples (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("dt", [0.125, 1.0 / 32.0])
def test_qe_m_prices_large_sample(case: str, dt: float) -> None:
    # N = 1e6 in batches of 100_000; |err| <= 4 SE + 0.10.
    seed = 9400 + 10 * list(CASES).index(case) + (0 if dt == 0.125 else 1)
    for r in measure_case_prices(case, dt, 1_000_000, seed, batch_size=100_000):
        assert abs(r["err"]) <= K_SE * r["se"] + QE_BIAS[dt], r


@pytest.mark.slow
def test_full_truncation_bias_shrinks_large_sample() -> None:
    # Case I, K = 100, N = 1e6: bias(1/32) < 0.5 bias(1/4) with 4-SE margins
    # (opinion: +1.99 at 1/4, +0.29 at 1/32).
    b4 = measure_case_prices("I", 0.25, 1_000_000, 9501, scheme="full_truncation")[1]
    b32 = measure_case_prices("I", 1.0 / 32.0, 1_000_000, 9502,
                              scheme="full_truncation")[1]
    assert b32["err"] + K_SE * b32["se"] < 0.5 * (b4["err"] - K_SE * b4["se"])


@pytest.mark.slow
@pytest.mark.parametrize("scheme", SCHEMES)
def test_gbm_limit_large_sample(scheme: str) -> None:
    r = measure_gbm_limit(scheme, 1e-3, 1_000_000, 5)
    assert abs(r["mean"] - r["mean_ref"]) <= K_SE * r["se_mean"] + r["slack_mean"]
    assert abs(r["var"] - r["var_ref"]) <= K_SE * r["se_var"] + r["slack_var"]
    assert r["ks_p"] > KS_MIN_P, r


@pytest.mark.slow
@pytest.mark.parametrize("scheme, seed", [("andersen_qe", 9601), ("full_truncation", 9602)])
def test_martingale_large_sample(scheme: str, seed: int) -> None:
    r = measure_martingale(scheme, dict(CASE_I), 0.07, 10.0, 40, 1_000_000, seed)
    assert abs(r["z"]) <= K_SE, r
