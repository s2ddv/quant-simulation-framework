"""Offline tests: GARCH-t fit, no look-ahead, test statistics. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backtest.engine import BacktestConfig, gaussian_forecast, run_backtest
from backtest.garch import fit_garch_t, next_variance, simulate_cumulative
from backtest.metrics import (brier_score, christoffersen_independence, es_residual_tests,
                              kupiec_pof, reliability_table)


def _sim_garch(n, seed, mu=0.0004, omega=2e-6, a=0.08, b=0.90, nu=6.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    h = omega / (1 - a - b); r = np.empty(n)
    for t in range(n):
        e = np.sqrt(h) * z[t]; r[t] = mu + e; h = omega + a * e * e + b * h
    return r


def test_garch_t_parameter_recovery_seeded():
    r = _sim_garch(6000, 1)
    f = fit_garch_t(r)
    assert f["converged"]
    assert abs(f["alpha"] - 0.08) < 0.03 and abs(f["beta"] - 0.90) < 0.04
    assert f["alpha"] + f["beta"] < 1
    assert 4 < f["nu"] < 10


def test_kupiec_known_value_and_edges():
    v = np.zeros(250, bool); v[:5] = True            # 5/250 at 99%: expected 2.5
    k = kupiec_pof(v, 0.99)
    p, x, n = 0.01, 5, 250
    pi = x / n
    ref = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    assert k["lr"] == pytest.approx(ref) and k["p_value"] == pytest.approx(stats.chi2.sf(ref, 1))
    zero = kupiec_pof(np.zeros(100, bool), 0.99)       # 0 violations: LR = -200*ln(0.99)
    assert zero["lr"] == pytest.approx(-200 * np.log(0.99)) and zero["p_value"] == pytest.approx(0.1563, abs=1e-3)
    assert kupiec_pof(np.zeros(100, bool), 0.95)["p_value"] < 0.05


def test_christoffersen_detects_clustering_and_handles_none():
    clustered = np.array(([0] * 40 + [1, 1, 1, 1]) * 5, bool)
    assert christoffersen_independence(clustered)["p_value"] < 0.01
    assert not christoffersen_independence(np.zeros(50, bool))["applicable"]
    assert not christoffersen_independence(np.array([1] + [0] * 20, bool))["applicable"]


def test_brier_and_reliability():
    assert brier_score(np.array([0.0, 1.0]), np.array([0, 1])) == 0
    assert brier_score(np.array([0.5, 0.5]), np.array([0, 1])) == 0.25
    with pytest.raises(ValueError):
        brier_score(np.array([1.2]), np.array([1]))
    t = reliability_table(np.array([0.05, 0.06, 0.5]), np.array([0, 1, 1]), [0, 0.1, 1.0])
    assert [b["n"] for b in t] == [2, 1] and t[0]["observed"] == 0.5


def test_es_residuals_flag_understated_es():
    rng = np.random.default_rng(3)
    loss = rng.standard_t(3, 4000) * 0.01
    var = np.quantile(loss, 0.99) * np.ones(4000)
    true_es = loss[loss >= var[0]].mean()
    ok = es_residual_tests(loss, var, np.full(4000, true_es), 0.99, n_boot=300)
    bad = es_residual_tests(loss, var, np.full(4000, true_es * 0.5), 0.99, n_boot=300)
    assert abs(ok["z2"]) < 0.05 and bad["z2"] > 0.8 and bad["z2_ci"][0] > 0.3


def test_gaussian_forecast_matches_monte_carlo():
    f = gaussian_forecast(0.0003, 0.01, 5)
    x = np.random.default_rng(4).normal(5 * 0.0003, 0.01 * np.sqrt(5), 2_000_000)
    loss = -np.expm1(x)
    assert f["p5"] == pytest.approx(np.mean(loss > 0.05), abs=3e-4)
    v = np.quantile(loss, 0.95)
    assert f["var95"] == pytest.approx(v, rel=0.01)
    assert f["es95"] == pytest.approx(loss[loss >= v].mean(), rel=0.01)


def test_no_lookahead_forecasts_independent_of_future():
    r = pd.Series(_sim_garch(1500, 5), index=pd.bdate_range("2012-01-02", periods=1500))
    cfg = BacktestConfig(min_train=500, refit_every=50, t_draws=2000, garch_draws=1000, seed=3)
    full, meta = run_backtest(r, cfg)
    cut = r.iloc[:1100].copy()
    part, _ = run_backtest(cut, cfg)
    common = part.index
    cols = [c for c in full.columns if c != "realized_loss"]
    pd.testing.assert_frame_equal(full.loc[common, cols], part[cols])
    # corrupt everything after the origin of an interior date: forecasts unchanged
    r2 = r.copy(); r2.iloc[1000:] = 0.5
    alt, _ = run_backtest(r2, cfg)
    d = full.index[full.index < r.index[1000]]
    pd.testing.assert_frame_equal(full.loc[d, cols], alt.loc[d, cols])
    assert meta["n_origins"] == len(full) and meta["n_refits"] >= 1


def test_garch_simulation_unconditional_variance_close():
    r = _sim_garch(4000, 9)
    f = fit_garch_t(r)
    cum = simulate_cumulative(f, f["h_next"], 5, 200_000, np.random.default_rng(1))
    assert np.std(cum) == pytest.approx(np.sqrt(5 * np.var(r)), rel=0.35)
    assert next_variance(r, f) == pytest.approx(f["h_next"], rel=1e-6)
