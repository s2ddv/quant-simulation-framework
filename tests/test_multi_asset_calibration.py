"""Multi-asset calibration (E6.2) and multi-ticker data boundary tests, offline.

Estimators under test (252 trading days)::

    sigma_i = sqrt(252) std(r_i, ddof=1),  mu_i = 252 mean(r_i) + sigma_i^2/2,  R = corr(r)

Recovery tolerances come from the estimation error of i.i.d. normal daily
returns with n observations (4 standard errors, docs/plano-expansao.md Sec. 2.5):

* SE(sigma_hat) ~ sigma / sqrt(2 n)  (annualized; delta method on the sample variance)
* SE(252 mean) = 252 sigma_d / sqrt(n) = sigma sqrt(252 / n); the sigma_hat^2/2 term
  adds sigma SE(sigma_hat), independent of the mean under normality.
* Fisher-z for every correlation: |atanh(r) - atanh(rho)| <= 4 / sqrt(n - 3).

No network: ``yf.download`` is monkeypatched for every data.py test.
"""

import numpy as np
import pandas as pd
import pytest

from calibration import data
from calibration.data import (
    download_prices_multi,
    multi_asset_snapshot,
    prices_from_multi_asset_snapshot,
)
from calibration.estimation import TRADING_DAYS, estimate_gbm_params
from calibration.multi_asset import (
    align_prices,
    estimate_multi_asset_gbm,
    ledoit_wolf_constant_correlation,
)
from simulation.multi_asset import validate_correlation

K_SE = 4.0


def _series(values, dates):
    return pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates))


@pytest.fixture
def ten_days():
    return pd.bdate_range("2025-03-03", periods=10)  # Mon 3 Mar .. Fri 14 Mar 2025


def _holiday_prices(days):
    # A trades every day; B misses d3 and d7; C misses d0 (listed later) and d5.
    a = _series(100 + np.arange(10), days)
    b = _series(50 + np.arange(10), days).drop(days[[3, 7]])
    c = _series(20 + np.arange(10), days).drop(days[[0, 5]])
    return {"A": a, "B": b, "C": c}


# --------------------------------------------------------------------------- alignment


def test_inner_join_counts_by_hand(ten_days):
    frame, report = align_prices(_holiday_prices(ten_days))
    common = ten_days.delete([0, 3, 5, 7])  # 6 dates present in all three
    assert list(frame.index) == list(common)
    assert list(frame.columns) == ["A", "B", "C"]
    assert report["common_dates"] == 6
    # A: 10 - 6 = 4, B: 8 - 6 = 2, C: 8 - 6 = 2.
    assert report["dropped_dates"] == {"A": 4, "B": 2, "C": 2}
    assert report["forward_fill"] is False and "filled_values" not in report
    assert report["first_date"] == common[0].isoformat()
    assert report["last_date"] == common[-1].isoformat()
    # No forward fill: the values are exactly the observed ones.
    assert frame.loc[ten_days[1], "B"] == 51.0 and not frame.isna().any().any()
    assert frame.dtypes.unique().tolist() == [np.float64]


def test_forward_fill_is_explicit_and_counted_by_hand(ten_days):
    frame, report = align_prices(_holiday_prices(ten_days), forward_fill=True)
    # Union = 10 dates; C is still missing at d0 after LOCF -> row d0 dropped.
    assert list(frame.index) == list(ten_days[1:])
    assert report["common_dates"] == 9
    # Filled: B at d3, d7 (2); C at d5 (1); A none.
    assert report["filled_values"] == {"A": 0, "B": 2, "C": 1}
    # Observations falling in the dropped leading row d0: A and B (1 each).
    assert report["dropped_dates"] == {"A": 1, "B": 1, "C": 0}
    assert frame.loc[ten_days[3], "B"] == 52.0  # carried from d2
    assert frame.loc[ten_days[5], "C"] == 24.0  # carried from d4


def test_too_few_common_dates_rejected(ten_days):
    a = _series([1, 2, 3, 4], ten_days[:4])
    b = _series([1, 2, 3, 4], ten_days[2:6])  # overlap = 2 dates
    with pytest.raises(ValueError, match="at least 3"):
        align_prices({"A": a, "B": b})
    frame, _ = align_prices({"A": a, "B": _series([1, 2, 3], ten_days[1:4])})
    assert len(frame) == 3


@pytest.mark.parametrize("bad", [
    {}, [], {"": pd.Series([1.0])}, {1: pd.Series([1.0])},
])
def test_align_rejects_bad_mappings(bad):
    with pytest.raises(ValueError):
        align_prices(bad)


def test_align_rejects_bad_series_and_options(ten_days):
    good = _series(np.arange(1, 11), ten_days)
    for bad in (pd.Series([1.0, 2.0, 3.0]),                       # no DatetimeIndex
                _series([1, 2, -3], ten_days[:3]),                  # nonpositive
                _series([1, np.nan, 3], ten_days[:3]),              # missing
                _series([1, 2, 3], ten_days[[2, 1, 0]]),            # decreasing dates
                [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError, match="B"):
            align_prices({"A": good, "B": bad})
    for flag in (1, "yes", None):
        with pytest.raises(ValueError, match="forward_fill"):
            align_prices({"A": good}, forward_fill=flag)
    aware = good.tz_localize("UTC")
    with pytest.raises(ValueError):
        align_prices({"A": good, "B": aware})


# --------------------------------------------------------------------------- estimation


def _synthetic_prices(n, mu, sigma, corr, seed, start="1940-01-01"):
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    daily_mean = (mu - sigma**2 / 2) / TRADING_DAYS
    daily_sd = sigma / np.sqrt(TRADING_DAYS)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(len(mu)), corr, size=n)
    returns = daily_mean + daily_sd * z
    logs = np.vstack([np.zeros(len(mu)), np.cumsum(returns, axis=0)])
    dates = pd.bdate_range(start, periods=n + 1)
    return {f"T{i}": pd.Series(10.0 * np.exp(logs[:, i]), index=dates) for i in range(len(mu))}


@pytest.mark.parametrize("n, seed", [(20_000, 2026), pytest.param(60_000, 7, marks=pytest.mark.slow)])
def test_parameter_recovery_within_estimation_error(n, seed):
    mu = np.array([0.08, -0.03, 0.15])
    sigma = np.array([0.18, 0.30, 0.45])
    corr = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.1], [-0.3, 0.1, 1.0]])
    result = estimate_multi_asset_gbm(_synthetic_prices(n, mu, sigma, corr, seed))
    assert result["n_returns"] == n and result["trading_days"] == 252
    assert result["tickers"] == ["T0", "T1", "T2"]
    se_sigma = sigma / np.sqrt(2 * n)
    assert np.all(np.abs(result["sigma"] - sigma) <= K_SE * se_sigma)
    se_mu = np.hypot(sigma * np.sqrt(TRADING_DAYS / n), sigma * se_sigma)
    assert np.all(np.abs(result["mu"] - mu) <= K_SE * se_mu)
    est = result["correlation"]
    for i in range(3):
        for j in range(i + 1, 3):
            z = (np.arctanh(est[i, j]) - np.arctanh(corr[i, j])) * np.sqrt(n - 3)
            assert abs(z) <= K_SE, (i, j, est[i, j])
    validate_correlation(est)


def test_estimates_equal_single_asset_estimator_on_common_dates(ten_days):
    prices = _synthetic_prices(400, [0.05, 0.1, 0.0], [0.2, 0.3, 0.25], np.eye(3), 5)
    prices["T1"] = prices["T1"].drop(prices["T1"].index[[10, 50, 51]])
    prices["T2"] = prices["T2"].iloc[7:]
    result = estimate_multi_asset_gbm(prices)
    frame, report = align_prices(prices)
    assert result["alignment"] == report
    assert report["dropped_dates"] == {"T0": 10, "T1": 7, "T2": 3}
    for k, ticker in enumerate(result["tickers"]):
        mu_k, sigma_k = estimate_gbm_params(frame[ticker])
        assert result["mu"][k] == pytest.approx(mu_k, rel=1e-12, abs=1e-15)
        assert result["sigma"][k] == pytest.approx(sigma_k, rel=1e-12)
    # Independent recomputation of the correlation (pandas Pearson).
    returns = np.log(frame).diff().dropna()
    np.testing.assert_allclose(result["correlation"], returns.corr().to_numpy(), atol=1e-12)
    # DataFrame input gives the same numbers.
    from_frame = estimate_multi_asset_gbm(frame)
    np.testing.assert_allclose(from_frame["mu"], result["mu"], rtol=1e-14)
    np.testing.assert_allclose(from_frame["correlation"], result["correlation"], atol=1e-15)
    assert from_frame["alignment"]["dropped_dates"] == {"T0": 0, "T1": 0, "T2": 0}


def test_estimation_with_forward_fill_and_errors(ten_days):
    prices = _synthetic_prices(60, [0.05, 0.1], [0.2, 0.3], np.eye(2), 1)
    prices["T1"] = prices["T1"].drop(prices["T1"].index[[5, 6]])
    filled = estimate_multi_asset_gbm(prices, forward_fill=True)
    assert filled["alignment"]["filled_values"] == {"T0": 0, "T1": 2}
    assert filled["n_returns"] == 60
    assert estimate_multi_asset_gbm(prices)["n_returns"] == 58
    flat = _series(np.full(10, 5.0), ten_days)
    with pytest.raises(ValueError, match="zero volatility"):
        estimate_multi_asset_gbm({"A": _series(np.arange(1, 11), ten_days), "B": flat})
    with pytest.raises(ValueError):
        estimate_multi_asset_gbm(pd.DataFrame({"A": [1.0, 2.0, np.nan, 3.0]},
                                              index=ten_days[:4]))
    dup = pd.DataFrame(np.ones((4, 2)) + np.arange(4)[:, None], index=ten_days[:4],
                       columns=["A", "A"])
    with pytest.raises(ValueError, match="unique"):
        estimate_multi_asset_gbm(dup)
    with pytest.raises(ValueError, match="no columns"):
        estimate_multi_asset_gbm(pd.DataFrame(index=ten_days[:4]))
    single = estimate_multi_asset_gbm({"A": _series(np.exp(np.sin(np.arange(10))), ten_days)})
    assert single["correlation"].shape == (1, 1) and single["correlation"][0, 0] == 1.0


# --------------------------------------------------------------------------- Ledoit-Wolf


def _lw_reference(x):
    """Ledoit & Wolf (2004) constant-correlation shrinkage, moment form.

    Re-derived from the authors' published covCor.m (raw fourth/third moments)
    rather than from the module: pi_ij = mean(x_i^2 x_j^2) - s_ij^2,
    theta_ii,ij = mean(x_i^3 x_j) - s_ii s_ij,
    rho = sum_i pi_ii + rbar sum_{i!=j} sqrt(s_jj/s_ii) theta_ii,ij.
    """
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    rbar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = rbar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)
    y = x**2
    phi_mat = y.T @ y / t - sample**2
    phi = phi_mat.sum()
    theta = (x**3).T @ x / t - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + rbar * np.sum(np.outer(1 / sqrtvar, sqrtvar) * theta)
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    kappa = (phi - rho) / gamma
    corr = sample / np.outer(sqrtvar, sqrtvar)
    return max(0.0, min(1.0, kappa / t)), rbar, corr, phi, rho, gamma


@pytest.mark.parametrize("t, n, seed", [(60, 4, 4), (250, 6, 1), (30, 10, 2), (8, 5, 3),
                                        (1000, 3, 9)])
def test_ledoit_wolf_matches_independent_reference(t, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T + n * np.eye(n)
    scale = rng.uniform(0.005, 0.03, n)  # heterogeneous daily volatilities
    x = rng.multivariate_normal(np.zeros(n), cov, size=t) * scale
    if seed == 9:
        x = rng.standard_t(4, size=(t, n)) * scale  # heavy tails
    result = ledoit_wolf_constant_correlation(x)
    delta, rbar, corr, phi, rho, gamma = _lw_reference(x)
    assert 0.0 <= result["shrinkage"] <= 1.0
    assert result["shrinkage"] == pytest.approx(delta, rel=1e-9, abs=1e-15)
    assert result["target_mean_correlation"] == pytest.approx(rbar, rel=1e-12, abs=1e-15)
    assert result["pi"] == pytest.approx(phi, rel=1e-9)
    assert result["rho"] == pytest.approx(rho, rel=1e-9)
    assert result["gamma"] == pytest.approx(gamma, rel=1e-9)
    shrunk = result["correlation"]
    expected = delta * rbar + (1 - delta) * corr
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(shrunk, expected, atol=1e-12)
    assert np.all(np.diag(shrunk) == 1.0)  # volatilities untouched by construction
    validate_correlation(shrunk)
    assert result["n_observations"] == t
    # DataFrame input is equivalent.
    again = ledoit_wolf_constant_correlation(pd.DataFrame(x))
    assert again["shrinkage"] == pytest.approx(result["shrinkage"], rel=1e-12, abs=1e-15)


def test_ledoit_wolf_shrinkage_is_clipped_at_one_for_tiny_samples():
    hits = []
    for seed in range(40):
        x = np.random.default_rng(seed).standard_normal((3, 6))
        result = ledoit_wolf_constant_correlation(x)
        assert 0.0 <= result["shrinkage"] <= 1.0
        hits.append(result["shrinkage"])
        assert result["shrinkage"] == pytest.approx(_lw_reference(x)[0], abs=1e-12)
    assert max(hits) == 1.0  # the min(., 1) branch is exercised


def test_ledoit_wolf_two_assets_and_one_asset_are_unshrunk():
    x = np.random.default_rng(0).standard_normal((100, 2))
    two = ledoit_wolf_constant_correlation(x)
    assert two["shrinkage"] == 0.0 and "fewer than 3" in two["note"]
    np.testing.assert_allclose(two["correlation"], np.corrcoef(x, rowvar=False), atol=1e-14)
    one = ledoit_wolf_constant_correlation(x[:, :1])
    assert one["shrinkage"] == 0.0 and one["target_mean_correlation"] is None
    assert one["correlation"].shape == (1, 1)


@pytest.mark.parametrize("bad", [
    np.ones((1, 3)), np.ones(5), np.ones((0, 3)), np.ones((5, 0)),
    np.array([[1.0, np.nan], [2.0, 3.0], [1.0, 1.0]]),
    np.array([[True, False], [False, True]]),
    np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),  # zero-variance column
])
def test_ledoit_wolf_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        ledoit_wolf_constant_correlation(bad)


def test_ledoit_wolf_does_not_modify_input():
    x = np.random.default_rng(3).standard_normal((50, 4))
    before = x.copy()
    ledoit_wolf_constant_correlation(x)
    np.testing.assert_array_equal(x, before)


# --------------------------------------------------------------------------- data.py (offline)


def _fake_download(frames, calls):
    def fake(ticker, *args, **kwargs):
        calls.append((ticker, kwargs.get("period")))
        assert kwargs["auto_adjust"] is True and kwargs["progress"] is False
        return frames[ticker]
    return fake


def _provider_frame(ticker, values, dates):
    return pd.DataFrame({("Close", ticker): values, ("Open", ticker): values}, index=dates)


def test_download_prices_multi_keeps_each_series_dates(monkeypatch, ten_days):
    frames = {"AAA": _provider_frame("AAA", np.arange(1.0, 11.0), ten_days),
              "BBB": _provider_frame("BBB", np.arange(5.0, 13.0), ten_days[2:])}
    calls = []
    monkeypatch.setattr("calibration.data.yf.download", _fake_download(frames, calls))
    out = download_prices_multi(["BBB", "AAA"], period="1y")
    assert list(out) == ["BBB", "AAA"]
    assert calls == [("BBB", "1y"), ("AAA", "1y")]
    assert len(out["AAA"]) == 10 and len(out["BBB"]) == 8  # not aligned here
    assert out["BBB"].index[0] == ten_days[2]


@pytest.mark.parametrize("tickers", [[], ["AAA", "AAA"], ["AAA", ""], ["AAA", "  "],
                                     ["AAA", 1], "AAA", None, ("AAA", None)])
def test_download_prices_multi_rejects_bad_tickers_without_network(monkeypatch, tickers):
    def forbidden(*args, **kwargs):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr("calibration.data.yf.download", forbidden)
    with pytest.raises(ValueError, match="ticker"):
        download_prices_multi(tickers)


def test_download_prices_multi_propagates_invalid_response(monkeypatch, ten_days):
    frames = {"AAA": _provider_frame("AAA", np.arange(1.0, 11.0), ten_days),
              "BBB": pd.DataFrame()}
    monkeypatch.setattr("calibration.data.yf.download", _fake_download(frames, []))
    with pytest.raises(ValueError, match="BBB"):
        download_prices_multi(["AAA", "BBB"])


def _snapshot(ten_days):
    prices = {"AAA": _series(np.linspace(10, 20, 10), ten_days),
              "BBB": _series(np.linspace(5, 3, 7), ten_days[3:])}
    return prices, multi_asset_snapshot(prices, period="1y",
                                        retrieved_at="2026-11-27T12:00:00Z")


def test_multi_asset_snapshot_round_trip(ten_days):
    prices, snapshot = _snapshot(ten_days)
    assert snapshot["tickers"] == ["AAA", "BBB"]
    assert snapshot["assets"]["BBB"]["count"] == 7
    assert snapshot["assets"]["AAA"]["sha256"] != snapshot["assets"]["BBB"]["sha256"]
    rebuilt = prices_from_multi_asset_snapshot(snapshot)
    for ticker in prices:
        pd.testing.assert_series_equal(rebuilt[ticker], prices[ticker], check_names=False,
                                       check_freq=False)
    import json
    restored = json.loads(json.dumps(snapshot, allow_nan=False))
    assert prices_from_multi_asset_snapshot(restored).keys() == prices.keys()


def test_multi_asset_snapshot_tampering_rejected(ten_days):
    _, snapshot = _snapshot(ten_days)
    import copy

    tampered = copy.deepcopy(snapshot)
    tampered["assets"]["BBB"]["observations"][2]["close"] += 1e-9
    with pytest.raises(ValueError, match="checksum"):
        prices_from_multi_asset_snapshot(tampered)
    renamed = copy.deepcopy(snapshot)
    renamed["assets"]["BBB"]["ticker"] = "AAA"
    with pytest.raises(ValueError, match="another ticker"):
        prices_from_multi_asset_snapshot(renamed)
    swapped = copy.deepcopy(snapshot)
    swapped["tickers"] = ["AAA", "CCC"]
    with pytest.raises(ValueError, match="match"):
        prices_from_multi_asset_snapshot(swapped)
    meta = copy.deepcopy(snapshot)
    meta["assets"]["AAA"]["count"] = 11
    with pytest.raises(ValueError, match="metadata"):
        prices_from_multi_asset_snapshot(meta)
    for broken in ({}, {"tickers": ["AAA"]}, {"tickers": ["AAA", "AAA"], "assets": {}},
                   {"tickers": "AAA", "assets": {}}, None):
        with pytest.raises(ValueError):
            prices_from_multi_asset_snapshot(broken)


def test_multi_asset_snapshot_rejects_bad_inputs(ten_days):
    good = _series(np.linspace(10, 20, 10), ten_days)
    with pytest.raises(ValueError):
        multi_asset_snapshot([good], period="1y", retrieved_at="2026-11-27T12:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        multi_asset_snapshot({"AAA": good}, period="1y", retrieved_at="2026-11-27")
    with pytest.raises(ValueError, match="ticker"):
        multi_asset_snapshot({"": good}, period="1y", retrieved_at="2026-11-27T12:00:00Z")
    with pytest.raises(ValueError):
        multi_asset_snapshot({"AAA": good.iloc[:2]}, period="1y",
                             retrieved_at="2026-11-27T12:00:00Z")
    assert data.multi_asset_snapshot is multi_asset_snapshot
