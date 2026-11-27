"""Multi-asset GBM calibration from aligned historical closes.

For each asset ``i`` with daily log returns ``r_it = ln S_it - ln S_i,t-1``
(on the common dates)::

    sigma_i = sqrt(252) * std(r_i, ddof=1)          (year^-1/2)
    mu_i    = 252 * mean(r_i) + sigma_i**2 / 2      (year^-1, price drift)
    R_ij    = Pearson correlation of (r_i, r_j)

Historical drift is not a risk-neutral drift and historical volatility is not
implied volatility. The sample correlation is the default; Ledoit-Wolf
shrinkage is a separate, explicit function and never applied by default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from calibration.estimation import TRADING_DAYS
from calibration.prices import validate_prices
from simulation.multi_asset import validate_correlation

_MIN_COMMON_DATES = 3


def _ticker_series(prices_by_ticker: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
    if not isinstance(prices_by_ticker, Mapping) or len(prices_by_ticker) == 0:
        raise ValueError("prices_by_ticker must be a non-empty mapping of ticker -> Series")
    validated: dict[str, pd.Series] = {}
    for ticker, series in prices_by_ticker.items():
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("tickers must be non-empty strings")
        try:
            validated[ticker] = validate_prices(series, require_dates=True).rename(ticker)
        except ValueError as exc:
            raise ValueError(f"{ticker}: {exc}") from exc
    return validated


def align_prices(prices_by_ticker: Mapping[str, pd.Series], *,
                 forward_fill: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align closes on common dates; return ``(frame, report)``.

    Each series is checked by ``validate_prices(..., require_dates=True)``
    (positive finite prices, strictly increasing unique ``DatetimeIndex``).

    * ``forward_fill=False`` (default): inner join on the dates present in
      every series. ``report["dropped_dates"][t]`` counts the observations of
      ``t`` that were discarded.
    * ``forward_fill=True`` (explicit opt-in): outer join on the union of
      dates, last-observation-carried-forward, then the leading rows that are
      still incomplete are dropped. ``report["filled_values"][t]`` counts the
      imputed values of ``t``; ``dropped_dates`` counts its observations that
      fell in the dropped leading rows. Carrying prices forward creates zero
      returns and biases volatility and correlation downward.

    Report keys: ``common_dates``, ``dropped_dates``, ``forward_fill``,
    ``filled_values`` (``forward_fill`` only), ``first_date``, ``last_date``
    (ISO strings). Columns follow the mapping order.

    Raises:
        ValueError: Invalid series, mixed time zones, or fewer than 3 aligned
            dates.
    """
    if not isinstance(forward_fill, bool):
        raise ValueError("forward_fill must be a bool")
    series = _ticker_series(prices_by_ticker)
    tickers = list(series)
    try:
        if forward_fill:
            outer = pd.concat(list(series.values()), axis=1, join="outer").sort_index()
            missing = outer.isna()
            filled = outer.ffill()
            keep = filled.notna().all(axis=1).to_numpy()
            frame = filled.loc[keep]
            filled_values = {t: int(missing[t].to_numpy()[keep].sum()) for t in tickers}
            dropped = {t: int(len(series[t]) - (~missing[t].to_numpy()[keep]).sum())
                       for t in tickers}
        else:
            frame = pd.concat(list(series.values()), axis=1, join="inner").sort_index()
            dropped = {t: int(len(series[t]) - len(frame)) for t in tickers}
    except TypeError as exc:
        raise ValueError(f"price indices cannot be aligned (time zones?): {exc}") from exc
    frame.columns = tickers
    if len(frame) < _MIN_COMMON_DATES:
        raise ValueError(
            f"aligned prices have {len(frame)} common dates; require at least {_MIN_COMMON_DATES}"
        )
    report: dict[str, Any] = {
        "common_dates": int(len(frame)),
        "dropped_dates": dropped,
        "forward_fill": forward_fill,
        "first_date": pd.Timestamp(frame.index[0]).isoformat(),
        "last_date": pd.Timestamp(frame.index[-1]).isoformat(),
    }
    if forward_fill:
        report["filled_values"] = filled_values
    return frame.astype(np.float64), report


def _frame_from_dataframe(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prices.shape[1] == 0:
        raise ValueError("prices DataFrame has no columns")
    names = [str(column) for column in prices.columns]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("prices DataFrame columns must be unique non-empty names")
    columns = {}
    for name, column in zip(names, prices.columns):
        try:
            columns[name] = validate_prices(prices[column], min_observations=_MIN_COMMON_DATES,
                                            require_dates=True)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    frame = pd.DataFrame(columns)
    report: dict[str, Any] = {
        "common_dates": int(len(frame)),
        "dropped_dates": {name: 0 for name in names},
        "forward_fill": False,
        "first_date": pd.Timestamp(frame.index[0]).isoformat(),
        "last_date": pd.Timestamp(frame.index[-1]).isoformat(),
        "source": "dataframe (already aligned; no NaN allowed)",
    }
    return frame, report


def estimate_multi_asset_gbm(prices: Mapping[str, pd.Series] | pd.DataFrame, *,
                             forward_fill: bool = False) -> dict[str, Any]:
    """Estimate per-asset GBM drifts, volatilities and the return correlation.

    Args:
        prices: Mapping ticker -> close Series (aligned with ``align_prices``
            and ``forward_fill``) or an already aligned DataFrame (one column
            per ticker, ``DatetimeIndex``, no missing values).
        forward_fill: Passed to ``align_prices`` for a mapping input.

    Returns:
        ``tickers`` (list), ``mu`` and ``sigma`` (float64 arrays, year^-1 and
        year^-1/2), ``correlation`` (validated ``(n, n)`` array),
        ``n_returns``, ``trading_days`` (252) and ``alignment`` (report).
        ``mu`` and ``sigma`` use the formulas of ``estimate_gbm_params``
        (``ddof=1``) with NumPy rather than pandas reductions, so they agree
        with it up to rounding.

    Raises:
        ValueError: Invalid prices, non-finite returns, a zero volatility
            (correlation undefined) or a sample correlation that fails
            ``validate_correlation``.
    """
    if isinstance(prices, pd.DataFrame):
        frame, alignment = _frame_from_dataframe(prices)
    else:
        frame, alignment = align_prices(prices, forward_fill=forward_fill)
    tickers = [str(column) for column in frame.columns]
    values = frame.to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        returns = np.diff(np.log(values), axis=0)
    if not np.isfinite(returns).all():
        raise ValueError("log returns exceed finite precision")
    daily_sigma = np.std(returns, axis=0, ddof=1)
    sigma = np.sqrt(TRADING_DAYS) * daily_sigma
    mu = TRADING_DAYS * np.mean(returns, axis=0) + 0.5 * sigma**2
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("calibrated moments exceed finite precision")
    zero = [t for t, s in zip(tickers, daily_sigma) if s == 0]
    if zero:
        raise ValueError(f"zero volatility for {zero}: correlation is undefined")
    correlation = np.atleast_2d(np.corrcoef(returns, rowvar=False))
    correlation = (correlation + correlation.T) / 2
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, 1.0)
    try:
        correlation = validate_correlation(correlation)
    except ValueError as exc:
        raise ValueError(
            f"sample correlation failed validation ({exc}); consider "
            "simulation.multi_asset.nearest_correlation explicitly"
        ) from exc
    return {
        "tickers": tickers,
        "mu": mu,
        "sigma": sigma,
        "correlation": correlation,
        "n_returns": int(returns.shape[0]),
        "trading_days": TRADING_DAYS,
        "alignment": alignment,
    }


def ledoit_wolf_constant_correlation(log_returns: pd.DataFrame | np.ndarray) -> dict[str, Any]:
    """Ledoit-Wolf shrinkage of the correlation toward constant correlation.

    Ledoit, O. and Wolf, M. (2004), "Honey, I shrunk the sample covariance
    matrix", J. Portfolio Management 30(4):110-119 (appendix formulas; section
    numbers to confirm). With ``T`` observations, demeaned returns
    ``y_it - ybar_i``, ``s_ij = (1/T) sum_t (y_it - ybar_i)(y_jt - ybar_j)``
    (divisor ``T`` as in the article), sample correlations ``r_ij`` and
    ``rbar = mean_{i != j} r_ij``::

        F_ii = s_ii,  F_ij = rbar sqrt(s_ii s_jj)
        pi_ij  = (1/T) sum_t [(y_it - ybar_i)(y_jt - ybar_j) - s_ij]^2,   pi = sum_ij pi_ij
        theta_ii,ij = (1/T) sum_t [(y_it - ybar_i)^2 - s_ii][(y_it - ybar_i)(y_jt - ybar_j) - s_ij]
        rho   = sum_i pi_ii + sum_{i != j} (rbar/2) (sqrt(s_jj/s_ii) theta_ii,ij
                                                    + sqrt(s_ii/s_jj) theta_jj,ij)
        gamma = sum_ij (F_ij - s_ij)^2,   kappa = (pi - rho)/gamma
        delta = max(0, min(kappa / T, 1))

    The shrunk covariance ``delta F + (1 - delta) S`` keeps ``s_ii``, so the
    volatilities are unchanged and the shrunk correlation is
    ``R*_ij = delta rbar + (1 - delta) r_ij`` off the diagonal. With fewer
    than 3 assets (``rbar`` reproduces the off-diagonal for 2 assets, so the
    target equals the sample) or ``gamma == 0``, the sample correlation is
    returned with ``delta = 0`` and an explanatory ``note``.

    Args:
        log_returns: ``(T, N)`` finite real returns, ``T >= 2``, ``N >= 1``,
            no zero-variance column.

    Returns:
        ``correlation`` (validated array), ``shrinkage`` (delta),
        ``target_mean_correlation`` (rbar, None if ``N == 1``), ``note`` (str
        or None), ``n_observations`` and the estimates ``pi``, ``rho``,
        ``gamma``, ``kappa`` (None when not computed).
    """
    array = np.asarray(log_returns)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"log_returns must be real numbers, got dtype {array.dtype}")
    y = np.array(array, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] == 0:
        raise ValueError(f"log_returns must be 2-D with at least 2 rows and 1 column, got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("log_returns must be finite")
    t_obs, n = y.shape
    y = y - y.mean(axis=0)
    s = (y.T @ y) / t_obs
    variances = np.diag(s).copy()
    if not np.all(variances > 0):
        raise ValueError("log_returns has a zero-variance column; correlation undefined")
    sd = np.sqrt(variances)
    sample = s / sd[:, None] / sd[None, :]
    sample = (sample + sample.T) / 2
    np.clip(sample, -1.0, 1.0, out=sample)
    np.fill_diagonal(sample, 1.0)

    result: dict[str, Any] = {
        "correlation": sample, "shrinkage": 0.0, "target_mean_correlation": None,
        "note": None, "n_observations": int(t_obs),
        "pi": None, "rho": None, "gamma": None, "kappa": None,
    }
    if n >= 2:
        off = ~np.eye(n, dtype=bool)
        rbar = float(np.mean(sample[off]))
        result["target_mean_correlation"] = rbar
    if n < 3:
        result["note"] = ("fewer than 3 assets: the constant-correlation target equals "
                          "the sample correlation; no shrinkage applied")
        result["correlation"] = validate_correlation(sample)
        return result

    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, variances)
    gamma = float(np.sum((target - s) ** 2))
    # Loop over i keeps temporaries at O(T N) instead of O(T N^2).
    pi_mat = np.empty((n, n))
    theta = np.empty((n, n))
    for i in range(n):
        products = y[:, i:i + 1] * y - s[i]
        pi_mat[i] = np.mean(products**2, axis=0)
        theta[i] = np.mean((y[:, i:i + 1] ** 2 - s[i, i]) * products, axis=0)
    pi_hat = float(np.sum(pi_mat))
    ratio = np.sqrt(variances[None, :] / variances[:, None])  # sqrt(s_jj / s_ii)
    off_terms = ratio * theta + ratio.T * theta.T
    rho_hat = float(np.trace(pi_mat) + rbar / 2 * np.sum(off_terms[off]))
    result.update({"pi": pi_hat, "rho": rho_hat, "gamma": gamma})
    if gamma == 0:
        result["note"] = "sample correlation already constant (gamma = 0); no shrinkage applied"
        result["correlation"] = validate_correlation(sample)
        return result
    kappa = (pi_hat - rho_hat) / gamma
    delta = float(max(0.0, min(kappa / t_obs, 1.0)))
    shrunk = delta * rbar + (1.0 - delta) * sample
    shrunk = (shrunk + shrunk.T) / 2
    np.fill_diagonal(shrunk, 1.0)
    result.update({"kappa": float(kappa), "shrinkage": delta,
                   "correlation": validate_correlation(shrunk)})
    return result
