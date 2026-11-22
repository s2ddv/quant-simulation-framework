"""Expanding-window, refit-every-k-days out-of-sample 5-day tail backtest.

At origin i (information through daily return i only) every model forecasts the
distribution of the cumulative 5-day simple return R = exp(sum r_{i+1..i+5}) - 1.
Losses are -R. Parameters are re-estimated on r[:i+1] every ``refit_every``
origins and frozen in between (the GARCH variance filter is still updated daily
with the information up to the origin). Nothing after the origin is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from backtest.garch import (NU_MAX, NU_MIN, fit_garch_normal, fit_garch_t, next_variance,
                            simulate_cumulative)

THRESHOLDS = (0.03, 0.05, 0.08, 0.086, 0.10)
VOL_WINDOW = 21                  # trailing days for the ex-ante regime indicator
LEVELS = (0.95, 0.99)
MODELS = ("gaussian", "student_t", "garch_t")


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 5
    min_train: int = 1260          # ~5 years of daily returns before first forecast
    refit_every: int = 21          # ~monthly
    t_draws: int = 400_000         # per refit (origin-invariant between refits)
    garch_draws: int = 40_000      # per origin
    seed: int = 20260101


def threshold_key(x: float) -> str:
    """Column name for a loss threshold: 0.03 -> 'p3', 0.086 -> 'p8.6' (no collisions)."""
    return f"p{round(100 * x, 6):g}"


def _summarize_loss(loss: np.ndarray) -> dict:
    out = {threshold_key(x): float(np.mean(loss > x)) for x in THRESHOLDS}
    for a in LEVELS:
        var = float(np.quantile(loss, a))
        out[f"var{int(100 * a)}"] = var
        out[f"es{int(100 * a)}"] = float(np.mean(loss[loss >= var]))
    return out


def _loss_from_log(cum: np.ndarray) -> np.ndarray:
    return -np.expm1(cum)


def gaussian_forecast(mean_d: float, sd_d: float, h: int) -> dict:
    """Exact lognormal summary: log R_h ~ N(h*mean, h*sd^2)."""
    m, s = h * mean_d, sd_d * np.sqrt(h)
    out = {threshold_key(x): float(stats.norm.cdf((np.log1p(-x) - m) / s)) for x in THRESHOLDS}
    for a in LEVELS:
        z = stats.norm.ppf(1 - a)                       # lower-tail quantile of log return
        out[f"var{int(100 * a)}"] = float(-np.expm1(m + s * z))
        # E[loss | log R <= q] = 1 - exp(m + s^2/2) * Phi(z - s) / (1 - a)
        out[f"es{int(100 * a)}"] = float(1 - np.exp(m + 0.5 * s * s) * stats.norm.cdf(z - s) / (1 - a))
    return out


def _rng(seed: int, stream: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream, index)))


def run_backtest(returns: pd.Series, config: BacktestConfig = BacktestConfig()) -> tuple[pd.DataFrame, dict]:
    """``returns``: daily log returns indexed by date. Returns forecasts and metadata."""
    r = returns.to_numpy(dtype=float)
    h = config.horizon
    n = r.size
    first = config.min_train - 1
    last = n - h - 1                                    # need r[i+1..i+h]
    if last < first:
        raise ValueError("series too short for min_train and horizon")
    cs = np.concatenate([[0.0], np.cumsum(r)])
    rows, refits = [], []
    t_state = g_state = None
    garch_fallbacks = 0
    for i in range(first, last + 1):
        if (i - first) % config.refit_every == 0:
            window = r[:i + 1]
            # --- Student-t i.i.d. daily shocks ---
            df, loc, scale = stats.t.fit(window)
            df = float(np.clip(df, NU_MIN, NU_MAX))
            draws = stats.t.rvs(df, loc=loc, scale=scale,
                                size=(config.t_draws, h), random_state=_rng(config.seed, 1, i)).sum(axis=1)
            t_state = _summarize_loss(_loss_from_log(draws))
            # --- GARCH(1,1)-t ---
            fit = fit_garch_t(window, None if g_state is None else g_state["theta_pct"])
            used_previous = False
            if not fit["converged"] and g_state is not None:
                garch_fallbacks += 1
                used_previous = True
            else:
                g_state = fit
            refits.append({"index": i, "date": returns.index[i].isoformat(), "n_obs": i + 1,
                           "t_df": df, "garch_converged": fit["converged"],
                           "garch_used_previous_params": used_previous,
                           "garch": {k: fit[k] for k in ("mu", "omega", "alpha", "beta", "nu")},
                           "garch_message": fit["message"]})
            mean_d, sd_d = float(np.mean(window)), float(np.std(window, ddof=1))
        g_params = g_state
        # The first origin (i == first) always refits, so both states are set.
        assert g_params is not None and t_state is not None
        h_next = next_variance(r[:i + 1], g_params)
        cum = simulate_cumulative(g_params, h_next, h, config.garch_draws, _rng(config.seed, 2, i))
        row = {"index": i, "date": returns.index[i],
               # known at the origin: std of the last VOL_WINDOW returns, annualized (252)
               "trailing_vol21": float(np.std(r[i + 1 - VOL_WINDOW:i + 1], ddof=1) * np.sqrt(252)),
               "realized_loss": float(-np.expm1(cs[i + 1 + h] - cs[i + 1]))}
        for name, summ in (("gaussian", gaussian_forecast(mean_d, sd_d, h)), ("student_t", t_state),
                           ("garch_t", _summarize_loss(_loss_from_log(cum)))):
            for k, v in summ.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
    meta = {"config": asdict(config), "n_returns": n, "n_origins": len(rows), "n_refits": len(refits),
            "garch_nonconverged_refits": sum(not x["garch_converged"] for x in refits),
            "garch_previous_params_used": garch_fallbacks, "refits": refits,
            "convention": "loss = -(exp(sum of 5 daily log returns)-1); VaR=quantile of loss; ES=mean loss>=VaR"}
    return pd.DataFrame(rows).set_index("date"), meta


def run_garch_normal_backtest(returns: pd.Series, config: BacktestConfig = BacktestConfig()) -> tuple[pd.DataFrame, dict]:
    """Ablation: GARCH(1,1) with Gaussian innovations on the SAME origins, refit schedule and seed
    stream as ``run_backtest`` (same data, window, horizon). Columns are prefixed ``garch_n_``."""
    r = returns.to_numpy(dtype=float)
    h, n = config.horizon, r.size
    first, last = config.min_train - 1, n - h - 1
    if last < first:
        raise ValueError("series too short for min_train and horizon")
    rows, refits, state, fallbacks = [], [], None, 0
    for i in range(first, last + 1):
        if (i - first) % config.refit_every == 0:
            fit = fit_garch_normal(r[:i + 1], None if state is None else state["theta_pct"])
            used_prev = False
            if not fit["converged"] and state is not None:
                fallbacks += 1; used_prev = True
            else:
                state = fit
            refits.append({"index": i, "date": returns.index[i].isoformat(), "converged": fit["converged"],
                           "used_previous_params": used_prev, "message": fit["message"],
                           "params": {k: fit[k] for k in ("mu", "omega", "alpha", "beta")}})
        cum = simulate_cumulative(state, next_variance(r[:i + 1], state), h, config.garch_draws, _rng(config.seed, 3, i))
        row = {"index": i, "date": returns.index[i]}
        row.update({f"garch_n_{k}": v for k, v in _summarize_loss(_loss_from_log(cum)).items()})
        rows.append(row)
    meta = {"config": asdict(config), "n_origins": len(rows), "n_refits": len(refits),
            "nonconverged_refits": sum(not x["converged"] for x in refits), "previous_params_used": fallbacks,
            "refits": refits}
    return pd.DataFrame(rows).set_index("date"), meta
