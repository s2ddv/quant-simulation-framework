"""Chart-ready, JSON-safe series built from a saved single-asset GBM report.

This module only reshapes data; it draws nothing and adds no dependency. Its
output feeds an HTML or dashboard front end (Plotly, Streamlit, Dash) without
that front end having to know the report schema or recompute statistics.

Units follow the report: prices and losses in the asset currency, time in
years, ``mu`` in year^-1 and ``sigma`` in year^-1/2. Percentile bands are
pointwise percentiles of the simulated sample under fixed calibrated
parameters; they exclude calibration uncertainty and model error.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np

from simulation.gbm import simulate_gbm
from simulation.plotting import _validate_paths
from simulation.report import prices_from_snapshot
from validation.statistics import terminal_theory

SCHEMA = "dashboard_data_v1"
FAN_PERCENTILES = (5, 25, 50, 75, 95)


def _floats(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _row_indices(n_rows: int, max_points: int | None) -> np.ndarray:
    """Evenly spaced row indices that always keep the first and last rows."""
    if max_points is None or max_points >= n_rows:
        return np.arange(n_rows)
    if type(max_points) is not int or max_points < 2:
        raise ValueError("max_points must be an integer >= 2 or None.")
    return np.unique(np.linspace(0, n_rows - 1, max_points).round().astype(int))


def history_series(snapshot: dict) -> dict:
    """Observed adjusted closes from a checksummed historical snapshot."""
    prices = prices_from_snapshot(snapshot)
    return {"dates": [date.date().isoformat() for date in prices.index],
            "close": _floats(prices.to_numpy())}


def fan_chart(paths: np.ndarray, *, mu: float, sigma: float, horizon: float,
              percentiles=FAN_PERCENTILES, n_sample_paths: int = 24,
              max_points: int | None = None) -> dict:
    """Pointwise sample percentiles, analytical benchmarks and example paths.

    ``paths`` keeps the project contract ``(steps + 1, n_sims)`` with row zero
    equal to the initial price. Time of row ``k`` is ``k * horizon / steps``.
    Analytical series are the exact GBM mean and 5%/50%/95% quantiles at each
    retained time. ``max_points`` thins rows for display only.
    """
    paths = _validate_paths(paths)
    if type(n_sample_paths) is not int or n_sample_paths < 0:
        raise ValueError("n_sample_paths must be a nonnegative integer.")
    percentiles = tuple(percentiles)
    if not percentiles or any(not 0 <= p <= 100 for p in percentiles):
        raise ValueError("percentiles must lie in [0, 100].")
    steps = paths.shape[0] - 1
    s0 = float(paths[0, 0])
    rows = _row_indices(paths.shape[0], max_points)
    times = rows * (float(horizon) / steps)
    bands = np.percentile(paths[rows], percentiles, axis=1)
    theory = [terminal_theory(s0, mu, sigma, float(t)) for t in times]
    chosen = np.linspace(0, paths.shape[1] - 1, min(n_sample_paths, paths.shape[1]),
                         dtype=int)
    return {
        "time_years": _floats(times),
        "step": [int(row) for row in rows],
        "percentiles": {f"p{p:g}": _floats(band) for p, band in zip(percentiles, bands)},
        "analytical": {
            "mean": [item["mean"] for item in theory],
            **{f"q{float(level):g}": [item["quantiles"][level] for item in theory]
               for level in ("0.05", "0.5", "0.95")},
        },
        "sample_paths": [_floats(paths[rows, index]) for index in chosen],
        "sample_path_indices": [int(index) for index in chosen],
    }


def lognormal_density(x, *, s0: float, mu: float, sigma: float, horizon: float) -> np.ndarray:
    """Exact density of S(T) under GBM; zero for x <= 0."""
    if sigma <= 0 or horizon <= 0:
        raise ValueError("The density needs sigma > 0 and horizon > 0.")
    x = np.asarray(x, dtype=float)
    scale = sigma * math.sqrt(horizon)
    location = math.log(s0) + (mu - sigma ** 2 / 2) * horizon
    density = np.zeros_like(x)
    positive = x > 0
    log_x = np.log(x[positive])
    density[positive] = np.exp(-((log_x - location) ** 2) / (2 * scale ** 2)
                               - log_x - math.log(scale * math.sqrt(2 * math.pi)))
    return density


def terminal_distribution(terminal, *, s0: float, mu: float, sigma: float,
                          horizon: float, bins: int = 70,
                          density_points: int = 200) -> dict:
    """Histogram of terminal prices with the lognormal density on the count scale.

    ``expected_count`` is ``n * pdf(x) * bin_width``, so it overlays the bars
    directly. It is omitted (None) for the degenerate sigma = 0 or T = 0 case.
    """
    terminal = np.asarray(terminal, dtype=float)
    if terminal.ndim != 1 or terminal.size == 0 or not np.isfinite(terminal).all():
        raise ValueError("terminal must be a nonempty vector of finite prices.")
    if type(bins) is not int or bins < 1 or type(density_points) is not int or density_points < 2:
        raise ValueError("bins must be >= 1 and density_points >= 2 (integers).")
    counts, edges = np.histogram(terminal, bins=bins)
    curve = None
    if sigma > 0 and horizon > 0:
        x = np.linspace(edges[0], edges[-1], density_points)
        width = float(edges[1] - edges[0])
        pdf = lognormal_density(x, s0=s0, mu=mu, sigma=sigma, horizon=horizon)
        curve = {"price": _floats(x), "density": _floats(pdf),
                 "expected_count": _floats(pdf * terminal.size * width)}
    return {"bin_edges": _floats(edges), "counts": [int(c) for c in counts],
            "n": int(terminal.size), "lognormal": curve}


def comparison_table(report: dict) -> list[dict]:
    """Simulated vs analytical terminal statistics as table rows."""
    terminal = report["results"]["terminal"]
    entries = [("mean", terminal["mean"]), ("variance", terminal["variance"])]
    entries += [(f"quantile_{level}", terminal["quantiles"][level])
                for level in sorted(terminal["quantiles"], key=float)]
    return [{"statistic": name, **values} for name, values in entries]


def convergence_series(report: dict, *, level: float = 0.95) -> dict:
    """Absolute mean error per nested prefix and the analytical z*SE envelope.

    Prefixes are nested subsets of one sample, so successive points are
    correlated and need not decrease monotonically.
    """
    z = NormalDist().inv_cdf(0.5 + level / 2)
    rows = report["convergence"]
    return {
        "n_sims": [row["n_sims"] for row in rows],
        "mean_absolute_error": [row["mean"]["absolute_error"] for row in rows],
        "mean_error_envelope": [z * row["mean_standard_error"] for row in rows],
        "variance_relative_error": [row["variance"]["relative_error"] for row in rows],
        "envelope_level": level,
    }


def sensitivity_grid(report: dict) -> dict | None:
    """Analytical scenario grid as matrices indexed [sigma_row][mu_column]."""
    saved = report.get("calibration_sensitivity")
    if saved is None:
        return None
    scenarios = saved["scenarios"]
    mu_axis = sorted({s["mu"] for s in scenarios})
    sigma_axis = sorted({s["sigma"] for s in scenarios})
    if len(mu_axis) * len(sigma_axis) != len(scenarios):
        raise ValueError("Sensitivity scenarios do not form a complete grid.")
    column = {value: j for j, value in enumerate(mu_axis)}
    row = {value: i for i, value in enumerate(sigma_axis)}
    levels = list(scenarios[0]["quantiles"])
    names = ["var", "mean"] + [f"q{float(level):g}" for level in levels]
    grid = {name: [[math.nan] * len(mu_axis) for _ in sigma_axis] for name in names}
    for scenario in scenarios:
        i, j = row[scenario["sigma"]], column[scenario["mu"]]
        values = [scenario["var"], scenario["mean"]] + [scenario["quantiles"][level] for level in levels]
        for name, value in zip(names, values):
            grid[name][i][j] = value
    if any(math.isnan(value) for matrix in grid.values() for line in matrix for value in line):
        raise ValueError("Sensitivity scenarios repeat a (mu, sigma) cell.")
    return {"mu": mu_axis, "sigma": sigma_axis, "metrics": grid,
            "baseline": {"mu": saved["baseline"]["mu"], "sigma": saved["baseline"]["sigma"]},
            "alpha": saved["configuration"]["alpha"]}


def backtest_series(report: dict) -> dict | None:
    """Historical VaR forecasts against realized losses, with summary metrics."""
    saved = report.get("out_of_sample")
    if saved is None:
        return None
    evaluation = saved["evaluation"]
    rows = evaluation["rows"]
    return {
        "target_date": [row["target_date"][:10] for row in rows],
        "var": [row["var"] for row in rows],
        "realized_loss": [row["realized_loss"] for row in rows],
        "violation": [bool(row["var_violation"]) for row in rows],
        "metrics": evaluation["metrics"],
        "violation_rate_interval": evaluation.get("violation_rate_interval"),
        "protocol": {key: saved["protocol"].get(key) for key in
                     ("calibration_observations", "calibration_mode",
                      "recalibration_frequency", "horizon_days", "alpha")},
    }


def regenerate_paths(report: dict) -> np.ndarray:
    """Rebuild the report's path matrix from its seed and check it matches.

    The check compares the terminal sample mean exactly with the saved value.
    A mismatch means another NumPy version or engine produced the report.
    Memory is ``(steps + 1) * n_sims`` float64 values, guarded by the simulator.
    """
    config, parameters = report["simulation"], report["parameters"]
    paths = simulate_gbm(report["last_price"], parameters["mu"], parameters["sigma"],
                         config["horizon"], config["steps"], config["n_sims"],
                         config["seed"])
    if float(paths[-1].mean()) != report["results"]["terminal"]["mean"]["estimate"]:
        raise ValueError("Regenerated paths do not reproduce the saved terminal mean; "
                         "pass the original paths or rerun the simulation.")
    return paths


def build_dashboard_data(report: dict, paths: np.ndarray | None = None, *,
                         n_sample_paths: int = 24, max_points: int | None = None,
                         bins: int = 70) -> dict[str, Any]:
    """Assemble every chart series for one report.

    ``paths`` defaults to a verified regeneration from the report's seed.
    Optional sections (sensitivity, backtest) are None when the report lacks them.
    """
    if report.get("schema_version") != 1 or report.get("model") != "single_asset_gbm":
        raise ValueError("Unsupported report schema/model.")
    config, parameters = report["simulation"], report["parameters"]
    if paths is None:
        paths = regenerate_paths(report)
    paths = _validate_paths(paths)
    if paths.shape != (config["steps"] + 1, config["n_sims"]):
        raise ValueError("paths shape does not match the report configuration.")
    if float(paths[0, 0]) != report["last_price"]:
        raise ValueError("paths do not start at the report's last price.")
    s0, mu, sigma, horizon = (report["last_price"], parameters["mu"],
                              parameters["sigma"], config["horizon"])
    risk = report["results"]["risk"]
    snapshot = report["historical_data"]
    data = {
        "schema": SCHEMA,
        "units": {"price": "asset_currency", "time": "years", "mu": "year^-1",
                  "sigma": "year^-1/2"},
        "summary": {
            "ticker": snapshot["ticker"], "last_date": snapshot["last_date"][:10],
            "last_price": s0, "mu": mu, "sigma": sigma, **config,
            "expected_price": risk["expected_price"], "var": risk["var"],
            "cvar": risk["cvar"], "prob_loss": risk["prob_loss"],
            "var_threshold_price": s0 - risk["var"],
            "cvar_threshold_price": s0 - risk["cvar"],
            "terminal_mean_confidence_interval":
                report["results"]["terminal_mean_confidence_interval"],
        },
        "history": history_series(snapshot),
        "fan": fan_chart(paths, mu=mu, sigma=sigma, horizon=horizon,
                         n_sample_paths=n_sample_paths, max_points=max_points),
        "terminal": terminal_distribution(paths[-1], s0=s0, mu=mu, sigma=sigma,
                                          horizon=horizon, bins=bins),
        "comparison": comparison_table(report),
        "convergence": convergence_series(report),
        "sensitivity": sensitivity_grid(report),
        "backtest": backtest_series(report),
        "provenance": {
            "snapshot": {key: snapshot[key] for key in
                         ("source", "price_field", "first_date", "last_date",
                          "count", "retrieved_at", "sha256")},
            "conventions": report["conventions"],
            "environment": report.get("environment"),
        },
    }
    return data
