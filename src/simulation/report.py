"""JSON simulation records with historical snapshots for offline replay."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import NormalDist

import numpy as np
import pandas as pd

from calibration.estimation import TRADING_DAYS, estimate_gbm_params
from calibration.prices import validate_prices
from simulation.gbm import simulate_gbm
from risk.intervals import cvar_interval, prob_loss_interval, var_interval
from risk.statistics import risk_summary
from validation.statistics import compare_terminal, convergence_analysis, terminal_theory
from validation.out_of_sample import evaluate_out_of_sample
from validation.calibration_sensitivity import calibration_sensitivity


@dataclass(frozen=True)
class SimulationConfig:
    horizon: float = 1.0
    steps: int = 252
    n_sims: int = 1000
    seed: int = 42
    confidence: float = 0.95

    def __post_init__(self):
        if not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("horizon must be positive and finite.")
        for name in ("steps", "n_sims", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid range.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be strictly between zero and one.")


def _digest(observations):
    return sha256(json.dumps(observations, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def historical_snapshot(prices, *, ticker, period, retrieved_at, source="yfinance"):
    """Record the exact adjusted closes, observation dates and retrieval time."""
    prices = validate_prices(prices, min_observations=3, require_dates=True)
    timestamp = pd.Timestamp(retrieved_at)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone.")
    observations = [{"date": date.isoformat(), "close": float(price)}
                    for date, price in prices.items()]
    return {"ticker": ticker, "requested_period": period, "source": source,
            "price_field": "adjusted_close", "interval": "1d",
            "retrieved_at": timestamp.isoformat(),
            "first_date": observations[0]["date"],
            "last_date": observations[-1]["date"], "count": len(prices),
            "sha256": _digest(observations), "observations": observations}


def prices_from_snapshot(snapshot):
    observations = snapshot["observations"]
    if _digest(observations) != snapshot["sha256"]:
        raise ValueError("Historical snapshot checksum mismatch.")
    prices = pd.Series([row["close"] for row in observations],
                       index=pd.DatetimeIndex([row["date"] for row in observations]))
    rebuilt = historical_snapshot(prices, ticker=snapshot["ticker"],
                                  period=snapshot["requested_period"],
                                  retrieved_at=snapshot["retrieved_at"],
                                  source=snapshot["source"])
    if rebuilt != snapshot:
        raise ValueError("Historical snapshot metadata mismatch.")
    return prices


def terminal_mean_confidence_interval(terminal, *, level=0.95):
    """Approximate normal CI for the iid GBM terminal-price sample mean.

    Conditional on fixed GBM parameters. The standard error is estimated from
    the simulated sample; the CLT approximation is undefined for a single path.
    """
    sample = np.asarray(terminal, dtype=float)
    if (sample.ndim != 1 or sample.size == 0 or not np.isfinite(sample).all()
            or np.any(sample <= 0) or not 0 < level < 1):
        raise ValueError("Require finite positive terminal prices and 0 < level < 1.")
    result = {"level": level, "unit": "asset_price_currency",
              "method": "normal_clt_sample_standard_error",
              "parameter_scope": "fixed_calibrated_gbm_parameters",
              "lower": None, "upper": None}
    if sample.size == 1:
        return result
    # Scale before squaring deviations: both tiny and huge finite prices can
    # have a representable standard error despite an unscaled variance that
    # underflows or overflows float64.
    scale = float(np.max(sample))
    normalized = sample / scale
    estimate = float(np.mean(normalized) * scale)
    standard_error = float((np.std(normalized, ddof=1)
                            / np.sqrt(sample.size)) * scale)
    half_width = NormalDist().inv_cdf((1 + level) / 2) * standard_error
    lower, upper = estimate - half_width, estimate + half_width
    if not np.isfinite([estimate, standard_error, lower, upper]).all():
        raise ValueError("Terminal mean confidence interval exceeds finite precision.")
    result.update({"lower": float(lower), "upper": float(upper)})
    return result


def risk_intervals(terminal, s0, confidence, *, level=0.95):
    """Monte Carlo estimator intervals for the risk metrics (E1.4).

    These quantify only Monte Carlo error of the point estimates with mu and
    sigma held fixed; they are not price percentiles and do not include
    calibration uncertainty (see ``calibration_sensitivity``). The samples
    are assumed i.i.d. (plain Monte Carlo). The CVaR uses the asymptotic
    interval: it is O(N), deterministic and needs no extra RNG stream.
    """
    return {
        "interval_kind": "monte_carlo_estimator",
        "level": level,
        "prob_loss": prob_loss_interval(terminal, s0, level=level, method="wilson"),
        "var": var_interval(terminal, s0, confidence=confidence, level=level),
        "cvar": cvar_interval(terminal, s0, confidence=confidence, level=level,
                              method="asymptotic"),
    }


def run_simulation(snapshot, config, out_of_sample_config=None, sensitivity_config=None):
    """Return a deterministic, JSON-safe report and its simulated paths."""
    prices = prices_from_snapshot(snapshot)
    mu, sigma = estimate_gbm_params(prices)
    s0 = float(prices.iloc[-1])
    theory = terminal_theory(s0, mu, sigma, config.horizon)
    paths = simulate_gbm(s0, mu, sigma, config.horizon, config.steps,
                         config.n_sims, config.seed)
    comparison = compare_terminal(paths[-1], theory)
    report = {
        "schema_version": 1, "model": "single_asset_gbm",
        "historical_data": snapshot, "last_price": s0,
        "parameters": {"mu": mu, "sigma": sigma,
                       "trading_days_per_year": TRADING_DAYS,
                       "log_return_std_ddof": 1},
        "simulation": asdict(config),
        "conventions": {"horizon_unit": "years", "terminal_variance_ddof": 1,
                        "quantile_method": "linear", "rng": "NumPy default_rng/PCG64",
                        "convergence": "nested prefixes of the same terminal sample",
                        "undefined_values": "null (N=1 variance or zero-reference relative error)"},
        "results": {"terminal": comparison,
                    "terminal_mean_confidence_interval":
                        terminal_mean_confidence_interval(paths[-1]),
                    "risk": risk_summary(paths, config.confidence),
                    "risk_intervals": risk_intervals(paths[-1], s0, config.confidence)},
        "convergence": convergence_analysis(paths[-1], theory),
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in
                           ("numpy", "pandas", "scipy", "matplotlib", "yfinance")}},
    }
    if out_of_sample_config is not None:
        if not isinstance(out_of_sample_config, dict):
            raise ValueError("out_of_sample_config must be a mapping")
        diagnostic = evaluate_out_of_sample(prices, **out_of_sample_config)
        diagnostic["historical_snapshot"] = {
            key: snapshot[key] for key in
            ("source", "price_field", "retrieved_at", "first_date", "last_date", "sha256")
        }
        report["out_of_sample"] = diagnostic
    if sensitivity_config is not None:
        if not isinstance(sensitivity_config, dict):
            raise ValueError("sensitivity_config must be a mapping")
        report["calibration_sensitivity"] = calibration_sensitivity(
            s0, mu, sigma, config.horizon, alpha=config.confidence,
            **sensitivity_config)
    # Fail on nonfinite computations instead of writing nonstandard JSON.
    json.dumps(report, allow_nan=False)
    return report, paths


def write_report(report, path):
    path = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_report(path):
    """Read a validated report, including any saved company snapshot."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report["schema_version"] != 1 or report["model"] != "single_asset_gbm":
        raise ValueError("Unsupported report schema/model.")
    snapshot = report["historical_data"]
    prices_from_snapshot(snapshot)  # Validates the saved snapshot; value unused.
    config = SimulationConfig(**report["simulation"])
    historical_config = None
    if "out_of_sample" in report:
        saved = report["out_of_sample"]
        protocol = saved["protocol"]
        historical_config = {
            "calibration_observations": protocol["calibration_observations"],
            "horizon_days": protocol["horizon_days"], "alpha": protocol["alpha"]}
        for key in ("calibration_mode", "recalibration_frequency"):
            if key in protocol:
                historical_config[key] = protocol[key]
    sensitivity_config = None
    if "calibration_sensitivity" in report:
        saved_sensitivity = report["calibration_sensitivity"]
        sensitivity_config = {
            "mu_delta": saved_sensitivity["configuration"]["mu_delta"],
            "sigma_relative_delta": saved_sensitivity["configuration"]["sigma_relative_delta"]}
        if "scenario_levels" in saved_sensitivity["configuration"]:
            sensitivity_config["scenario_levels"] = saved_sensitivity["configuration"]["scenario_levels"]
    expected, _ = run_simulation(snapshot, config, historical_config,
                                 sensitivity_config)
    # Risk intervals were added after schema v1 was published; older v1
    # reports without them still replay against the remaining results.
    if isinstance(report.get("results"), dict) and "risk_intervals" not in report["results"]:
        expected["results"].pop("risk_intervals", None)
    for key in ("last_price", "parameters", "conventions", "results", "convergence"):
        if key not in report or report[key] != expected[key]:
            raise ValueError(f"Replay result mismatch in {key}: saved values differ from the reproduced run.")
    if "out_of_sample" in report:
        saved_diagnostic = report["out_of_sample"]
        expected_diagnostic = expected["out_of_sample"]
        # The Wilson interval was added after the original diagnostic schema.
        if "violation_rate_interval" not in saved_diagnostic.get("evaluation", {}):
            expected_diagnostic["evaluation"].pop("violation_rate_interval", None)
        if saved_diagnostic != expected_diagnostic:
            raise ValueError("Replay result mismatch in out_of_sample: saved values differ from the reproduced run.")
    if "calibration_sensitivity" in report and (report["calibration_sensitivity"]
                                                 != expected["calibration_sensitivity"]):
        raise ValueError("Replay result mismatch in calibration_sensitivity.")
    return report


def read_replay(path):
    report = read_report(path)
    snapshot = report["historical_data"]
    return snapshot, SimulationConfig(**report["simulation"])
