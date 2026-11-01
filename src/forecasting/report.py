"""Offline experimental reports and staged publication, separate from GBM."""

import json
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import tempfile

from simulation.report import _digest, prices_from_snapshot
from forecasting.core import ForecastConfig, evaluate


def run_experiment(snapshot, config):
    result = evaluate(_session_prices(snapshot), config)
    report = {"schema_version": 1, "model": "experimental_volatility_forecast",
              "historical_data": snapshot, "experiment": result,
              "environment": {"python": platform.python_version(), **{
                  name: version(name) for name in
                  ("numpy", "pandas", "scipy", "matplotlib")}}}
    json.dumps(report, allow_nan=False)
    return report


def _session_prices(snapshot):
    """Validate exact saved observations, then use local session wall dates.

    ISO offsets may change across daylight-saving transitions. A UTC copy is
    used only for shared snapshot validation; the original snapshot remains
    unchanged and the experiment retains local session dates without timezone.
    """
    import pandas as pd
    observations = snapshot["observations"]
    if _digest(observations) != snapshot["sha256"]:
        raise ValueError("Historical snapshot checksum mismatch.")
    stamps = [pd.Timestamp(row["date"]) for row in observations]
    if any(pd.isna(stamp) or stamp.isoformat() != row["date"]
           for stamp, row in zip(stamps, observations)):
        raise ValueError("Historical snapshot metadata mismatch.")
    aware = [stamp.tzinfo is not None for stamp in stamps]
    if any(aware) and not all(aware):
        raise ValueError("Historical snapshot mixes dated timezone conventions.")
    normalized = [{"date": (stamp.tz_convert("UTC") if stamp.tzinfo else stamp).isoformat(),
                   "close": row["close"]} for stamp, row in zip(stamps, observations)]
    canonical = dict(snapshot, observations=normalized, sha256=_digest(normalized))
    if (not observations or snapshot["first_date"] != observations[0]["date"]
            or snapshot["last_date"] != observations[-1]["date"]):
        raise ValueError("Historical snapshot metadata mismatch.")
    canonical.update(first_date=normalized[0]["date"], last_date=normalized[-1]["date"])
    prices = prices_from_snapshot(canonical)
    prices.index = pd.DatetimeIndex([stamp.tz_localize(None) for stamp in stamps])
    return prices


def read_experiment(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if (saved.get("schema_version") != 1 or
            saved.get("model") != "experimental_volatility_forecast"):
        raise ValueError("Unsupported experimental report schema/model")
    expected = run_experiment(saved["historical_data"],
                              ForecastConfig(**saved["experiment"]["configuration"]))
    if expected["experiment"] != saved["experiment"]:
        raise ValueError("Experimental replay result mismatch")
    return saved


def plot_experiment(report, path):
    import matplotlib.pyplot as plt
    import pandas as pd
    rows = report["experiment"]["test"]["forecasts"]
    dates = pd.DatetimeIndex([pd.Timestamp(row["target_date"]).tz_localize(None)
                              for row in rows])
    marker = "o" if len(rows) == 1 else None
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.plot(dates, [(252 * row["realized_squared_return"]) ** .5 for row in rows],
            color="0.65", alpha=.6, linewidth=.8, marker=marker, label="Realized |log return| × √252 (noisy proxy)")
    for model in ("historical", "ewma", "ml"):
        ax.plot(dates, [row["models"][model]["annualized_volatility"] for row in rows],
                label=model, linewidth=1.5, marker=marker)
    ax.set(title="Experimental one-session volatility forecasts — final test",
           xlabel="Target close date", ylabel="Annualized volatility (year⁻½)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.2)
    try:
        fig.savefig(path, dpi=150)
    except Exception:
        plt.close(fig)
        raise
    return fig


def publish_experiment(report, report_path, no_plot=True):
    report_path = Path(report_path)
    destination = report_path.with_suffix("")
    if report_path.exists() or destination.exists():
        raise FileExistsError(f"Execution already exists: {destination} or {report_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    figure = None
    try:
        if no_plot:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = plot_experiment(report, staging / "volatility.png")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        if report_path.exists() or destination.exists():
            raise FileExistsError(f"Execution already exists: {destination}")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    finally:
        if figure is not None:
            plt.close(figure)
    return destination
