"""Optional forecasting CLI: python -m forecasting (PYTHONPATH=src)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from forecasting.core import ForecastConfig
from forecasting.report import publish_experiment, read_experiment, run_experiment
from simulation.report import historical_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="snapshot JSON or existing GBM report")
    source.add_argument("--replay", type=Path, help="experimental report, strictly offline")
    source.add_argument("--ticker", help="download adjusted daily prices (requires network)")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--config", type=Path, help="ForecastConfig JSON; forbidden with replay")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.with_suffix("").exists():
        parser.error("output already exists; choose a new destination")
    if args.replay:
        if args.config:
            parser.error("replay uses saved configuration")
        saved = read_experiment(args.replay)
        snapshot = saved["historical_data"]
        config = ForecastConfig(**saved["experiment"]["configuration"])
    else:
        config = ForecastConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else ForecastConfig()
        if args.snapshot:
            data = json.loads(args.snapshot.read_text(encoding="utf-8"))
            snapshot = data.get("historical_data", data)
        else:
            from calibration.data import download_prices
            from calibration.fundamentals import normalize_ticker
            ticker = normalize_ticker(args.ticker)
            snapshot = historical_snapshot(download_prices(ticker, period=args.period),
                ticker=ticker, period=args.period, retrieved_at=datetime.now(timezone.utc).isoformat())
    report = run_experiment(snapshot, config)
    destination = publish_experiment(report, args.report)
    print("Final test (configurations selected only on validation):")
    print("model       n         MSE         MAE    quantile loss    violations")
    for name, metrics in report["experiment"]["test"]["metrics"].items():
        print(f"{name:10} {metrics['forecast_count']:4d} {metrics['mse']:11.5g} "
              f"{metrics['mae']:11.5g} {metrics['quantile_loss']:14.5g} "
              f"{metrics['violation_count']} ({metrics['violation_rate']:.2%})")
    print(f"Report: {destination / 'report.json'}")
    print(f"Figure: {destination / 'volatility.png'}")


if __name__ == "__main__":
    main()
