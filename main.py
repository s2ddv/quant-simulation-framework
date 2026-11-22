"""Analyze a selected company and simulate its price with GBM."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import sys

# Allow `python main.py` from the repository without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.data import download_prices
from calibration.fundamentals import download_company, format_company, normalize_ticker
from simulation.plotting import (plot_convergence, plot_final_distribution,
                                plot_paths)
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                              run_simulation, write_report)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number") from exc
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return number


def unit_interval(value: str) -> float:
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def confidence_level(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be between 0 and 1") from exc
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--next-session', action='store_true',
                        help='experimental daily forecast mode; use --next-session --help')
    parser.add_argument("--ticker", type=normalize_ticker,
                        help="market ticker (asks when omitted; e.g. PETR4.SA, AAPL)")
    parser.add_argument("--period", default="3y", help="price history period")
    parser.add_argument("--horizon", type=positive_float, default=1.0,
                        help="simulation horizon in years (default: 1)")
    parser.add_argument("--steps", type=positive_int, default=252,
                        help="steps per simulation (default: 252)")
    parser.add_argument("--n-sims", type=positive_int, default=1000,
                        help="number of paths (default: 1000)")
    parser.add_argument("--seed", type=nonnegative_int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--confidence", type=confidence_level, default=0.95,
                        help="VaR confidence level (default: 0.95)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the interactive chart")
    parser.add_argument("--interactive", action="store_true",
                        help="also save paths.html, an interactive dark fan chart "
                             "(requires: pip install -e \".[interactive]\")")
    parser.add_argument("--report", type=Path,
                        help="output run name (default: a free name under reports/)")
    parser.add_argument("--replay", type=Path,
                        help="replay a report offline; uses its data and simulation settings")
    parser.add_argument("--out-of-sample", action="store_true",
                        help="add chronological historical diagnostics")
    parser.add_argument("--calibration-observations", type=positive_int,
                        help="number of initial closes used only for historical calibration")
    parser.add_argument("--evaluation-horizon-days", type=positive_int, default=1,
                        help="nonoverlapping historical horizon in trading observations (default: 1)")
    parser.add_argument("--evaluation-confidence", type=confidence_level, default=0.95,
                        help="historical VaR confidence (default: 0.95)")
    parser.add_argument("--calibration-mode", choices=("fixed", "rolling", "expanding"),
                        default="fixed", help="historical calibration window (default: fixed)")
    parser.add_argument("--recalibration-frequency", type=positive_int, default=1,
                        help="refit every N forecast origins (default: 1)")
    parser.add_argument("--calibration-sensitivity", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="analytic parameter scenarios (enabled for new runs; replay keeps saved settings)")
    parser.add_argument("--scenario-levels", type=int, choices=(3, 5, 7, 9, 100), default=100,
                        help="levels per parameter: 9/25/49/81/10000 scenarios (default: 100)")
    parser.add_argument("--mu-delta", type=nonnegative_float, default=0.02,
                        help="absolute annual drift perturbation (default: 0.02)")
    parser.add_argument("--sigma-relative-delta", type=unit_interval, default=0.2,
                        help="relative volatility perturbation (default: 0.2)")
    return parser


def default_report_path() -> Path:
    """Choose a free implicit name without altering any previous run."""
    preferred = Path("reports/simulation.json")
    if not preferred.exists() and not preferred.with_suffix("").exists():
        return preferred
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = preferred.with_name(f"simulation-{stamp}.json")
    suffix = 1
    while candidate.exists() or candidate.with_suffix("").exists():
        candidate = preferred.with_name(f"simulation-{stamp}-{suffix}.json")
        suffix += 1
    return candidate


def check_destination(report_path: Path) -> Path:
    """Reject an occupied run name before acquiring data or simulating."""
    destination = report_path.with_suffix("")
    if destination.exists() or report_path.exists():
        raise FileExistsError(
            f"Execution already exists: {destination} or {report_path}. Choose another --report name.")
    return destination


def _risk_interval_lines(intervals) -> list[str]:
    """Monte Carlo estimator intervals; absent in reports saved before E1."""
    if not intervals:
        return []
    level = intervals["level"]
    lines = [f"IC {level:.0%} do estimador Monte Carlo (erro de amostragem, "
             "mu/sigma fixos; não é percentil de preço):"]
    for key, label in (("prob_loss", "P(perda)"), ("var", "VaR"), ("cvar", "CVaR")):
        item = intervals[key]
        if item["lower"] is None or item["upper"] is None:
            bounds = ("indefinido (poucas perdas na cauda)" if item.get("note")
                      else "indefinido")
        else:
            bounds = f"[{item['lower']:.6g}, {item['upper']:.6g}]"
        if item.get("saturated") and item.get("coverage") is not None:
            bounds += f" (saturado; cobertura real {item['coverage']:.1%})"
        lines.append(f"  {label}: {item['estimate']:.6g} {bounds}")
    return lines


def format_simulation_summary(report, config, terminal_mean, terminal_std) -> str:
    """Build the terminal summary without calculation or I/O side effects."""
    snapshot = report["historical_data"]
    s0 = report["last_price"]
    mu, sigma = report["parameters"]["mu"], report["parameters"]["sigma"]
    lines = [f"\nSimulação GBM: {snapshot['ticker']} (preços históricos ajustados)",
             f"Último preço: {s0:.2f}", f"mu = {mu:.2%} | sigma = {sigma:.2%}",
             str(report["results"]["risk"]),
             *_risk_interval_lines(report["results"].get("risk_intervals")),
             f"Preço esperado teórico ({config.horizon:g} ano(s)): "
             f"{report['results']['terminal']['mean']['theoretical']:.2f}",
             f"Média dos preços finais:  {terminal_mean}",
             f"Desvio padrão dos preços finais:  {terminal_std}",
             "Validação analítica (estimativa, teoria e erros):",
             str(report["results"]["terminal"])]
    if "calibration_sensitivity" in report:
        sensitivity = report["calibration_sensitivity"]
        scenarios = sensitivity["scenarios"]
        lines.extend([
            f"\nSensibilidade: {len(scenarios)} cenários de drift e volatilidade (GBM).",
            "Valores na moeda do ativo; mu anual e sigma anualizada. * = cenário-base.",
            "       mu      sigma       Média          P5         P50         P95         VaR",
        ])
        displayed = scenarios if len(scenarios) <= 81 else (scenarios[0], sensitivity["baseline"], scenarios[-1])
        for row in displayed:
            marker = "*" if row == sensitivity["baseline"] else " "
            quantiles = row["quantiles"]
            lines.append(
                f"{marker} {row['mu']:9.2%} {row['sigma']:9.2%} "
                f"{row['mean']:11.2f} {quantiles['0.05']:11.2f} "
                f"{quantiles['0.5']:11.2f} {quantiles['0.95']:11.2f} {row['var']:11.2f}")
        if len(scenarios) > 81:
            lines.append("Tabela resumida: primeiro, base e último cenário; todos constam no JSON.")
        lines.append("P5/P50/P95 são percentis condicionais de preços; cenários não são "
                     "previsões garantidas nem intervalos de confiança. Grades mais densas "
                     "podem manter os mesmos extremos.")
    if "out_of_sample" in report:
        evaluation = report["out_of_sample"]["evaluation"]
        metrics = evaluation["metrics"]
        lines.append("Cobertura histórica observada (janelas sem sobreposição): "
                     f"{metrics['violation_count']}/{metrics['forecast_count']} violações; "
                     f"taxa observada {metrics['observed_violation_rate']:.2%}, "
                     f"referência nominal {metrics['nominal_violation_rate']:.2%}.")
        interval = evaluation.get("violation_rate_interval")
        if interval:
            lines.append(f"IC {interval['confidence_level']:.0%} Wilson da taxa observada "
                         f"({metrics['forecast_count']} previsões): "
                         f"[{interval['lower']:.2%}, {interval['upper']:.2%}].")
        lines.extend(f"Aviso: {warning}" for warning in evaluation["warnings"])
    return "\n".join(lines)


def publish_run(report, paths, config, report_path, no_plot, interactive=False):
    """Write a complete bundle to staging, then publish it atomically."""
    destination = report_path.with_suffix("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    figures = []
    plt = None
    published = False
    try:
        if no_plot:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ticker = report["historical_data"]["ticker"]
        figures.append(plot_convergence(report["convergence"], staging / "convergence.png"))
        figures.append(plot_paths(paths, title=f"GBM price fan chart: {ticker}",
                                  save_path=staging / "paths.png", style="fan"))
        figures.append(plot_final_distribution(
            paths, var=report["results"]["risk"]["var"],
            title=f"Final prices: {ticker} (VaR confidence: {config.confidence:.1%})",
            save_path=staging / "final-distribution.png"))
        if interactive:
            from simulation.interactive import plot_fan_html
            plot_fan_html(paths, title=f"GBM price fan chart: {ticker}",
                          save_path=staging / "paths.html")
        write_report(report, staging / "report.json")
        if destination.exists() or report_path.exists():
            raise FileExistsError(f"Execution already exists: {destination} or {report_path}.")
        staging.rename(destination)
        published = True
    except Exception:
        shutil.rmtree(staging)
        raise
    finally:
        if plt is not None:
            try:
                if published and not no_plot:
                    plt.show()
            finally:
                for figure in figures:
                    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if '--next-session' in tokens:
        from forecasting.daily import main as daily_main
        tokens.remove('--next-session')
        return daily_main(tokens)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interactive:
        try:
            import plotly  # noqa: F401
        except ImportError:
            parser.error('--interactive needs plotly: pip install -e ".[interactive]"')
    if args.report is None:
        args.report = default_report_path()
    check_destination(args.report)
    company = None
    if args.replay:
        saved = read_report(args.replay)
        snapshot = saved["historical_data"]
        config = SimulationConfig(**saved["simulation"])
        company = saved.get("company_analysis")
        historical_config = None
        if "out_of_sample" in saved:
            protocol = saved["out_of_sample"]["protocol"]
            historical_config = {
                "calibration_observations": protocol["calibration_observations"],
                "horizon_days": protocol["horizon_days"], "alpha": protocol["alpha"]}
            for key in ("calibration_mode", "recalibration_frequency"):
                if key in protocol:
                    historical_config[key] = protocol[key]
        sensitivity_config = None
        if "calibration_sensitivity" in saved:
            sensitivity_config = {
                "mu_delta": saved["calibration_sensitivity"]["configuration"]["mu_delta"],
                "sigma_relative_delta": saved["calibration_sensitivity"]["configuration"]["sigma_relative_delta"]}
            if "scenario_levels" in saved["calibration_sensitivity"]["configuration"]:
                sensitivity_config["scenario_levels"] = saved["calibration_sensitivity"]["configuration"]["scenario_levels"]
        if args.out_of_sample:
            if args.calibration_observations is None:
                parser.error("--out-of-sample requires --calibration-observations")
            historical_config = {
                "calibration_observations": args.calibration_observations,
                "horizon_days": args.evaluation_horizon_days,
                "alpha": args.evaluation_confidence,
                "calibration_mode": args.calibration_mode,
                "recalibration_frequency": args.recalibration_frequency}
        if args.calibration_sensitivity:
            sensitivity_config = {"mu_delta": args.mu_delta,
                                  "sigma_relative_delta": args.sigma_relative_delta,
                                  "scenario_levels": args.scenario_levels}
        elif args.calibration_sensitivity is False:
            sensitivity_config = None
    else:
        if args.out_of_sample and args.calibration_observations is None:
            parser.error("--out-of-sample requires --calibration-observations")
        if not args.out_of_sample and args.calibration_observations is not None:
            parser.error("--calibration-observations requires --out-of-sample")
        if not args.out_of_sample and (args.calibration_mode != "fixed"
                                      or args.recalibration_frequency != 1):
            parser.error("calibration mode/frequency requires --out-of-sample")
        historical_config = ({"calibration_observations": args.calibration_observations,
                              "horizon_days": args.evaluation_horizon_days,
                              "alpha": args.evaluation_confidence,
                              "calibration_mode": args.calibration_mode,
                              "recalibration_frequency": args.recalibration_frequency}
                             if args.out_of_sample else None)
        sensitivity_config = ({"mu_delta": args.mu_delta,
                               "sigma_relative_delta": args.sigma_relative_delta,
                               "scenario_levels": args.scenario_levels}
                              if args.calibration_sensitivity is not False else None)
        if args.ticker is None:
            while True:
                try:
                    args.ticker = normalize_ticker(input(
                        "Digite o ticker da ação (ex.: PETR4.SA, VALE3.SA, AAPL): "))
                    break
                except ValueError as exc:
                    print(exc)
                except (EOFError, KeyboardInterrupt):
                    parser.error("Informe uma ação com --ticker (ex.: --ticker PETR4.SA).")
        print(f"Consultando {args.ticker}...")
        company = download_company(args.ticker)
        print(format_company(company))
        try:
            prices = download_prices(args.ticker, period=args.period)
        except Exception as exc:
            parser.exit(1, f"Não foi possível obter preços de {args.ticker}: {exc}. "
                        "Confira o ticker, o período e a conexão e tente novamente.\n")
        snapshot = historical_snapshot(prices, ticker=args.ticker, period=args.period,
                                       retrieved_at=datetime.now(timezone.utc).isoformat())
        config = SimulationConfig(args.horizon, args.steps, args.n_sims,
                                  args.seed, args.confidence)
    report, paths = run_simulation(snapshot, config, historical_config,
                                   sensitivity_config)
    if company is not None:
        report["company_analysis"] = company
        if args.replay:
            print(format_company(company))
    elif args.replay:
        print("Este relatório antigo não contém indicadores da empresa.")
    terminal_mean = paths[-1].mean()
    terminal_std = paths[-1].std()
    print(format_simulation_summary(report, config, terminal_mean, terminal_std))
    destination = publish_run(report, paths, config, args.report, args.no_plot,
                               interactive=args.interactive)
    print(f"Relatório: {destination / 'report.json'}")
    for name in ("convergence.png", "paths.png", "final-distribution.png"):
        print(f"Figura: {destination / name}")
    if args.interactive:
        print(f"Figura interativa: {destination / 'paths.html'}")


def cli(argv: list[str] | None = None) -> None:
    """Show expected operational errors without an internal traceback."""
    try:
        main(argv)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}") from None


if __name__ == "__main__":
    cli()
