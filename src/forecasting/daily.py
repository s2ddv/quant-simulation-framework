"""Experimental one-session GBM integration; snapshots and replay are offline."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

from scipy.special import ndtri
import numpy as np

from forecasting.origin import forecast_at_origin
from forecasting.report import _session_prices
from simulation.gbm import simulate_gbm
from simulation.report import (SimulationConfig, terminal_mean_confidence_interval,
                               _digest)
from risk.statistics import risk_summary
from validation.statistics import terminal_theory, compare_terminal, convergence_analysis


def run_daily(snapshot, model, *, window=None, factor=None, selection=None,
              config=None):
    config = config or SimulationConfig(horizon=1/252, steps=1)
    if config.horizon != 1/252 or config.steps != 1:
        raise ValueError("Next-session mode requires horizon=1/252 year and steps=1.")
    forecast = forecast_at_origin(_session_prices(snapshot), model, window=window,
                                  factor=factor, selection=selection)
    sigma = forecast['annualized_volatility']
    mu = .5 * sigma**2
    s0 = forecast['origin_price']
    theory = terminal_theory(s0, mu, sigma, config.horizon)
    paths = simulate_gbm(s0, mu, sigma, config.horizon, 1, config.n_sims, config.seed)
    ci = terminal_mean_confidence_interval(paths[-1])
    ci['parameter_scope'] = 'conditional_on_fixed_predicted_variance; excludes forecast/model uncertainty'
    report = {
        'schema_version': 1, 'model': 'experimental_next_session_gbm',
        'historical_data': snapshot, 'forecast': forecast,
        'request': {'model': model, 'window': window, 'factor': factor, 'selection': selection},
        'last_price': s0, 'parameters': {'mu': mu, 'sigma': sigma},
        'simulation': asdict(config),
        'assumptions': {'log_return': 'Normal(0, predicted daily variance)',
                        'horizon_sessions': 1,
                        'drift': 'mu=sigma^2/2; zero log-return mean; not risk-neutral',
                        'forecast_status': 'Unscored next session after snapshot; no target observed',
                        'limitations': ['Conditional normality', 'Noisy squared-return variance proxy',
                                        'Forecast and calibration uncertainty excluded'],
                        'units': 'prices and losses: asset currency; mu: year^-1; sigma: year^-1/2'},
        'conventions': {'rng': 'NumPy default_rng/PCG64; shape (1,n_sims)',
                        'loss': 'S0-terminal; CVaR includes ties', 'quantiles': 'distribution percentiles, not estimator confidence intervals'},
        'results': {'terminal': compare_terminal(paths[-1], theory),
                    'terminal_mean_confidence_interval': ci,
                    'risk': risk_summary(paths, config.confidence),
                    'analytical_var': float(s0 - s0*np.exp(-np.sqrt(forecast['daily_variance'])*ndtri(config.confidence)))},
        'convergence': convergence_analysis(paths[-1], theory),
        'environment': {'python': platform.python_version(), **{
            name: version(name) for name in ('numpy','pandas','scipy','matplotlib')}},
    }
    json.dumps(report, allow_nan=False)
    return report, paths


def read_daily(path):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    if saved.get('schema_version') != 1 or saved.get('model') != 'experimental_next_session_gbm':
        raise ValueError('Unsupported daily report schema/model')
    expected, paths = run_daily(saved['historical_data'],
                               config=SimulationConfig(**saved['simulation']), **saved['request'])
    for key in expected:
        if key != 'environment' and expected[key] != saved.get(key):
            raise ValueError(f'Daily replay mismatch: {key}')
    return expected, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot', type=Path)
    source.add_argument('--replay', type=Path)
    parser.add_argument('--model', choices=('historical','ewma','ml'))
    parser.add_argument('--window', type=int)
    parser.add_argument('--factor', type=float)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--n-sims', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--horizon', type=float, default=1/252)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    from main import check_destination, publish_run
    check_destination(args.report)
    if args.replay:
        if any(x is not None for x in (args.model,args.window,args.factor,args.selection)):
            parser.error('Replay uses saved model and parameters')
        # Reject even explicit default overrides, rather than silently ignoring them.
        tokens = argv or []
        if any(t.split('=')[0] in ('--horizon','--steps','--n-sims','--seed','--confidence') for t in tokens):
            parser.error('Replay uses saved simulation configuration')
        report, paths = read_daily(args.replay)
    else:
        if args.model is None:
            parser.error('--model is required with --snapshot')
        snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
        snapshot = snapshot.get('historical_data', snapshot)
        selection = (json.loads(args.selection.read_text(encoding='utf-8'))
                     if args.selection else None)
        report, paths = run_daily(snapshot, args.model, window=args.window, factor=args.factor,
            selection=selection, config=SimulationConfig(args.horizon,args.steps,args.n_sims,args.seed,args.confidence))
    from forecasting.study import git_identity
    report['code'] = git_identity()
    root = Path(__file__).resolve().parents[2]
    report['code']['source_sha256'] = _digest({str(p.relative_to(root)): p.read_text(encoding='utf-8')
        for p in sorted((root/'src').rglob('*.py'))} | {'main.py': (root/'main.py').read_text(encoding='utf-8')})
    destination = publish_run(report, paths, SimulationConfig(**report['simulation']), args.report, args.no_plot)
    print(destination / 'report.json')


if __name__ == '__main__':
    main()
