"""Frozen multi-asset study. Freeze once, then run/replay strictly offline."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
from forecasting.core import ForecastConfig, _metrics
from forecasting.report import run_experiment, _session_prices, plot_experiment
from forecasting.uncertainty import block_bootstrap
from simulation.report import historical_snapshot

ASSETS = ('SPY', 'AAPL', 'PETR4.SA')
MODELS = ('historical', 'ewma', 'ml')
PAIRS = (('ml', 'historical'), ('ml', 'ewma'), ('historical', 'ewma'))
PERIODS = {'full': ('2019-01-01', '2026-01-01'),
           '2019': ('2019-01-01', '2020-01-01'),
           '2020-2021': ('2020-01-01', '2022-01-01'),
           '2022-2023': ('2022-01-01', '2024-01-01'),
           '2024-2025': ('2024-01-01', '2026-01-01')}


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def git_identity():
    git = shutil.which('git')
    if git is None and Path('C:/Program Files/Git/cmd/git.exe').is_file():
        git = 'C:/Program Files/Git/cmd/git.exe'
    if git is None:
        raise RuntimeError('Git executable not found')
    def call(*args):
        return subprocess.check_output([git, *args], text=True).strip()
    return {'commit': call('rev-parse', 'HEAD'), 'status': call('status', '--porcelain'),
            'diff_sha256': sha256(subprocess.check_output([git, 'diff', 'HEAD'])).hexdigest()}


def freeze(destination):
    import yfinance as yf
    destination.mkdir(parents=True, exist_ok=False)
    protocol = Path('research/volatility-protocol-v1.md').read_bytes()
    (destination / 'protocol.md').write_bytes(protocol)
    manifest = {'protocol_sha256': sha256(protocol).hexdigest(), 'git': git_identity(),
                'versions': {m: version(m) for m in ('numpy', 'pandas', 'scipy', 'yfinance')},
                'assets': {}, 'failures': {}}
    options = dict(start='2010-01-01', end='2026-01-01', auto_adjust=True,
                   interval='1d', progress=False, threads=False)
    for ticker in ASSETS:
        try:
            data = yf.download(ticker, **options)
            prices = data['Close']
            if getattr(prices, 'ndim', 1) == 2:
                prices = prices[ticker]
            snapshot = historical_snapshot(prices, ticker=ticker, period='2010-01-01/2025-12-31',
                       retrieved_at=datetime.now(timezone.utc).isoformat())
            checked = _session_prices(snapshot)
            if checked.index.min().isoformat()[:10] < options['start'] or checked.index.max().isoformat()[:10] >= options['end']:
                raise ValueError('Provider dates outside requested range')
            gaps = checked.index.to_series().diff().dt.days
            name = ticker + '.json'
            write_new(destination / name, snapshot)
            manifest['assets'][ticker] = {'file': name, 'sha256': snapshot['sha256'],
                'file_sha256': sha256((destination / name).read_bytes()).hexdigest(),
                'first_date': snapshot['first_date'], 'last_date': snapshot['last_date'],
                'count': snapshot['count'], 'download_options': options,
                'calendar_gaps_over_7_days': [d.isoformat() for d in gaps[gaps > 7].index],
                'exclusions': [], 'duplicates': 0, 'invalid_prices': 0}
        except Exception as exc:
            manifest['failures'][ticker] = f'{type(exc).__name__}: {exc}'
    write_new(destination / 'manifest.json', manifest)
    return manifest


def summarize(rows):
    if not rows:
        return {'count': 0, 'warning': 'No observations'}
    columns, labels = [], []
    for a, b in PAIRS:
        for metric in ('mse', 'mae', 'quantile_loss'):
            # losses() is called immediately below, so late binding of
            # `metric` is safe; the closure never outlives this iteration.
            def losses(model):
                error = np.array([r['models'][model]['daily_variance'] - r['realized_squared_return'] for r in rows])
                return error**2 if metric == 'mse' else np.abs(error) if metric == 'mae' else np.array([r['models'][model]['quantile_loss'] for r in rows])  # noqa: B023
            columns.append(losses(a) - losses(b))
            labels.append(f'{a}-{b}:{metric}')
    for model in MODELS:
        columns.append([r['models'][model]['violation'] for r in rows])
        labels.append(f'{model}:violation_rate')
    values = np.array(columns).T
    intervals = {}
    for length in (5, 10, 20):
        intervals[str(length)] = block_bootstrap(values, block_length=length) if len(rows) >= length else {'warning': 'Sample shorter than block'}
    return {'count': len(rows), 'metrics': {m: _metrics(rows, m, .95) for m in MODELS},
            'bootstrap_labels': labels, 'bootstrap': intervals}


def compute(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if sha256((directory / 'protocol.md').read_bytes()).hexdigest() != manifest['protocol_sha256']:
        raise ValueError('Protocol checksum mismatch')
    results = {}
    for ticker, entry in manifest['assets'].items():
        if ticker not in ASSETS or entry.get('file') != ticker + '.json':
            raise ValueError('Invalid manifest asset identity or path')
        path = directory / entry['file']
        if sha256(path.read_bytes()).hexdigest() != entry['file_sha256']:
            raise ValueError('Snapshot file checksum mismatch')
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if snapshot.get('ticker') != ticker:
            raise ValueError('Snapshot ticker does not match manifest asset')
        if snapshot['sha256'] != entry['sha256']:
            raise ValueError('Manifest checksum mismatch')
        report = run_experiment(snapshot, ForecastConfig(validation_start='2016-01-01', test_start='2019-01-01'))
        rows = report['experiment']['test']['forecasts']
        results[ticker] = {'report': report, 'periods': {
            name: summarize([r for r in rows if start <= r['target_date'][:10] < end])
            for name, (start, end) in PERIODS.items()}}
    return results


def figures(directory, ticker, result):
    import matplotlib.pyplot as plt
    fig = plot_experiment(result['report'], directory / f'{ticker}-volatility.png')
    plt.close(fig)
    rows = result['report']['experiment']['test']['forecasts']
    import pandas as pd
    dates = pd.to_datetime([r['target_date'] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    for a, b in PAIRS:
        losses = [(r['models'][a]['daily_variance']-r['realized_squared_return'])**2 -
                  (r['models'][b]['daily_variance']-r['realized_squared_return'])**2 for r in rows]
        ax.plot(dates, np.cumsum(losses), label=f'{a}-{b}')
    ax.axhline(0, color='black', lw=.7)
    ax.set(xlabel='Target session', ylabel='Cumulative MSE loss difference (log return⁴)', title=ticker)
    ax.legend(); fig.savefig(directory / f'{ticker}-cumulative.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    y = 0
    for period, summary in result['periods'].items():
        ci = summary.get('bootstrap', {}).get('10', {})
        if 'lower' not in ci:
            continue
        for i in (0, 3, 6):
            mean, lo, hi = ci['mean'][i], ci['lower'][i], ci['upper'][i]
            ax.plot([lo, hi], [y, y], color='C0'); ax.plot(mean, y, 'o', color='C0')
            ax.text(0.01, y, f"{period} {summary['bootstrap_labels'][i]}", transform=ax.get_yaxis_transform(), fontsize=7)
            y += 1
    ax.axvline(0, color='black', lw=.8); ax.set(xlabel='Mean paired MSE difference (log return⁴), percentile 95% CI, L=10', yticks=[], title=ticker)
    fig.savefig(directory / f'{ticker}-intervals.png', dpi=140); plt.close(fig)


def markdown(results, manifest):
    text = ['# Frozen volatility experiment', '', 'See [protocol](../protocol.md) and [manifest](../manifest.json) for design, source, hashes and code identity.',
            'Validation scores select parameters; the following tables are final sequential test only.',
            'Negative paired differences favor the first model. No cross-asset currency aggregation.', '']
    if manifest['failures']:
        text += ['## Download limitations', json.dumps(manifest['failures']), '']
    if not results:
        text += ['Empirical experiment blocked: no valid real snapshots. No synthetic empirical results.', '']
    for ticker, result in results.items():
        experiment = result['report']['experiment']
        text += [f'## {ticker}', f"Selected only on validation: {experiment['selected_parameters']}",
                 '']
        for period, summary in result['periods'].items():
            if not summary['count']:
                text += [f'\n{period}: no observations']; continue
            for length, ci in summary['bootstrap'].items():
                if ci.get('warning'):
                    text += ['', f"Few blocks / limited sample (L={length}): {ci['warning']}", '']
            text += ['', '|Period|n|Model|MSE|MAE|Pinball (currency)|VaR violations / nominal 5%|Wilson 95%|', '|---|---:|---|---:|---:|---:|---|---|']
            metrics = summary['metrics']
            for model, m in metrics.items():
                text += [f"|{period}|{summary['count']}|{model}|{m['mse']:.7g}|{m['mae']:.7g}|{m['quantile_loss']:.7g}|{m['violation_count']} ({m['violation_rate']:.2%})|{m['wilson_95']}|"]
            winners = {metric: min(MODELS, key=lambda m: metrics[m][metric]) for metric in ('mse','mae','quantile_loss')}
            text += ['', f'{period}: lowest losses {winners}.', '', '|Pair (MSE)|Mean|Relative to second MSE|L=5 CI|L=10 CI|L=20 CI|', '|---|---:|---:|---|---|---|']
            for i, (a,b) in zip((0,3,6), PAIRS):
                fields=[]
                for length in ('5','10','20'):
                    ci=summary['bootstrap'][length]
                    fields.append(f"[{ci['lower'][i]:.6g}, {ci['upper'][i]:.6g}] " + ('crosses zero' if ci['lower'][i] <= 0 <= ci['upper'][i] else 'excludes zero') if 'lower' in ci else ci['warning'])
                difference=metrics[a]['mse']-metrics[b]['mse']
                relative=difference/metrics[b]['mse'] if metrics[b]['mse'] else None
                text += [f'|{a}-{b}|{difference:.6g}|{relative}|'+ '|'.join(fields)+'|']
            text += ['', 'All secondary paired intervals and violation-rate block intervals are in results.json.', '']
        text += [f'![volatility]({ticker}-volatility.png)', f'![paired intervals]({ticker}-intervals.png)', f'![cumulative losses]({ticker}-cumulative.png)', '']
    text += ['## Interpretation limits', 'Inspect period winners and cumulative losses for concentration; block sensitivity is shown without selecting favorable intervals. Conflicting metric winners imply different objectives. Smaller violation rates alone may reflect excessive conservatism.',
             'Pointwise intervals crossing zero are inconclusive, not equivalence. Wilson assumes independence; blocks allow local temporal dependence under approximate stationarity. No multiplicity correction. Conditional on frozen data, fitted models and protocol; not full selection, estimation, revision or future-regime uncertainty. Current adjusted prices are not authenticated point-in-time data. These three assets do not establish general superiority or profitability.']
    return '\n'.join(text)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','run','replay'))
    parser.add_argument('directory', type=Path)
    args=parser.parse_args(argv)
    if args.action == 'freeze':
        print(json.dumps(freeze(args.directory), indent=2)); return
    destination = args.directory / 'results'
    output = destination / 'results.json'
    if args.action == 'run':
        # Refuse legacy/partial artifacts too; never replace an existing figure.
        artifacts = ('results.json', 'execution.json', 'report.md')
        if destination.exists() or any((args.directory / name).exists() for name in artifacts) or any(args.directory.glob('*.png')):
            raise FileExistsError(f'Execution artifacts already exist: {args.directory}')
    results = compute(args.directory)
    if args.action == 'replay':
        saved = json.loads(output.read_text(encoding='utf-8'))
        environment_changes = {}
        for ticker in set(results) & set(saved):
            current = results[ticker]['report'].get('environment')
            previous = saved[ticker]['report'].get('environment')
            if current != previous:
                environment_changes[ticker] = {'saved': previous, 'current': current}
            # Environment versions are provenance, not scientific results.
            results[ticker]['report'].pop('environment', None)
            saved[ticker]['report'].pop('environment', None)
        if results != saved:
            raise ValueError('Offline replay differs')
        print('Offline replay verified')
        if environment_changes:
            print('Environment differences: ' + json.dumps(environment_changes, sort_keys=True))
        return
    staging = Path(tempfile.mkdtemp(prefix='.results-', dir=args.directory))
    try:
        write_new(staging / 'results.json', results)
        write_new(staging / 'execution.json', git_identity())
        manifest = json.loads((args.directory / 'manifest.json').read_text(encoding='utf-8'))
        with (staging / 'report.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(markdown(results, manifest))
        for ticker, result in results.items():
            figures(staging, ticker, result)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(output)

if __name__ == '__main__':
    main()
