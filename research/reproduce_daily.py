"""Reproduce the unscored AAPL daily example offline into a NEW directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))

from forecasting.core import ForecastConfig
from forecasting.origin import make_selection
from forecasting.report import _session_prices
from forecasting.daily import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args = parser.parse_args()
    snapshot_path = ROOT/'research/experiments/frozen-network-v1/AAPL.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    artifact = make_selection(_session_prices(snapshot),
        ForecastConfig(validation_start='2016-01-01',test_start='2019-01-01'),
        source='research/volatility-protocol-v1.md; frozen-network-v1/AAPL.json; validation only')
    args.destination.mkdir(parents=True,exist_ok=False)
    selection_path = args.destination/'selection.json'
    selection_path.write_text(json.dumps(artifact,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    main(['--snapshot',str(snapshot_path),'--model','ml','--selection',str(selection_path),
          '--n-sims','100000','--seed','42','--no-plot','--report',str(args.destination/'forecast.json')])
    main(['--replay',str(args.destination/'forecast/report.json'),'--no-plot',
          '--report',str(args.destination/'replay.json')])
