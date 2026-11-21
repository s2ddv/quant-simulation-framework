"""Download ^GSPC daily closes (price index, no dividends) once and freeze a checksummed snapshot.
Network is needed only here; every later step is offline. Run once; do not overwrite an existing snapshot."""
import json, sys, datetime as dt
from pathlib import Path
sys.path.insert(0, "src")
import pandas as pd, yfinance as yf
from simulation.report import historical_snapshot
from calibration.prices import validate_prices

OUT = Path("research/experiments/sp500-gspc-1990/GSPC.json")
if OUT.exists():
    sys.exit(f"{OUT} exists; refusing to overwrite")
START, END = "1990-01-01", (dt.date.today() + dt.timedelta(days=1)).isoformat()
raw = yf.download("^GSPC", start=START, end=END, auto_adjust=False, progress=False, group_by="column")
close = raw["Close"]["^GSPC"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
close = close.dropna()
close.index = pd.DatetimeIndex(close.index).tz_localize(None)
prices = validate_prices(close, min_observations=3, require_dates=True)
snap = historical_snapshot(prices, ticker="^GSPC", period=f"{START}/{END}",
                           retrieved_at=pd.Timestamp.now(tz="UTC").isoformat())
OUT.write_text(json.dumps(snap))
gaps = prices.index.to_series().diff().dt.days
print({"count": len(prices), "first": str(prices.index[0].date()), "last": str(prices.index[-1].date()),
       "max_gap_days": int(gaps.max()), "gaps_gt7": [str(d.date()) for d in gaps[gaps > 7].index]})
