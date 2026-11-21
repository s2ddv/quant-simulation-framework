"""Ablation: GARCH(1,1) with NORMAL innovations on the same GSPC origins/schedule/seed as the main backtest."""
import json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from forecasting.report import _session_prices
from backtest.engine import BacktestConfig, run_garch_normal_backtest

SNAP = "research/experiments/sp500-gspc-1990/GSPC.json"
OUT = "research/tail-risk/backtest_gspc"
prices = _session_prices(json.load(open(SNAP)))
rets = np.log(prices).diff().dropna()
t0 = time.time()
df, meta = run_garch_normal_backtest(rets, BacktestConfig())
meta["snapshot"] = SNAP; meta["runtime_s"] = round(time.time() - t0, 1)
df.to_csv(f"{OUT}/forecasts_garch_n.csv"); json.dump(meta, open(f"{OUT}/meta_garch_n.json", "w"), indent=1)
print({k: meta[k] for k in ("n_origins", "n_refits", "nonconverged_refits", "previous_params_used", "runtime_s")})
