"""Phase 3: run the expanding-window SPY backtest on the frozen snapshot (offline)."""
import json, sys, platform, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from forecasting.report import _session_prices
from backtest.engine import BacktestConfig, run_backtest

SNAP = "research/experiments/frozen-network-v1/SPY.json"
OUT = "research/tail-risk/backtest"
snap = json.load(open(SNAP))
prices = _session_prices(snap)                      # verifies sha256
rets = np.log(prices).diff().dropna()
t0 = time.time()
df, meta = run_backtest(rets, BacktestConfig())
meta["data"] = {k: snap[k] for k in ("ticker", "source", "interval", "price_field", "requested_period",
                                     "first_date", "last_date", "count", "sha256", "retrieved_at")}
meta["first_origin"], meta["last_origin"] = df.index[0].isoformat(), df.index[-1].isoformat()
meta["env"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
meta["runtime_s"] = round(time.time() - t0, 1)
df.to_csv(f"{OUT}/forecasts.csv")
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print({k: meta[k] for k in ("n_returns", "n_origins", "n_refits", "garch_nonconverged_refits",
                            "garch_previous_params_used", "first_origin", "last_origin", "runtime_s")})
