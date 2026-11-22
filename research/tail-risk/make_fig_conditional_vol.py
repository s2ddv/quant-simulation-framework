"""Fig 7: effect of conditional volatility (2x2 ablation: static/dynamic variance x thin/fat shocks)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = "research/tail-risk/backtest_gspc", "research/tail-risk/figures_gspc"
ab = json.load(open(f"{D}/ablation_garch_normal.json"))
a = pd.read_csv(f"{D}/forecasts.csv", index_col=0, parse_dates=True)
b = pd.read_csv(f"{D}/forecasts_garch_n.csv", index_col=0, parse_dates=True)
df = a.join(b.drop(columns="index"))
meta = json.load(open(f"{D}/meta.json")); d = meta["data"]
COL = {"gaussian": "#0072B2", "student_t": "#D55E00", "garch_n": "#CC79A7", "garch_t": "#009E73"}
NAME = {"gaussian": "Static variance, normal shocks (Gaussian GBM)", "student_t": "Static variance, Student-t shocks",
        "garch_n": "Time-varying variance, normal shocks (GARCH-normal)", "garch_t": "Time-varying variance, Student-t shocks (GARCH-t)"}
MK = {"gaussian": "s", "student_t": "s", "garch_n": "o", "garch_t": "o"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})

# ---- A: calibration ratio observed/expected, 3 extreme thresholds ----
ax = axs[0]; keys = ["p5", "p8.6", "p10"]; x = np.arange(3)
for j, m in enumerate(COL):
    ev = [ab["events"][k]["models"][m] for k in keys]
    y = np.array([e["obs_over_exp"] for e in ev]); lo = np.array([e["oe_ci"][0] for e in ev]); hi = np.array([e["oe_ci"][1] for e in ev])
    ax.errorbar(x + (j - 1.5) * .13, y, yerr=[y - lo, hi - y], fmt=MK[m], color=COL[m], ms=8, capsize=3, label=NAME[m],
                mfc=COL[m] if m in ("gaussian", "garch_n") else "white", mew=1.8)
ax.axhline(1, color="#111", ls="--", lw=1.2, label="Perfectly calibrated (O/E = 1)")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["loss ≥ 5%", "loss ≥ 8.6%", "loss ≥ 10%"])
ax.set_ylabel("Observed / expected number of 5-day losses (log)")
ax.set_title("A. Calibration: who gets the tail frequency right?", loc="left", fontweight="bold", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.6, loc="upper left")

# ---- B: VaR 99% through the 2007-09 crisis ----
ax = axs[1]; w = df.loc["2007-01-01":"2009-12-31"]
ax.plot(w.index, w["realized_loss"] * 100, color="#bbb", lw=.8, label="Realized 5-day loss (historical)")
for m in ("gaussian", "garch_n", "garch_t"):
    ax.plot(w.index, w[f"{m}_var99"] * 100, color=COL[m], lw=1.5 if m != "garch_t" else 1.0, label="VaR 99% — " + {"gaussian": "Gaussian (static)", "garch_n": "GARCH-normal", "garch_t": "GARCH-t"}[m],
            ls="-" if m != "garch_t" else "--")
v = {m: float((w["realized_loss"] > w[f"{m}_var99"]).mean() * 100) for m in ("gaussian", "garch_n", "garch_t")}
ax.axhline(0, color="#444", lw=.6); ax.set_ylim(-8, 22); ax.set_ylabel("5-day loss (% of price)")
ax.set_title("B. 2007–2009: a fixed variance vs. one that reacts", loc="left", fontweight="bold", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.text(0.99, 0.97, "share of windows with loss > VaR 99% (expected 1%):\n" + "\n".join(f"{ {'gaussian': 'Gaussian', 'garch_n': 'GARCH-normal', 'garch_t': 'GARCH-t'}[k]}: {x:.1f}%" for k, x in v.items()),
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#333")
fig.suptitle("Most of the improvement comes from letting variance change over time (circles); fat-tailed shocks add little",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, f"HISTORICAL outcomes vs OUT-OF-SAMPLE forecasts, origins {ab['period'][0]} to {ab['period'][1]} ({ab['n']} overlapping 5-day windows). Expanding window (≥5 years), refit every 21 sessions, "
         f"same data, seed and draws for all models. Panel A: O/E with 95% moving-block bootstrap CI (windows overlap; counts: ≥5%: {ab['events']['p5']['observed']}, ≥8.6%: {ab['events']['p8.6']['observed']}, ≥10%: {ab['events']['p10']['observed']} windows, "
         f"in few independent episodes). Panel B: the 2007–09 window (≈{len(w)} windows) is one crisis, not a statistical test; y-axis clipped at 22% (GARCH VaR and realized loss exceed it briefly in Oct–Nov 2008). Data: ^GSPC close, price index (Yahoo Finance, snapshot sha256 {d['sha256'][:10]}…).",
         fontsize=7.3, color="#444", va="bottom", wrap=True)
fig.subplots_adjust(bottom=0.2, top=0.88, left=0.07, right=0.985, wspace=0.18)
fig.savefig(f"{OUT}/fig7_conditional_volatility.png", dpi=200)
print(v, len(w))
