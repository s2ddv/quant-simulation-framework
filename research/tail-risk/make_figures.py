"""Phase 4 figures. Inputs: backtest/{forecasts.csv,meta.json,metrics.json}, validation_results.json."""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

B, O, G, K = "#0072B2", "#D55E00", "#009E73", "#111111"      # Gaussian, Student-t, GARCH-t, observed
COL = {"gaussian": B, "student_t": O, "garch_t": G}
NAME = {"gaussian": "Gaussian GBM", "student_t": "Student-t (daily shocks)", "garch_t": "GARCH(1,1)-t"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100})
OUT = "research/tail-risk/figures"
meta = json.load(open("research/tail-risk/backtest/meta.json"))
met = json.load(open("research/tail-risk/backtest/metrics.json"))
fc = pd.read_csv("research/tail-risk/backtest/forecasts.csv", index_col=0, parse_dates=True)
val = json.load(open("research/tail-risk/validation_results.json"))
d = meta["data"]; per = f"{met['period'][0]} to {met['period'][1]}"
src = (f"Data: {d['ticker']} {d['price_field']} daily ({d['source']}, snapshot sha256 {d['sha256'][:10]}…, "
       f"{d['first_date'][:10]}–{d['last_date'][:10]}). 5-trading-day simple returns.")

def cap(fig, text):
    fig.text(0.01, 0.01, text, fontsize=7.5, color="#444", va="bottom", wrap=True)
    fig.subplots_adjust(bottom=0.2)

# ---------- Fig 1: SIMULATED loss-exceedance curves at equal variance ----------
g = meta["refits"][-1]["garch"]
unc_sd = np.sqrt(g["omega"] / (1 - g["alpha"] - g["beta"]))
rng = np.random.default_rng(20260101)
N, H, df_t = 200_000, 5, meta["refits"][-1]["t_df"]
mu_d = 0.0
def garch_paths(n):
    h = np.full(n, unc_sd**2); nu = g["nu"]; sc = np.sqrt((nu - 2) / nu); tot = np.zeros(n)
    for day in range(500 + H):                       # 500-day burn-in -> stationary mixture of h
        e = np.sqrt(h) * rng.standard_t(nu, n) * sc
        if day >= 500: tot += mu_d + e
        h = g["omega"] + g["alpha"] * e * e + g["beta"] * h
    return tot
xs = np.linspace(0.01, 0.14, 60)
cum = {"gaussian": rng.normal(0, unc_sd * np.sqrt(H), N),
       "student_t": (rng.standard_t(4, (N, H)) * np.sqrt(2 / 4)).sum(1) * unc_sd,
       "garch_t": garch_paths(N)}
fig, ax = plt.subplots(figsize=(8, 5.4))
for k, c in cum.items():
    loss = -np.expm1(c)
    ax.plot(xs * 100, [np.mean(loss > x) for x in xs], color=COL[k], lw=2,
            label={"student_t": "Student-t df=4, daily shocks summed", **NAME}[k])
wk = stats.t.sf(-np.log1p(-xs) / (unc_sd * np.sqrt(H)) * np.sqrt(2), 4)
ax.plot(xs * 100, wk, color="#777", ls="--", lw=1.5, label="Student-t df=4 on the weekly return (contrast)")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlabel("Loss threshold x (%)"); ax.set_ylabel("P(5-day loss > x)")
ax.set_title("Simulated tail exceedance at equal variance: Mediocristan vs. heavier tails", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
cap(fig, f"SIMULATED (not historical, not a forecast). Daily σ = {unc_sd*100:.2f}% for all models (≈{unc_sd*np.sqrt(252)*100:.1f}% annual, the GARCH "
        f"unconditional level from the last fit), μ=0, 200,000 paths, seed 20260101. GARCH-t: α={g['alpha']:.3f}, β={g['beta']:.3f}, ν={g['nu']:.2f}, "
        f"500-day burn-in. Monte Carlo floor ≈ 5e-6 (1/N); curves near it are noisy. The weekly-t line is a different model, shown only to contrast aggregation.")
fig.savefig(f"{OUT}/fig1_simulated_exceedance.png", dpi=200); plt.close(fig)

# ---------- Fig 2: HISTORICAL observed vs OUT-OF-SAMPLE predicted ----------
fig, ax = plt.subplots(figsize=(8, 5.4))
keys = ["p3", "p5", "p8", "p10"]; xl = np.arange(4)
obs = [met["events"][k]["freq"] for k in keys]
lo = [met["events"][k]["ci_eff_n_wilson"][0] for k in keys]; hi = [met["events"][k]["ci_eff_n_wilson"][1] for k in keys]
for j, m in enumerate(COL):
    ax.plot(xl + (j - 1) * 0.12, [met["events"][k]["models"][m]["mean_pred"] for k in keys], "o", ms=9, color=COL[m], label=NAME[m] + " (OOS forecast, mean)")
ax.errorbar(xl + 0.3, obs, yerr=[np.array(obs) - lo, np.array(hi) - obs], fmt="s", color=K, ms=8, capsize=4,
           label="Historical frequency (95% CI, n/5 effective)")
floor = 1e-5
ax.set_yscale("log"); ax.set_ylim(1e-5, 0.2)
ax.set_xticks(xl + 0.1); ax.set_xticklabels(["> 3%", "> 5%", "> 8%", "> 10%"]); ax.set_xlabel("5-day loss threshold")
ax.set_ylabel("Probability / frequency (log scale)")
ax.set_title("Historical frequency vs. out-of-sample predicted probability", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
n = met["n_overlapping"]; ev = [met["events"][k]["observed_overlapping"] for k in keys]
cap(fig, f"HISTORICAL OBSERVATIONS vs OUT-OF-SAMPLE FORECASTS, origins {per} ({n} overlapping 5-day windows; events: {', '.join(map(str, ev))}). "
        f"Expanding window (min 1260 days), refit every 21 days; forecasts use only past data. Overlap inflates CI dependence, so n/5 effective windows are used "
        f"(approximation). Counts of >8% and >10% events are very small: intervals are wide. {src}")
fig.savefig(f"{OUT}/fig2_observed_vs_predicted.png", dpi=200); plt.close(fig)

# ---------- Fig 3: VaR and ES ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 5.2), sharey=False)
Lr = fc["realized_loss"].to_numpy()
for ax, tag in zip(axs, ("95", "99")):
    a = int(tag) / 100; w = 0.25
    for j, m in enumerate(COL):
        v = met["var_es"][tag][m]
        ax.bar(np.array([0, 1]) + (j - 1) * w, [v["mean_var"] * 100, v["mean_es"] * 100], w * 0.9, color=COL[m], label=NAME[m])
    ev, ee = np.quantile(Lr, a), Lr[Lr >= np.quantile(Lr, a)].mean()
    for x0, y0 in ((0, ev), (1, ee)):
        ax.hlines(y0 * 100, x0 - 0.45, x0 + 0.45, color=K, lw=2, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)           # headroom so the legend never overlaps data
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"VaR {tag}%", f"ES {tag}%"]); ax.set_ylabel("5-day loss (% of price)")
    ax.set_title(f"{tag}% level", loc="left", fontweight="bold")
axs[0].plot([], [], color=K, ls="--", lw=2, label="Realized (unconditional, OOS windows)")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Forecast VaR and Expected Shortfall vs. realized tail losses", x=0.01, ha="left", fontweight="bold")
vs = {t: {m: met["var_es"][t][m]["nonoverlap0"] for m in COL} for t in ("95", "99")}
cap(fig, f"OUT-OF-SAMPLE forecasts averaged over origins ({per}) vs HISTORICAL realized loss quantile/tail mean over the same windows (a single unconditional benchmark; "
         f"averaging conditional forecasts is not a coverage test - see Kupiec/Christoffersen tables in the report). "
         f"VaR violation counts (non-overlapping weekly, n={vs['99']['gaussian']['n']}): 99% G/t/GARCH = "
         f"{vs['99']['gaussian']['violations']}/{vs['99']['student_t']['violations']}/{vs['99']['garch_t']['violations']} (expected 5.5). {src}")
fig.savefig(f"{OUT}/fig3_var_es.png", dpi=200); plt.close(fig)

# ---------- Fig 4: Monte Carlo convergence and uncertainty ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))
ax = axs[0]
cv = val["convergence"]; ns = [r["params"]["n_sims"] for r in cv]; p = np.array([r["probability"] for r in cv]) * 100
lo = np.array([r["wilson"][0] for r in cv]) * 100; hi = np.array([r["wilson"][1] for r in cv]) * 100
ax.errorbar(ns, p, yerr=[p - lo, hi - p], fmt="o", color=B, capsize=4, label="MC estimate, 95% Wilson CI")
ax.axhline(val["analytic_normal_5pct"] * 100, color=K, ls="--", label="Exact normal value")
ax.set_xscale("log"); ax.set_xlabel("Simulations n"); ax.set_ylabel("P(5-day loss > 5%) (%)")
ax.set_title("Normal GBM: convergence (σ=20%)", loc="left", fontweight="bold"); ax.legend(frameon=False, fontsize=8)
ax = axs[1]
sd = val["seeds"]; pp = np.array([r["probability"] for r in sd]) * 100
lo = np.array([r["wilson"][0] for r in sd]) * 100; hi = np.array([r["wilson"][1] for r in sd]) * 100
ax.errorbar(range(len(sd)), pp, yerr=[pp - lo, hi - pp], fmt="o", color=O, capsize=3, label="Student-t df=4: 10 seeds, 95% Wilson CI")
ax.axhline(pp.mean(), color="#777", ls=":", label=f"Mean over seeds {pp.mean():.3f}%")
ax.set_xlabel("Seed"); ax.set_ylabel("P(5-day loss > 8.6%) (%)"); ax.set_title("Student-t: seed sensitivity (1e6 paths)", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8)
fig.suptitle("Monte Carlo uncertainty (simulated, not historical)", x=0.01, ha="left", fontweight="bold")
cap(fig, "SIMULATED. μ=5%, σ=20% (σ_week from T=5/252), 5 daily steps. Intervals quantify Monte Carlo sampling error of a binomial proportion only "
         "(Wilson); they exclude model, calibration and parameter uncertainty. No moment-based standard errors are used for t-shock prices (infinite mean). "
         "Left: all sizes use seed 7 and batches of 1e5, so larger samples contain the smaller ones (correlated, one realization each); not a convergence-rate guarantee.")
fig.savefig(f"{OUT}/fig4_mc_uncertainty.png", dpi=200); plt.close(fig)
print("ok", unc_sd * np.sqrt(252))
