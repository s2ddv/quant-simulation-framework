"""Re-create the original -8.6% weekly-drop comparison with the audited simulator.
Simulated only: 1,000,000 paths, 5 daily steps, sigma=20%, mu=5%, seed 20260101."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from validation.tail_probability import estimate_loss_probability as est, normal_loss_probability as nlp

MU, SIG, T, LOSS, SEED = 0.05, 0.20, 5 / 252, 0.086, 20260101
kw = dict(mu=MU, sigma=SIG, horizon=T, steps=5, n_sims=1_000_000, seed=SEED)
n = est(LOSS, **kw)
t = est(LOSS, dist_type="t_student", df=4, **kw)
weekly = float(stats.t.cdf((np.log1p(-LOSS) - (MU - SIG**2 / 2) * T) / (SIG * np.sqrt(T)) * np.sqrt(2), 4))
rows = [("Normal GBM", n["probability"], n["wilson"], "#0072B2"),
        ("Student-t df=4\n(daily shocks, summed)", t["probability"], t["wilson"], "#D55E00")]
fig, ax = plt.subplots(figsize=(7.5, 5))
for i, (name, p, w, c) in enumerate(rows):
    ax.bar(i, p * 100, 0.55, color=c)
    ax.errorbar(i, p * 100, yerr=[[p * 100 - w[0] * 100], [w[1] * 100 - p * 100]], color="#111", capsize=5)
    ax.text(i, w[1] * 100 + 0.01, f"{p*100:.3f}%\n[{w[0]*100:.3f}; {w[1]*100:.3f}]", ha="center", fontsize=9)
ax.axhline(weekly * 100, color="#777", ls="--")
ax.text(1.45, weekly * 100 + 0.01, f"t(4) on the weekly return: {weekly*100:.2f}% (different model)", ha="right", fontsize=8, color="#444")
ax.set_xticks([0, 1]); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 0.65)
ax.set_ylabel("P(5-day simple return ≤ −8.6%) (%)")
ax.set_title("−8.6% weekly drop: Normal vs Student-t (SIMULATED)", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, f"SIMULATED, not historical. 1,000,000 paths, 5 daily steps, σ=20%/yr, μ=5%/yr (inferred; original μ unknown), unit-variance t shocks, seed {SEED}.\n"
         f"Bars: Monte Carlo estimate with 95% Wilson CI (error of sampling only). Exact normal value: {nlp(LOSS, MU, SIG, T)*100:.4f}%.\n"
         "Original figure: 0.06% / 0.28%. Reproduced in magnitude; sensitive to μ (see audit.md).", fontsize=7.5, color="#444", va="bottom")
fig.subplots_adjust(bottom=0.22)
fig.savefig("research/tail-risk/figures/fig0_original_8p6_reproduction.png", dpi=200)
print(json.dumps({"normal": [n["hits"], n["probability"], n["wilson"]], "t4_daily": [t["hits"], t["probability"], t["wilson"]], "t4_weekly": weekly,
                  "normal_exact": nlp(LOSS, MU, SIG, T)}, indent=1))
