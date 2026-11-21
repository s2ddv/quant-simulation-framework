"""Phase 2 experiments. Seeds are fixed; every row stores its parameters.

Event: 5-trading-day simple return <= -8.6%, mu=0.05, T=5/252, 5 daily steps.
"""
import json, sys
sys.path.insert(0, "src")
from validation.tail_probability import estimate_loss_probability as est, normal_loss_probability as nlp

MU, T, LOSS = 0.05, 5 / 252, 0.086
out = {"event": "R_5d <= -8.6% (simple)", "mu": MU, "T": T, "grid": [], "convergence": [], "seeds": []}
for sigma in (0.10, 0.15, 0.20, 0.30):
    for df in (None, 8, 5, 4, 3):
        r = est(LOSS, mu=MU, sigma=sigma, horizon=T, steps=5, n_sims=1_000_000, seed=20260101,
                dist_type="normal" if df is None else "t_student", df=df, level=0.95)
        r["analytic_normal"] = nlp(LOSS, MU, sigma, T)
        out["grid"].append(r)
for n in (10_000, 100_000, 1_000_000, 4_000_000):
    out["convergence"].append(est(0.05, mu=MU, sigma=0.20, horizon=T, steps=5, n_sims=n, seed=7))
out["analytic_normal_5pct"] = nlp(0.05, MU, 0.20, T)
for seed in range(10):
    out["seeds"].append(est(LOSS, mu=MU, sigma=0.20, horizon=T, steps=5, n_sims=1_000_000,
                            seed=seed, dist_type="t_student", df=4))
json.dump(out, open("research/tail-risk/validation_results.json", "w"), indent=1)
print("grid (sigma, df, p%, wilson%, analytic normal%)")
for r in out["grid"]:
    p = r["params"]; w = r["wilson"]
    print(p["sigma"], p["df"], "%.4f" % (100 * r["probability"]), "[%.4f, %.4f]" % (100 * w[0], 100 * w[1]),
          "%.4f" % (100 * r["analytic_normal"]))
print("convergence", [(r["params"]["n_sims"], r["hits"], [round(100*x,4) for x in r["wilson"]]) for r in out["convergence"]],
      round(100*out["analytic_normal_5pct"],4))
ps = [100 * r["probability"] for r in out["seeds"]]
print("seeds t4 p%:", [round(x, 4) for x in ps], "min", min(ps), "max", max(ps))
