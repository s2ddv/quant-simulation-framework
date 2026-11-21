"""Phase 3 metrics from backtest/forecasts.csv (no re-simulation)."""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from backtest.engine import MODELS, THRESHOLDS, LEVELS
from backtest.metrics import (brier_score, christoffersen_independence, effective_wilson,
                              es_residual_tests, kupiec_pof, reliability_table)
from validation.tail_probability import wilson_interval, clopper_pearson_interval

D = "research/tail-risk/backtest"
df = pd.read_csv(f"{D}/forecasts.csv", index_col=0, parse_dates=True)
H = 5
L = df["realized_loss"].to_numpy()
res = {"n_overlapping": len(df), "period": [str(df.index[0].date()), str(df.index[-1].date())]}
sub = {k: df.iloc[k::H] for k in range(H)}           # non-overlapping weekly samples
res["n_nonoverlap"] = [len(sub[k]) for k in range(H)]

def block_boot_diff(a, b, y, block=20, n=2000, seed=1):
    """Moving-block bootstrap CI of mean Brier difference (overlapping series)."""
    d = (a - y) ** 2 - (b - y) ** 2
    N = d.size; rng = np.random.default_rng(seed); nb = -(-N // block)
    m = []
    for _ in range(n):
        s = rng.integers(0, N - block + 1, nb)
        m.append(np.concatenate([d[i:i + block] for i in s])[:N].mean())
    return float(d.mean()), [float(x) for x in np.percentile(m, [2.5, 97.5])]

# ---- events ----
res["events"] = {}
for x in THRESHOLDS:
    key = f"p{int(100 * x)}"
    y = (L > x).astype(float)
    k, n = int(y.sum()), y.size
    ev = {"threshold": x, "observed_overlapping": k, "n": n, "freq": k / n,
          "ci_eff_n_wilson": effective_wilson(k, n, H),
          "nonoverlap_k": [int((sub[o]["realized_loss"] > x).sum()) for o in range(H)],
          "nonoverlap_cp": clopper_pearson_interval(int((sub[0]["realized_loss"] > x).sum()), len(sub[0])),
          "models": {}}
    for m in MODELS:
        p = df[f"{m}_{key}"].to_numpy()
        ev["models"][m] = {"mean_pred": float(p.mean()), "brier": brier_score(p, y),
                           "brier_nonoverlap0": brier_score(sub[0][f"{m}_{key}"].to_numpy(),
                                                            (sub[0]["realized_loss"] > x).to_numpy().astype(float))}
        if m != "gaussian":
            d, ci = block_boot_diff(p, df[f"gaussian_{key}"].to_numpy(), y)
            ev["models"][m]["brier_minus_gaussian"] = d; ev["models"][m]["brier_diff_ci"] = ci
    res["events"][key] = ev
# reliability of the 3% event (overlapping)
res["reliability_p3"] = {m: reliability_table(df[f"{m}_p3"].to_numpy(), (L > 0.03).astype(float),
                         [0, 0.005, 0.01, 0.02, 0.04, 1.0]) for m in MODELS}

# ---- VaR / ES ----
res["var_es"] = {}
for a in LEVELS:
    tag = int(100 * a)
    block = {}
    for m in MODELS:
        v = df[f"{m}_var{tag}"].to_numpy(); e = df[f"{m}_es{tag}"].to_numpy()
        viol = L > v
        pofs, inds = [], []
        for o in range(H):
            s = sub[o]; vv = (s["realized_loss"] > s[f"{m}_var{tag}"]).to_numpy()
            pofs.append(kupiec_pof(vv, a)["p_value"]); inds.append(christoffersen_independence(vv)["p_value"])
        s0 = sub[0]; v0 = (s0["realized_loss"] > s0[f"{m}_var{tag}"]).to_numpy()
        kp = kupiec_pof(v0, a)
        block[m] = {"mean_var": float(v.mean()), "mean_es": float(e.mean()),
                    "viol_rate_overlapping": float(viol.mean()),
                    "viol_rate_overlapping_ci": effective_wilson(int(viol.sum()), viol.size, H),
                    "nonoverlap0": {"n": kp["n"], "violations": kp["violations"], "rate": kp["rate"],
                                    "kupiec_p": kp["p_value"],
                                    "christoffersen": christoffersen_independence(v0)},
                    "kupiec_p_by_offset": pofs, "christoffersen_p_by_offset": inds,
                    "es_nonoverlap0": es_residual_tests(s0["realized_loss"].to_numpy(), s0[f"{m}_var{tag}"].to_numpy(),
                                                        s0[f"{m}_es{tag}"].to_numpy(), a, seed=11)}
    res["var_es"][str(tag)] = block
json.dump(res, open(f"{D}/metrics.json", "w"), indent=1, default=float)

f = lambda x: "%.3f%%" % (100 * x)
for key, ev in res["events"].items():
    print(key, "obs", ev["observed_overlapping"], "/", ev["n"], f(ev["freq"]), "CI(n/5)", [f(c) for c in ev["ci_eff_n_wilson"]],
          "nonov k", ev["nonoverlap_k"])
    for m, v in ev["models"].items():
        print("   ", m, "pred", f(v["mean_pred"]), "brier %.6f" % v["brier"], v.get("brier_diff_ci", ""))
for tag, b in res["var_es"].items():
    print("VaR/ES", tag)
    for m, v in b.items():
        n0 = v["nonoverlap0"]
        print("   ", m, "viol_ov", f(v["viol_rate_overlapping"]), "| nonov", n0["violations"], "/", n0["n"],
              "kupiec p %.3f" % n0["kupiec_p"], "christ p", n0["christoffersen"]["p_value"],
              "| ES z2 %.2f %s" % (v["es_nonoverlap0"]["z2"], np.round(v["es_nonoverlap0"]["z2_ci"], 2)),
              "meanVaR %.4f ES %.4f" % (v["mean_var"], v["mean_es"]))
