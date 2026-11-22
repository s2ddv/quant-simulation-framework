"""Ablation analysis: does the benefit of GARCH-t come from conditional variance or from fat tails?
Compares gaussian (static), student_t (static, fat), garch_n (dynamic, thin), garch_t (dynamic, fat) on the same origins."""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from backtest.metrics import (brier_difference, brier_score, christoffersen_independence, effective_wilson,
                              es_residual_tests, kupiec_pof, obs_over_exp_ci)

D = "research/tail-risk/backtest_gspc"
a = pd.read_csv(f"{D}/forecasts.csv", index_col=0, parse_dates=True)
b = pd.read_csv(f"{D}/forecasts_garch_n.csv", index_col=0, parse_dates=True)
assert a.index.equals(b.index) and (a["index"] == b["index"]).all(), "origins differ"
df = a.join(b.drop(columns="index"))
M = ["gaussian", "student_t", "garch_n", "garch_t"]
H, KEYS = 5, ["p3", "p5", "p8", "p8.6", "p10"]
L = df["realized_loss"].to_numpy()
res = {"n": len(df), "period": [str(df.index[0].date()), str(df.index[-1].date())], "events": {}, "var_es": {}, "regimes": {}}
pairs = [("garch_t", "garch_n"), ("garch_n", "gaussian"), ("garch_n", "student_t"), ("garch_t", "student_t")]

def ev_block(fr, key, x):
    y = (fr["realized_loss"].to_numpy() >= x).astype(float)
    out = {"observed": int(y.sum()), "n": int(y.size), "models": {}}
    for m in M:
        p = fr[f"{m}_{key}"].to_numpy()
        e = {"mean_pred": float(p.mean()), "brier": brier_score(p, y), "obs_over_exp": float(y.sum() / p.sum()) if p.sum() > 0 else None}
        if y.size >= 100: e["oe_ci"] = obs_over_exp_ci(y, p)
        out["models"][m] = e
    if y.size >= 100:
        out["brier_diff"] = {}
        for m1, m2 in pairs:
            d, ci = brier_difference(fr[f"{m1}_{key}"].to_numpy(), fr[f"{m2}_{key}"].to_numpy(), y)
            out["brier_diff"][f"{m1}-{m2}"] = {"diff": d, "ci": ci}
    return out

for k in KEYS:
    res["events"][k] = ev_block(df, k, float(k[1:]) / 100)

for al in (0.95, 0.99):
    t = int(100 * al); blk = {}
    for m in M:
        v, e = df[f"{m}_var{t}"].to_numpy(), df[f"{m}_es{t}"].to_numpy()
        viol = L > v
        pofs = [kupiec_pof((df.iloc[o::H]["realized_loss"] > df.iloc[o::H][f"{m}_var{t}"]).to_numpy(), al)["p_value"] for o in range(H)]
        inds = [christoffersen_independence((df.iloc[o::H]["realized_loss"] > df.iloc[o::H][f"{m}_var{t}"]).to_numpy())["p_value"] for o in range(H)]
        s0 = df.iloc[0::H]
        blk[m] = {"viol_rate": float(viol.mean()), "viol_ci": effective_wilson(int(viol.sum()), viol.size, H),
                  "kupiec_p_by_offset": pofs, "christoffersen_p_by_offset": inds,
                  "es": es_residual_tests(s0["realized_loss"].to_numpy(), s0[f"{m}_var{t}"].to_numpy(), s0[f"{m}_es{t}"].to_numpy(), al, seed=11),
                  "mean_var": float(v.mean()), "mean_es": float(e.mean())}
    res["var_es"][str(t)] = blk

masks = {"high_vol_at_origin": (df["trailing_vol21"] >= 0.20).to_numpy(), "low_vol_at_origin": (df["trailing_vol21"] < 0.20).to_numpy()}
cr = np.zeros(len(df), bool)
for lo, hi in (("2000-03-24", "2002-10-09"), ("2007-10-09", "2009-03-09"), ("2020-02-19", "2020-04-30"), ("2022-01-03", "2022-10-12")):
    cr |= (df.index >= lo) & (df.index <= hi)
masks["inside_crisis_windows"] = cr; masks["outside_crisis_windows"] = ~cr
for name, m in masks.items():
    fr = df[m]
    res["regimes"][name] = {"n": int(m.sum()), "events": {k: ev_block(fr, k, float(k[1:]) / 100) for k in ("p5", "p8.6")},
                            "viol99": {mm: float((fr["realized_loss"] > fr[f"{mm}_var99"]).mean()) for mm in M}}
json.dump(res, open(f"{D}/ablation_garch_normal.json", "w"), indent=1, default=float)

f = lambda x: "%.3f%%" % (100 * x)
print("n", res["n"], res["period"])
for k, ev in res["events"].items():
    print(k, "obs", ev["observed"], "/", ev["n"])
    for m, v in ev["models"].items():
        print("   %-9s pred %-8s O/E %-6s %s brier %.6f" % (m, f(v["mean_pred"]), round(v["obs_over_exp"], 2), np.round(v["oe_ci"], 2), v["brier"]))
    for pr, v in ev["brier_diff"].items():
        print("   Brier %-20s %+.5f CI [%+.5f, %+.5f]%s" % (pr, v["diff"], v["ci"][0], v["ci"][1], "  *" if v["ci"][1] < 0 or v["ci"][0] > 0 else ""))
for t, b_ in res["var_es"].items():
    print("VaR/ES", t)
    for m, v in b_.items():
        print("   %-9s viol %s %s | kupiec p by offset %s | christ %s | ES z2 %.2f %s" % (m, f(v["viol_rate"]), [f(c) for c in v["viol_ci"]],
              np.round(v["kupiec_p_by_offset"], 3), np.round(v["christoffersen_p_by_offset"], 2), v["es"]["z2"], np.round(v["es"]["z2_ci"], 2)))
for name, r in res["regimes"].items():
    print("REGIME", name, r["n"])
    for k, ev in r["events"].items():
        print("   ", k, "obs", ev["observed"], " ".join("%s O/E %s" % (m, round(v["obs_over_exp"], 2) if v["obs_over_exp"] is not None else None) for m, v in ev["models"].items()))
    print("    VaR99 viol:", {m: f(v) for m, v in r["viol99"].items()})
