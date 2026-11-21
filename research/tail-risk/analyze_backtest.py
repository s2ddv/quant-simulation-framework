"""Phase 3 metrics. usage: analyze_backtest.py [bt_dir] [snapshot.json]
Reads bt_dir/forecasts.csv (+ snapshot for the full-sample historical frequency); writes bt_dir/metrics.json.
Regime rules were fixed BEFORE looking at results (see report): ex-ante high-vol = trailing 21d realized vol >= 20% (annualized);
calendar crisis windows are ex-post labels (peak-to-trough dates) and are secondary."""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from backtest.engine import MODELS, LEVELS, VOL_WINDOW
from backtest.metrics import (brier_score, christoffersen_independence, effective_wilson,
                              es_residual_tests, kupiec_pof, reliability_table)
from validation.tail_probability import clopper_pearson_interval

D = sys.argv[1] if len(sys.argv) > 1 else "research/tail-risk/backtest"
SNAP = sys.argv[2] if len(sys.argv) > 2 else None
H, VOL_CUT = 5, 0.20
CRISES = {"dotcom_bear 2000-03-24..2002-10-09": ("2000-03-24", "2002-10-09"),
          "gfc 2007-10-09..2009-03-09": ("2007-10-09", "2009-03-09"),
          "covid 2020-02-19..2020-04-30": ("2020-02-19", "2020-04-30"),
          "bear_2022 2022-01-03..2022-10-12": ("2022-01-03", "2022-10-12")}
df = pd.read_csv(f"{D}/forecasts.csv", index_col=0, parse_dates=True)
L = df["realized_loss"].to_numpy()
keys = sorted({c.split("_", 2)[-1] if c.startswith("student_t") or c.startswith("garch_t") else c.split("_", 1)[1]
               for c in df.columns if c.split("_")[-1].startswith("p") and c.split("_")[-1][1:2].isdigit()},
              key=lambda k: float(k[1:]))
thr = {k: float(k[1:]) / 100 for k in keys}
res = {"dir": D, "n_overlapping": len(df), "period": [str(df.index[0].date()), str(df.index[-1].date())], "thresholds": thr}
sub = {k: df.iloc[k::H] for k in range(H)}
res["n_nonoverlap"] = [len(sub[k]) for k in range(H)]

def episodes(dates, gap=10):
    """Cluster event origins closer than `gap` trading rows into one episode."""
    idx = np.asarray(dates); out = []
    for i in idx:
        if not out or i - out[-1][-1] > gap: out.append([i])
        else: out[-1].append(i)
    return out

def mbb(fn, n, block=20, reps=2000, seed=1):
    rng = np.random.default_rng(seed); nb = -(-n // block); out = []
    for _ in range(reps):
        s = rng.integers(0, max(1, n - block + 1), nb)
        out.append(fn(np.concatenate([np.arange(i, min(i + block, n)) for i in s])[:n]))
    return out

def ratio_ci(y, p):
    b = mbb(lambda i: y[i].sum() / p[i].sum() if p[i].sum() > 0 else np.nan, y.size)
    return [float(x) for x in np.nanpercentile(b, [2.5, 97.5])]

def diff_ci(a, b, y):
    d = (a - y) ** 2 - (b - y) ** 2
    m = mbb(lambda i: d[i].mean(), d.size)
    return float(d.mean()), [float(x) for x in np.percentile(m, [2.5, 97.5])]

# ---- Q1: historical frequency of 5-day loss >= x (no model involved) ----
hist = {}
if SNAP:
    from forecasting.report import _session_prices
    pr = _session_prices(json.load(open(SNAP)))
    lr = np.log(pr).diff().dropna()
    cs = np.concatenate([[0.0], np.cumsum(lr.to_numpy())])
    full = -np.expm1(cs[H:] - cs[:-H])                       # all overlapping windows over the whole sample
    fdates = lr.index[H - 1:]                                  # window end dates
    for tag, loss in (("full_sample", full), ("oos_origins", L)):
        for x in sorted(set([0.05, 0.086]) | set(thr.values())):
            y = loss >= x; k, n = int(y.sum()), y.size
            ev = np.flatnonzero(y)
            hist.setdefault(tag, {})[f"{x:g}"] = {
                "k_overlapping": k, "n": n, "freq": k / n, "ci_wilson_n_over_5": effective_wilson(k, n, H),
                "k_nonoverlap_by_offset": [int((loss[o::H] >= x).sum()) for o in range(H)],
                "n_nonoverlap": int(len(loss[0::H])),
                "cp_offset0": clopper_pearson_interval(int((loss[0::H] >= x).sum()), len(loss[0::H])),
                "episodes": len(episodes(ev)),
                "episode_starts": [str((fdates if tag == "full_sample" else df.index)[e[0]].date()) for e in episodes(ev)][:40]}
    hist["full_sample_period"] = [str(lr.index[0].date()), str(lr.index[-1].date())]
res["historical_frequency"] = hist

# ---- events: observed vs predicted, calibration ----
def event_block(frame, key, x):
    y = (frame["realized_loss"].to_numpy() >= x).astype(float)
    ev = {"threshold": x, "observed": int(y.sum()), "n": int(y.size), "freq": float(y.mean()), "models": {}}
    if y.size: ev["ci_n_over_5"] = effective_wilson(int(y.sum()), y.size, H)
    for m in MODELS:
        p = frame[f"{m}_{key}"].to_numpy()
        e = {"mean_pred": float(p.mean()), "expected_count": float(p.sum()), "brier": brier_score(p, y)}
        e["obs_over_exp"] = float(y.sum() / p.sum()) if p.sum() > 0 else None
        if y.size >= 100: e["obs_over_exp_ci"] = ratio_ci(y, p)
        if m != "gaussian" and y.size >= 100:
            e["brier_minus_gaussian"], e["brier_diff_ci"] = diff_ci(p, frame[f"gaussian_{key}"].to_numpy(), y)
        ev["models"][m] = e
    return ev

res["events"] = {k: event_block(df, k, thr[k]) for k in keys}
res["reliability_p5"] = {m: reliability_table(df[f"{m}_p5"].to_numpy(), (L >= 0.05).astype(float),
                         [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]) for m in MODELS if "p5" in keys}

# ---- VaR / ES (non-overlapping tests) ----
def var_es_block(frame, a, with_tests=True):
    tag = int(100 * a); out = {}
    Lr = frame["realized_loss"].to_numpy()
    for m in MODELS:
        v, e = frame[f"{m}_var{tag}"].to_numpy(), frame[f"{m}_es{tag}"].to_numpy()
        viol = Lr > v
        o = {"mean_var": float(v.mean()), "mean_es": float(e.mean()), "viol_rate_overlapping": float(viol.mean()),
             "viol_ci_n_over_5": effective_wilson(int(viol.sum()), viol.size, H)}
        if with_tests:
            pofs, inds = [], []
            for off in range(H):
                s = frame.iloc[off::H]; vv = (s["realized_loss"] > s[f"{m}_var{tag}"]).to_numpy()
                pofs.append(kupiec_pof(vv, a)["p_value"]); inds.append(christoffersen_independence(vv)["p_value"])
            s0 = frame.iloc[0::H]; v0 = (s0["realized_loss"] > s0[f"{m}_var{tag}"]).to_numpy(); kp = kupiec_pof(v0, a)
            o.update(nonoverlap0={"n": kp["n"], "violations": kp["violations"], "rate": kp["rate"], "kupiec_p": kp["p_value"],
                                  "christoffersen": christoffersen_independence(v0)},
                     kupiec_p_by_offset=pofs, christoffersen_p_by_offset=inds,
                     es_nonoverlap0=es_residual_tests(s0["realized_loss"].to_numpy(), s0[f"{m}_var{tag}"].to_numpy(),
                                                      s0[f"{m}_es{tag}"].to_numpy(), a, seed=11))
        out[m] = o
    return out
res["var_es"] = {str(int(100 * a)): var_es_block(df, a) for a in LEVELS}

# ---- Q4: regimes ----
reg = {}
masks = {"ex_ante_high_vol (vol21>=20%)": (df["trailing_vol21"] >= VOL_CUT).to_numpy(),
         "ex_ante_low_vol (vol21<20%)": (df["trailing_vol21"] < VOL_CUT).to_numpy()} if "trailing_vol21" in df else {}
any_crisis = np.zeros(len(df), bool)
for name, (a, b) in CRISES.items():
    m = ((df.index >= a) & (df.index <= b)); any_crisis |= m
    if m.any(): masks[f"crisis(ex-post) {name}"] = m
masks["ex_post_all_crisis_windows"] = any_crisis
masks["ex_post_outside_crisis_windows"] = ~any_crisis
for name, m in masks.items():
    fr = df[m]
    if len(fr) < 50: continue
    reg[name] = {"n": int(len(fr)), "share": float(m.mean()),
                 "events": {k: event_block(fr, k, thr[k]) for k in keys if k in ("p5", "p8.6", "p10") or k == "p3"},
                 "var_es": {str(int(100 * a)): var_es_block(fr, a, with_tests=False) for a in LEVELS},
                 "kupiec99_nonoverlap0": {mm: kupiec_pof((fr.iloc[0::H]["realized_loss"] > fr.iloc[0::H][f"{mm}_var99"]).to_numpy(), 0.99)
                                           for mm in MODELS} if len(fr.iloc[0::H]) >= 20 else None}
res["regimes"] = reg
json.dump(res, open(f"{D}/metrics.json", "w"), indent=1, default=float)

f = lambda x: "%.3f%%" % (100 * x)
print("period", res["period"], "n", res["n_overlapping"])
for tag, blk in hist.items():
    if tag == "full_sample_period": continue
    print("HIST", tag, hist["full_sample_period"] if tag == "full_sample" else "")
    for x, v in blk.items():
        print("  loss>=%s: %d/%d %s CI%s nonov%s episodes=%d" % (x, v["k_overlapping"], v["n"], f(v["freq"]),
              [f(c) for c in v["ci_wilson_n_over_5"]], v["k_nonoverlap_by_offset"], v["episodes"]))
for k, ev in res["events"].items():
    print(k, "obs", ev["observed"], "/", ev["n"], f(ev["freq"]), [f(c) for c in ev["ci_n_over_5"]])
    for m, v in ev["models"].items():
        print("   %-9s pred %s O/E %s %s brier %.6f diff %s" % (m, f(v["mean_pred"]), None if v["obs_over_exp"] is None else round(v["obs_over_exp"], 2),
              np.round(v.get("obs_over_exp_ci", [np.nan, np.nan]), 2), v["brier"], np.round(v.get("brier_diff_ci", [np.nan, np.nan]), 5)))
for tag, b in res["var_es"].items():
    print("VaR/ES", tag)
    for m, v in b.items():
        n0 = v["nonoverlap0"]
        print("   %-9s viol_ov %s | nonov %d/%d kupiec p %.3f christ p %.3f | ES z2 %.2f %s | VaR %.4f ES %.4f" % (m, f(v["viol_rate_overlapping"]),
              n0["violations"], n0["n"], n0["kupiec_p"], n0["christoffersen"]["p_value"], v["es_nonoverlap0"]["z2"], np.round(v["es_nonoverlap0"]["z2_ci"], 2), v["mean_var"], v["mean_es"]))
for name, r in reg.items():
    print("REGIME", name, "n", r["n"], "share %.2f" % r["share"])
    for k, ev in r["events"].items():
        print("  ", k, "obs", ev["observed"], f(ev["freq"]), " ".join("%s %s O/E %s" % (m[:5], f(v["mean_pred"]), None if v["obs_over_exp"] is None else round(v["obs_over_exp"], 2)) for m, v in ev["models"].items()))
    print("   VaR99 viol:", {m: f(v["viol_rate_overlapping"]) for m, v in r["var_es"]["99"].items()})
