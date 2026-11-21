"""Phase 1: try to reproduce P(5-day loss <= 8.6%) = 0.06% (Normal) / 0.28% (t, df=4).

Analytic and Monte Carlo (seeded) under explicit conventions. sigma=0.20/yr,
5 trading days, T=5/252, mu=0 for the log-return location unless stated.
"""
import json
import numpy as np
from scipy import stats

SIG, T, DF, N, SEED = 0.20, 5 / 252, 4.0, 1_000_000, 20260101
s_w = SIG * np.sqrt(T)
thr_simple, thr_log = -0.086, np.log(1 - 0.086)
out = {"sigma_week": s_w, "thr_log_of_simple": thr_log}

def t_std_cdf(x, df):  # unit-variance t
    return stats.t.cdf(x * np.sqrt(df / (df - 2)), df)

for conv, thr in (("simple -8.6%", thr_log), ("log -8.6%", thr_simple)):
    for loc_name, loc in (("loc=0", 0.0), ("loc=-s^2T/2 (GBM mu=0)", -0.5 * s_w**2)):
        z = (thr - loc) / s_w
        out[f"{conv} | {loc_name}"] = {
            "normal": float(stats.norm.cdf(z)),
            "t4_weekly_standardized": float(t_std_cdf(z, DF)),
        }
# daily t(4) shocks summed over 5 days: Monte Carlo in batches
rng = np.random.default_rng(SEED)
for conv, thr in (("simple -8.6%", thr_log), ("log -8.6%", thr_simple)):
    for loc_name, loc in (("loc=0", 0.0), ("loc=-s^2T/2 (GBM mu=0)", -0.5 * s_w**2)):
        hits = 0
        for _ in range(10):
            z = rng.standard_t(DF, size=(100_000, 5)).sum(axis=1)
            z *= np.sqrt((DF - 2) / DF) / np.sqrt(5)       # unit variance after sum
            hits += int(np.sum(loc + s_w * z <= thr))
        out[f"{conv} | {loc_name}"]["t4_daily_sum_MC"] = hits / N
        out[f"{conv} | {loc_name}"]["t4_daily_sum_hits"] = hits
print(json.dumps(out, indent=1))
