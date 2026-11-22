"""Is 0.0704% (seed 20260101, 1e6 paths) vs exact 0.0656% a bug or sampling noise?
Event: simple 5-day return <= -8.6%, mu=5%, sigma=20%, 5 daily steps. 200 independent seeds."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from scipy import stats
from validation.tail_probability import estimate_loss_probability as est, normal_loss_probability as nlp
kw = dict(mu=0.05, sigma=0.20, horizon=5/252, steps=5, n_sims=1_000_000)
p = nlp(0.086, 0.05, 0.20, 5/252); n = kw["n_sims"]
sd = np.sqrt(n * p * (1 - p))
one = est(0.086, seed=20260101, **kw)
z1 = (one["hits"] - n * p) / sd
hits = np.array([est(0.086, seed=s, **kw)["hits"] for s in range(1000, 1200)])
z = (hits - n * p) / sd
tot, N = hits.sum(), n * len(hits)
pooled = (tot - N * p) / np.sqrt(N * p * (1 - p))
out = {"exact": p, "expected_hits": n * p, "binomial_sd": sd, "seed20260101_hits": one["hits"], "z_seed20260101": z1,
       "two_sided_p": float(2 * stats.norm.sf(abs(z1))),
       "200_seeds": {"mean_p": float(hits.mean() / n), "z_mean": float(z.mean()), "z_sd": float(z.std(ddof=1)),
                     "pooled_z": float(pooled), "frac_|z|>1.96": float(np.mean(np.abs(z) > 1.96)),
                     "ks_p_vs_N01": float(stats.kstest(z, "norm").pvalue)}}
print(json.dumps(out, indent=1)); json.dump(out, open("research/tail-risk/normal_gap_investigation.json", "w"), indent=1)
