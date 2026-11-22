"""Backtest statistics: Kupiec, Christoffersen, Brier, ES residual tests.

Conventions: losses are positive fractions of the initial price (simple
return R -> loss = -R); VaR_a is the a-quantile of the loss; a violation is
loss > VaR_a. All tests assume the observations passed are non-overlapping.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlogy


def kupiec_pof(violations: np.ndarray, alpha: float) -> dict:
    """Unconditional coverage LR test; expected violation probability 1-alpha."""
    v = np.asarray(violations, dtype=bool)
    n, x = v.size, int(v.sum())
    p = 1 - alpha
    if n == 0:
        raise ValueError("empty sample")
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = float(max(0.0, -2 * (ll0 - ll1)))
    return {"n": n, "violations": x, "rate": pi, "expected_rate": p,
            "lr": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen_independence(violations: np.ndarray) -> dict:
    """First-order Markov independence LR test; NaN when not identifiable."""
    v = np.asarray(violations, dtype=int)
    a, b = v[:-1], v[1:]
    n00 = int(np.sum((a == 0) & (b == 0))); n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0))); n11 = int(np.sum((a == 1) & (b == 1)))
    out = {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "lr": float("nan"),
           "p_value": float("nan"), "applicable": False}
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return out          # no violations or no violation followed by an observation
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = (xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11))
    lr = float(max(0.0, -2 * (ll0 - ll1)))
    out.update(lr=lr, p_value=float(stats.chi2.sf(lr, 1)), applicable=True)
    return out


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    p, y = np.asarray(prob, float), np.asarray(outcome, float)
    if p.shape != y.shape or p.size == 0 or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0,1] and match outcomes")
    return float(np.mean((p - y) ** 2))


def reliability_table(prob: np.ndarray, outcome: np.ndarray, edges: list[float]) -> list[dict]:
    """Mean forecast vs observed frequency per probability bin (right-open)."""
    p, y = np.asarray(prob, float), np.asarray(outcome, float)
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.any():
            rows.append({"bin": [edges[k], edges[k + 1]], "n": int(m.sum()),
                         "mean_forecast": float(p[m].mean()), "observed": float(y[m].mean()),
                         "events": int(y[m].sum())})
    return rows


def es_residual_tests(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Acerbi-Szekely Z1/Z2 style ES diagnostics with iid-bootstrap intervals.

    Z1 = mean(loss/ES | violation) - 1 (needs violations).
    Z2 = mean(loss*1{viol}/(ES*(1-alpha))) - 1 over all dates; both ~0 if ES is
    right. This is a diagnostic with a bootstrap CI, NOT an exact test: the
    bootstrap ignores forecast-distribution dependence and is unreliable with
    few violations. Positive values indicate ES understates the tail.
    """
    L, V, E = (np.asarray(a, float) for a in (loss, var, es))
    viol = L > V
    n = L.size

    def z1(i):
        m = viol[i]
        return np.mean(L[i][m] / E[i][m]) - 1 if m.any() else np.nan

    def z2(i):
        return np.mean(L[i] * viol[i] / (E[i] * (1 - alpha))) - 1

    allidx = np.arange(n)
    rng = np.random.default_rng(seed)
    b1, b2 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        b1.append(z1(i)); b2.append(z2(i))
    def ci(b):
        return [float(x) for x in np.nanpercentile(b, [2.5, 97.5])] if np.isfinite(b).any() else [np.nan, np.nan]

    return {"n": n, "violations": int(viol.sum()), "z1": float(z1(allidx)), "z1_ci": ci(b1),
            "z2": float(z2(allidx)), "z2_ci": ci(b2), "bootstrap": n_boot, "seed": seed}


def effective_wilson(k: int, n: int, overlap: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson interval with n replaced by n/overlap (approximate, for overlapping windows)."""
    from validation.tail_probability import wilson_interval
    n_eff = max(1, int(round(n / overlap)))
    return wilson_interval(int(round(k * n_eff / n)), n_eff, level)


def block_bootstrap(fn, n: int, block: int = 20, reps: int = 2000, seed: int = 1) -> list:
    """Moving-block bootstrap draws of ``fn(index_array)`` over a series of length n."""
    rng = np.random.default_rng(seed)
    nb = -(-n // block)
    out = []
    for _ in range(reps):
        s = rng.integers(0, max(1, n - block + 1), nb)
        out.append(fn(np.concatenate([np.arange(i, min(i + block, n)) for i in s])[:n]))
    return out


def obs_over_exp_ci(y: np.ndarray, p: np.ndarray, **kw) -> list:
    """95% block-bootstrap CI for sum(y)/sum(p) (calibration-in-the-large)."""
    b = block_bootstrap(lambda i: y[i].sum() / p[i].sum() if p[i].sum() > 0 else np.nan, y.size, **kw)
    return [float(x) for x in np.nanpercentile(b, [2.5, 97.5])]


def brier_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray, **kw) -> tuple:
    """Mean Brier(a) - Brier(b) and its 95% block-bootstrap CI (negative favours a)."""
    d = (a - y) ** 2 - (b - y) ** 2
    m = block_bootstrap(lambda i: d[i].mean(), d.size, **kw)
    return float(d.mean()), [float(x) for x in np.percentile(m, [2.5, 97.5])]
