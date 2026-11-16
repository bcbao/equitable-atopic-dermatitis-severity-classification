"""Methods 2.13 — patient-level bootstrap CIs, paired bootstrap, DeLong, McNemar."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as st


def _patient_index(df: pd.DataFrame):
    groups = df.groupby("patient_id").indices
    return list(groups.values())


def bootstrap_ci(df: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                 n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Patient-level (cluster) bootstrap: resample patients with replacement."""
    rng = np.random.default_rng(seed)
    idx = _patient_index(df)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        try:
            vals.append(metric(df.iloc[rows]))
        except ValueError:
            continue
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return (float(metric(df)), float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric, n_boot=1000, seed=42):
    """Two-sided paired bootstrap test of metric(A) − metric(B) on the same images.
    df_a and df_b must be row-aligned (same images in the same order).
    Returns (delta, ci_lo, ci_hi, p)."""
    assert (df_a["image_path"].to_numpy() == df_b["image_path"].to_numpy()).all(), "rows not aligned"
    rng = np.random.default_rng(seed)
    idx = _patient_index(df_a)
    obs = metric(df_a) - metric(df_b)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        try:
            deltas.append(metric(df_a.iloc[rows]) - metric(df_b.iloc[rows]))
        except ValueError:
            continue
    d = np.asarray([x for x in deltas if np.isfinite(x)])
    # p-value: proportion of bootstrap deltas on the other side of 0 (two-sided)
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    p = max(min(p, 1.0), 1.0 / len(d))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


# --------------------------------------------------------------------------- #
# DeLong test for two correlated AUROCs (Sun & Xu 2014 fast implementation)
# --------------------------------------------------------------------------- #
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y_bin, score_a, score_b):
    """Two-sided DeLong test for AUROC(A) vs AUROC(B) on the same cases.
    Returns (auc_a, auc_b, z, p)."""
    y_bin = np.asarray(y_bin).astype(int)
    order = np.argsort(-y_bin, kind="mergesort")
    m = int(y_bin.sum())
    preds = np.vstack([np.asarray(score_a), np.asarray(score_b)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * st.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def mcnemar_test(y, pred_a, pred_b):
    """McNemar test on paired correct/incorrect outcomes.
    Exact binomial if discordant pairs < 25, else χ² with continuity correction."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb))
    n10 = int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    if n01 + n10 < 25:
        p = st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = st.chi2.sf(chi2, 1)
    return n01, n10, float(p)


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
