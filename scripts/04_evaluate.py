#!/usr/bin/env python
"""Evaluate all runs: CV mean±SD, held-out test with patient-level bootstrap
95% CI, calibration, Fitzpatrick fairness, AD-TSS, and pairwise comparisons.

  python scripts/04_evaluate.py --runs runs --out results \
      --compare mobilenet_v2__full:mobilenet_v2__baseline \
                mobilenet_v2__full:mobilenet_v2__no_metadata \
                mobilenet_v2__full:mobilenet_v2__no_style_aug
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from adpipe import FITZ_GROUPS, SEVERITY
from adpipe.adtss import ad_tss, ad_tss_category
from adpipe.metrics import (PROB_COLS, all_metrics, fairness_metrics, macro_auroc, macro_f1,
                            nri_multiclass, reliability_table)
from adpipe.stats import bootstrap_ci, delong_test, mcnemar_test, mean_sd, paired_bootstrap

KEY_METRICS = ["macro_f1", "accuracy", "macro_auroc", "ece", "brier",
               "auroc_mild", "auroc_moderate", "auroc_severe",
               "sens_mild", "sens_moderate", "sens_severe",
               "spec_mild", "spec_moderate", "spec_severe"]


def f1_of(d):
    return macro_f1(d.label, d[PROB_COLS].to_numpy().argmax(1))


def auc_of(d):
    return macro_auroc(d.label.to_numpy(), d[PROB_COLS].to_numpy())


def load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*__*"))):
        name = os.path.basename(d)
        cv = os.path.join(d, "cv_predictions.csv")
        te = os.path.join(d, "test_predictions.csv")
        runs[name] = {"cv": pd.read_csv(cv) if os.path.exists(cv) else None,
                      "test": pd.read_csv(te) if os.path.exists(te) else None}
    return runs


def summarise_cv(name, cv):
    rows = []
    for k, g in cv.groupby("fold"):
        m = all_metrics(g.label, g[PROB_COLS].to_numpy())
        m.update(fairness_metrics(g))
        m.update({"run": name, "fold": k})
        rows.append(m)
    by_fold = pd.DataFrame(rows)
    summ = {"run": name}
    for c in KEY_METRICS + [c for c in by_fold.columns if c.startswith("gap_") or c.endswith(":macro_f1")]:
        if c in by_fold:
            mu, sd = mean_sd(by_fold[c])
            summ[f"{c}_mean"], summ[f"{c}_sd"] = mu, sd
    return by_fold, summ


def summarise_test(name, te, n_boot, seed):
    rows = []
    for metric, fn in [("macro_f1", f1_of), ("macro_auroc", auc_of),
                       ("accuracy", lambda d: np.mean(d.label == d[PROB_COLS].to_numpy().argmax(1))),
                       ("ece", lambda d: all_metrics(d.label, d[PROB_COLS].to_numpy())["ece"]),
                       ("brier", lambda d: all_metrics(d.label, d[PROB_COLS].to_numpy())["brier"])]:
        v, lo, hi = bootstrap_ci(te, fn, n_boot, seed)
        rows.append({"run": name, "metric": metric, "value": v, "ci_lo": lo, "ci_hi": hi})
    full = all_metrics(te.label, te[PROB_COLS].to_numpy())
    for k in KEY_METRICS:
        if k not in {r["metric"] for r in rows}:
            rows.append({"run": name, "metric": k, "value": full[k], "ci_lo": np.nan, "ci_hi": np.nan})
    for g in FITZ_GROUPS:
        sub = te[te.fitz_group == g]
        if len(sub):
            v, lo, hi = bootstrap_ci(sub, f1_of, n_boot, seed)
            rows.append({"run": name, "metric": f"{g}:macro_f1", "value": v, "ci_lo": lo, "ci_hi": hi})
    fm = fairness_metrics(te)
    for k, v in fm.items():
        if k.startswith("gap_"):
            rows.append({"run": name, "metric": k, "value": v, "ci_lo": np.nan, "ci_hi": np.nan})
    return rows


def align(a, b):
    m = a.merge(b, on="image_path", suffixes=("_a", "_b"))
    A = a.set_index("image_path").loc[m.image_path].reset_index()
    B = b.set_index("image_path").loc[m.image_path].reset_index()
    return A, B


def compare(name_a, name_b, A, B, n_boot, seed, where):
    A, B = align(A, B)
    rows = []
    d, lo, hi, p = paired_bootstrap(A, B, f1_of, n_boot, seed)
    rows.append({"set": where, "A": name_a, "B": name_b, "test": "paired bootstrap ΔmacroF1",
                 "estimate": d, "ci_lo": lo, "ci_hi": hi, "p": p})
    d, lo, hi, p = paired_bootstrap(A, B, auc_of, n_boot, seed)
    rows.append({"set": where, "A": name_a, "B": name_b, "test": "paired bootstrap ΔmacroAUROC",
                 "estimate": d, "ci_lo": lo, "ci_hi": hi, "p": p})
    for k, s in enumerate(SEVERITY):
        yk = (A.label.to_numpy() == k).astype(int)
        if 0 < yk.sum() < len(yk):
            aa, ab, z, p = delong_test(yk, A[PROB_COLS[k]], B[PROB_COLS[k]])
            rows.append({"set": where, "A": name_a, "B": name_b, "test": f"DeLong AUROC {s}",
                         "estimate": aa - ab, "ci_lo": np.nan, "ci_hi": np.nan, "p": p})
    pa, pb = A[PROB_COLS].to_numpy().argmax(1), B[PROB_COLS].to_numpy().argmax(1)
    n01, n10, p = mcnemar_test(A.label, pa, pb)
    rows.append({"set": where, "A": name_a, "B": name_b, "test": f"McNemar (A right/B wrong={n01}, A wrong/B right={n10})",
                 "estimate": n01 - n10, "ci_lo": np.nan, "ci_hi": np.nan, "p": p})
    rows.append({"set": where, "A": name_a, "B": name_b, "test": "NRI (A vs B)",
                 "estimate": nri_multiclass(A.label, pb, pa), "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan})
    for g in FITZ_GROUPS:
        ma, mb = (A.fitz_group == g).to_numpy(), (B.fitz_group == g).to_numpy()
        if ma.sum() > 5:
            d, lo, hi, p = paired_bootstrap(A[ma].reset_index(drop=True), B[mb].reset_index(drop=True),
                                            f1_of, n_boot, seed)
            rows.append({"set": where, "A": name_a, "B": name_b, "test": f"ΔmacroF1 Fitzpatrick {g}",
                         "estimate": d, "ci_lo": lo, "ci_hi": hi, "p": p})

    def gap(d):
        f = [f1_of(d[d.fitz_group == g]) for g in FITZ_GROUPS]
        return abs(f[0] - f[1])
    d, lo, hi, p = paired_bootstrap(A, B, gap, n_boot, seed)
    rows.append({"set": where, "A": name_a, "B": name_b, "test": "Δ Fitzpatrick macro-F1 gap",
                 "estimate": d, "ci_lo": lo, "ci_hi": hi, "p": p})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs="*", default=[], help="runA:runB pairs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    if not runs:
        sys.exit(f"no runs found in {a.runs}")

    by_fold_all, cv_summ, test_rows = [], [], []
    for name, r in runs.items():
        if r["cv"] is not None:
            bf, s = summarise_cv(name, r["cv"])
            by_fold_all.append(bf)
            cv_summ.append(s)
        if r["test"] is not None:
            te = r["test"]
            test_rows += summarise_test(name, te, a.n_boot, a.seed)
            reliability_table(te.label.to_numpy(), te[PROB_COLS].to_numpy()).to_csv(
                os.path.join(a.out, f"reliability_test__{name}.csv"), index=False)
            te = te.copy()
            te["ad_tss"] = ad_tss(te[PROB_COLS].to_numpy())
            te["ad_tss_category"] = ad_tss_category(te["ad_tss"])
            te.to_csv(os.path.join(a.out, f"test_with_adtss__{name}.csv"), index=False)
            (te.groupby(["severity", "fitz_group"])["ad_tss"].describe()
               .to_csv(os.path.join(a.out, f"adtss_by_group__{name}.csv")))

    if by_fold_all:
        pd.concat(by_fold_all).to_csv(os.path.join(a.out, "cv_metrics_by_fold.csv"), index=False)
        pd.DataFrame(cv_summ).to_csv(os.path.join(a.out, "cv_summary.csv"), index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_csv(os.path.join(a.out, "test_summary.csv"), index=False)

    comp = []
    for pair in a.compare:
        ra, rb = pair.split(":")
        for where in ("cv", "test"):
            A, B = runs[ra][where], runs[rb][where]
            if A is not None and B is not None:
                comp += compare(ra, rb, A, B, a.n_boot, a.seed, where)
    if comp:
        pd.DataFrame(comp).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)

    # short console report
    if cv_summ:
        print("\nCV (mean ± SD across folds)")
        for s in cv_summ:
            print(f"  {s['run']:35s} macro-F1 {s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f} | "
                  f"AUROC {s['macro_auroc_mean']:.3f} ± {s['macro_auroc_sd']:.3f} | "
                  f"ECE {100 * s['ece_mean']:.1f}% | F1 gap {100 * s.get('gap_macro_f1_mean', np.nan):.1f} pts")
    if test_rows:
        t = pd.DataFrame(test_rows)
        print("\nHeld-out test (95% CI, patient bootstrap)")
        for name, g in t.groupby("run"):
            r = g.set_index("metric")
            print(f"  {name:35s} macro-F1 {r.at['macro_f1', 'value']:.3f} "
                  f"[{r.at['macro_f1', 'ci_lo']:.3f}, {r.at['macro_f1', 'ci_hi']:.3f}] | "
                  f"AUROC {r.at['macro_auroc', 'value']:.3f}")
    print(f"\nresults written to {a.out}/")


if __name__ == "__main__":
    main()
