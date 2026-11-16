#!/usr/bin/env python
"""Fig 3 — diagnostic performance.

  python scripts/05_figure3.py --results results --runs runs --out figures/figure3.png

A  macro-F1 per backbone: baseline/full × internal CV (mean ± SD) / held-out test (95% CI)
B  macro-AUROC, same layout
C  held-out one-vs-rest ROC curves, MobileNetV2 full pipeline
D  6×6 phenotype-severity matrix (row-normalised), MobileNetV2 full pipeline, held-out
   (needs the `phenotype` column and a model trained with aux_phenotype: true)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

from adpipe import SEVERITY
from adpipe.metrics import PROB_COLS

BACKBONES = [("vgg16", "VGG16"), ("inception_v3", "InceptionV3"), ("mobilenet_v2", "MobileNetV2")]
SERIES = [("baseline", "cv", "Baseline · internal CV", "#2b7bd6", None),
          ("full", "cv", "Full pipeline · internal CV", "#f2a20c", "////"),
          ("baseline", "test", "Baseline · held-out test", "#1bab7a", None),
          ("full", "test", "Full pipeline · held-out test", "#e0474c", "////")]


def bar_panel(ax, cv, te, metric, ylim, title, letter):
    w = 0.19
    x = np.arange(len(BACKBONES))
    for j, (var, where, lab, col, hatch) in enumerate(SERIES):
        vals, err = [], []
        for bb, _ in BACKBONES:
            run = f"{bb}__{var}"
            if where == "cv":
                r = cv[cv.run == run]
                v = r[f"{metric}_mean"].iloc[0] if len(r) else np.nan
                e = [r[f"{metric}_sd"].iloc[0]] * 2 if len(r) else [0, 0]
            else:
                r = te[(te.run == run) & (te.metric == metric)]
                v = r.value.iloc[0] if len(r) else np.nan
                e = [v - r.ci_lo.iloc[0], r.ci_hi.iloc[0] - v] if len(r) else [0, 0]
            vals.append(v)
            err.append(e)
        err = np.array(err).T
        pos = x + (j - 1.5) * w
        ax.bar(pos, vals, w * 0.95, color=col, hatch=hatch, edgecolor="black", linewidth=0.8, label=lab,
               yerr=err, capsize=3, error_kw={"elinewidth": 1})
        for p, v, e in zip(pos, vals, err.T):
            if np.isfinite(v):
                ax.text(p, v + e[1] + 0.012 * (ylim[1] - ylim[0]), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [b[1] for b in BACKBONES])
    ax.set_ylim(*ylim)
    ax.set_ylabel(title.split(" across")[0])
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.1, 1.06, letter, transform=ax.transAxes, fontsize=18, fontweight="bold")


def roc_panel(ax, te):
    cols = ["#2b7bd6", "#f2a20c", "#1bab7a"]
    y = te.label.to_numpy()
    for k, (s, c) in enumerate(zip(SEVERITY, cols)):
        yk = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(yk, te[PROB_COLS[k]])
        auc = roc_auc_score(yk, te[PROB_COLS[k]])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{s.capitalize()} AD, AUROC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], ":", color="#d33", lw=1.5, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Held-out test one-vs-rest ROC curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.1, 1.06, "C", transform=ax.transAxes, fontsize=18, fontweight="bold")


def phenotype_bins(te, bins, pheno_classes):
    """True bin = (severity, recorded dominant phenotype); predicted bin = predicted
    severity + whichever of that severity's two phenotypes has the higher
    auxiliary-head probability. Images whose true pair is not one of the 6 bins
    are excluded (count printed)."""
    names = [f"{s}|{p}" for s, p in bins]
    true = (te.severity.str.lower() + "|" + te.phenotype.astype(str).str.lower())
    keep = true.isin(names).to_numpy()
    print(f"Fig 3D: {keep.sum()}/{len(te)} images fall into the 6 phenotype-severity bins")
    pred_sev = np.array(SEVERITY)[te[PROB_COLS].to_numpy().argmax(1)]
    pred = []
    for i, ps in enumerate(pred_sev):
        cand = [(s, p) for s, p in bins if s == ps]
        best = max(cand, key=lambda sp: te[f"ph_{sp[1]}"].iloc[i])
        pred.append(f"{best[0]}|{best[1]}")
    t = true[keep].map({n: i for i, n in enumerate(names)}).to_numpy()
    p = pd.Series(pred)[keep].map({n: i for i, n in enumerate(names)}).to_numpy()
    m = np.zeros((len(names), len(names)))
    for a, b in zip(t, p):
        m[a, b] += 1
    m = m / np.maximum(m.sum(1, keepdims=True), 1)
    return m, [f"{s.capitalize()}\n{p}" for s, p in bins]


def matrix_panel(fig, ax, te, cfg):
    ax.text(-0.25, 1.06, "D", transform=ax.transAxes, fontsize=18, fontweight="bold")
    if "phenotype" not in te.columns or not any(c.startswith("ph_") for c in te.columns):
        ax.axis("off")
        ax.text(0.5, 0.5, "Panel D needs a `phenotype` column and a model\ntrained with aux_phenotype: true",
                ha="center", va="center")
        return
    m, labels = phenotype_bins(te, cfg["phenotype_bins"], cfg["phenotype_classes"])
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if m[i, j] > 0.5 else "white")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.set_xlabel("Predicted phenotype-severity bin")
    ax.set_ylabel("True phenotype-severity bin")
    ax.set_title("Fine-grained phenotype-severity error matrix\nMobileNetV2 full pipeline, held-out test",
                 fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalized proportion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--roc-run", default="mobilenet_v2__full")
    ap.add_argument("--f1-ylim", nargs=2, type=float, default=[0.55, 0.90])
    ap.add_argument("--auc-ylim", nargs=2, type=float, default=[0.75, 0.95])
    ap.add_argument("--out", default="figures/figure3.png")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    cfg = yaml.safe_load(open(a.config))
    cv = pd.read_csv(os.path.join(a.results, "cv_summary.csv"))
    te_sum = pd.read_csv(os.path.join(a.results, "test_summary.csv"))
    te = pd.read_csv(os.path.join(a.runs, a.roc_run, "test_predictions.csv"))

    fig = plt.figure(figsize=(14.5, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, top=0.9)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    bar_panel(ax_a, cv, te_sum, "macro_f1", a.f1_ylim, "Macro-F1 across model backbones", "A")
    bar_panel(ax_b, cv, te_sum, "macro_auroc", a.auc_ylim, "Macro-AUROC across model backbones", "B")
    roc_panel(fig.add_subplot(gs[1, 0]), te)
    matrix_panel(fig, fig.add_subplot(gs[1, 1]), te, cfg)
    handles = [Patch(facecolor=c, edgecolor="black", hatch=h, label=l) for _, _, l, c, h in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.97))
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
