#!/usr/bin/env python
"""Fig 2 — topographic maps (A–D) and AD-TSS panel (E).

  python scripts/06_figure2_topographic.py \
      --images data/preprocessed/A.png data/preprocessed/B.png data/preprocessed/C.png data/preprocessed/D.png \
      --labels "Mild AD" "Severe AD" "Fitzpatrick I–III stratum" "Fitzpatrick IV–VI stratum" \
      --predictions results/test_with_adtss__mobilenet_v2__full.csv \
      --out figures/figure2.png

AD-TSS values in panel E are read from the predictions file (column ad_tss,
matched on image_path). If an image is not in that file, pass --tss manually.
Also writes the topographic descriptors of each panel to <out>_descriptors.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from adpipe.adtss import ad_tss_category
from adpipe.topographic import keypoints, plot_topographic, topographic_descriptors

CAT_COLORS = {"mild-likely": "#2e9e3e", "moderate-likely": "#f0901e", "severe-likely": "#d01c1c"}


def adtss_panel(fig, gs, labels, scores, illustrative=False):
    ax = fig.add_subplot(gs)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.97, "E", fontsize=20, fontweight="bold", va="top")
    ax.text(0.04, 0.95, "Derived AD Topographic Severity Score (AD-TSS)", fontsize=13, fontweight="bold", va="top")
    ax.text(0.33, 0.80, r"AD-TSS $= \dfrac{0\times P_{\mathrm{mild}} + 1\times P_{\mathrm{moderate}} + 2\times P_{\mathrm{severe}}}{2}$",
            fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", fc="#eef4fb", ec="#7aa6d8"))
    ax.text(0.80, 0.80, "Higher AD-TSS indicates greater\nalgorithm-predicted disease severity.",
            fontsize=10.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", fc="#fdf3e3", ec="#e5a64a"))
    # colour scale
    cmap = LinearSegmentedColormap.from_list("tss", ["#2e9e3e", "#f5c518", "#f0901e", "#d01c1c"])
    x0, x1, yb = 0.06, 0.94, 0.58
    grad = np.linspace(0, 1, 512)[None, :]
    ax.imshow(grad, extent=[x0, x1, yb - 0.025, yb + 0.025], aspect="auto", cmap=cmap)
    ax.text(x0 - 0.02, yb, "0", fontsize=14, fontweight="bold", ha="right", va="center")
    ax.text(x1 + 0.02, yb, "1", fontsize=14, fontweight="bold", ha="left", va="center")
    ax.text(0.5, yb + 0.07, "AD-TSS Score Scale (0 to 1)", fontsize=11, fontweight="bold", ha="center")
    for t in (0.33, 0.66):
        xt = x0 + t * (x1 - x0)
        ax.plot([xt, xt], [yb - 0.04, yb + 0.04], "k--", lw=1.2)
        ax.text(xt, yb - 0.07, f"{t:.2f}", ha="center", fontsize=11, fontweight="bold")
    for (lab, rng), cx in zip([("Mild-likely", "(0.00–0.33)"), ("Moderate-likely", "(0.34–0.66)"),
                               ("Severe-likely", "(0.67–1.00)")], (0.17, 0.5, 0.83)):
        col = CAT_COLORS[lab.lower()]
        ax.text(cx, yb - 0.13, lab, color=col, fontsize=11.5, fontweight="bold", ha="center")
        ax.text(cx, yb - 0.19, rng, fontsize=10.5, ha="center")
    ax.text(0.5, 0.29, "Example AD-TSS for panels A–D" + (" (illustrative)" if illustrative else ""),
            fontsize=11, fontweight="bold", ha="center")
    n = len(labels)
    for i, (lab, s) in enumerate(zip(labels, scores)):
        cx = (i + 0.5) / n
        lx0, lx1 = cx - 0.09, cx + 0.09
        ax.text(cx, 0.21, lab, ha="center", fontsize=9.5)
        ax.plot([lx0, lx1], [0.14, 0.14], color="0.4", lw=1)
        for e in (lx0, lx1):
            ax.plot([e, e], [0.125, 0.155], color="0.4", lw=1)
        ax.text(lx0, 0.09, "0", ha="center", fontsize=8)
        ax.text(lx1, 0.09, "1", ha="center", fontsize=8)
        col = CAT_COLORS[ad_tss_category(s)[0]]
        ax.scatter([lx0 + s * (lx1 - lx0)], [0.14], s=110, color=col, edgecolor="k", zorder=3)
        ax.text(cx, 0.02, f"{s:.2f}", ha="center", fontsize=14, fontweight="bold", color=col)
        if i < n - 1:
            ax.plot([(i + 1) / n] * 2, [0.02, 0.22], color="0.8", lw=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs=4, required=True)
    ap.add_argument("--labels", nargs=4, default=["Mild AD", "Severe AD",
                                                  "Fitzpatrick I–III stratum", "Fitzpatrick IV–VI stratum"])
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--image-root", default="data/preprocessed")
    ap.add_argument("--tss", nargs=4, type=float, default=None)
    ap.add_argument("--out", default="figures/figure2.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--grad-threshold", type=float, default=0.04,
                    help="fixed gradient threshold for key points (same for all images)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.tss is not None:
        scores = a.tss
    elif a.predictions:
        p = pd.read_csv(a.predictions).set_index("image_path")
        scores = []
        for im in a.images:
            rel = os.path.relpath(im, a.image_root)
            if rel not in p.index:
                sys.exit(f"{rel} not found in {a.predictions}; use --tss")
            scores.append(float(p.at[rel, "ad_tss"]))
    else:
        sys.exit("give --predictions or --tss")

    fig = plt.figure(figsize=(14, 13))
    outer = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.08, wspace=0.08)
    desc = []
    imgs = [cv2.imread(im) for im in a.images]
    kps = [keypoints(img, grad_threshold=a.grad_threshold) for img in imgs]
    allg = np.concatenate([k["g"] for k in kps if len(k["g"])]) if any(len(k["g"]) for k in kps) else np.array([0.1])
    vmax = float(np.percentile(allg, 99))  # one colour scale for all panels so they are comparable
    allz = np.concatenate([np.abs(k["z"]) for k in kps if len(k["z"])]) if any(len(k["z"]) for k in kps) else np.array([0.1])
    zlim = float(np.ceil(np.percentile(allz, 99) * 20) / 20)  # one Z range for all panels
    for i, (im, lab, img, kp) in enumerate(zip(a.images, a.labels, imgs, kps)):
        d = topographic_descriptors(kp)
        d.update({"panel": "ABCD"[i], "label": lab, "image": im})
        desc.append(d)
        sub = outer[i // 2, i % 2].subgridspec(1, 2, width_ratios=[0.75, 1.25], wspace=0.02)
        ax_img = fig.add_subplot(sub[0])
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")
        ax_img.set_title(f"$\\bf{{{'ABCD'[i]}}}$   {lab}", loc="left", fontsize=12)
        ax3 = fig.add_subplot(sub[1], projection="3d")
        sc = plot_topographic(ax3, kp, vmax=vmax, zlim=zlim, title=f"3D topographic map ({len(kp['x'])} key points)")
        cb = fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.22)
        cb.set_label("Local surface gradient (a.u.)", fontsize=8)
        ax3.tick_params(labelsize=7)
    adtss_panel(fig, outer[2, :], a.labels, scores, illustrative=a.tss is not None)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    pd.DataFrame(desc).to_csv(os.path.splitext(a.out)[0] + "_descriptors.csv", index=False)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
