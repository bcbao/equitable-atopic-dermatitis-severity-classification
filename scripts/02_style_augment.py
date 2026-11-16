#!/usr/bin/env python
"""Methods 2.8 — generate Fitzpatrick IV–VI style-transferred training images.

  python scripts/02_style_augment.py --metadata data/metadata.csv \
      --image-root data/preprocessed --style-dir data/style_refs_IV_VI \
      --ratio 1.0 --method nst

Writes
  data/synthetic_manifest.csv       one row per synthetic image (QC + review columns)
  data/metadata_with_synthetic.csv  original rows + accepted synthetic rows
Point `metadata_csv` in the config to the second file.

Review workflow: open synthetic_manifest.csv, set review_status to
'rejected' for unrealistic images (dermatologist screening), then re-run with
--merge-only to rebuild metadata_with_synthetic.csv.

Synthetic rows keep the *source patient's* patient_id, so they can only enter
the training side of a fold (and never validation/test).
Style references: best is a folder of normal/peri-lesional skin patches from
Fitzpatrick IV–VI patients (--style-dir). Without it, IV–VI development images
are sampled as references.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from adpipe import SEVERITY
from adpipe.data import load_metadata
from adpipe.style_augment import VGGFeatures, edge_ssim, neural_style_transfer, reinhard_transfer


def merge(meta_csv, manifest_csv, out_csv):
    base = pd.read_csv(meta_csv)
    base = base[base.get("synthetic", 0).fillna(0).astype(int) == 0] if "synthetic" in base else base
    man = pd.read_csv(manifest_csv)
    keep = man[(man.review_status != "rejected") & (~man.qc_flag.astype(bool) | (man.review_status == "accepted"))]
    out = pd.concat([base, keep[[c for c in keep.columns if c in base.columns or c in
                                 ("synthetic", "source_image", "style_image")]]], ignore_index=True)
    out["synthetic"] = out["synthetic"].fillna(0).astype(int)
    out.to_csv(out_csv, index=False)
    print(f"{len(keep)}/{len(man)} synthetic images kept -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--image-root", default="data/preprocessed")
    ap.add_argument("--style-dir", default=None)
    ap.add_argument("--style-fitzpatrick", type=int, default=5,
                    help="Fitzpatrick type recorded for synthetic rows when --style-dir is used")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="target IV–VI : I–III image ratio per severity class in the dev set")
    ap.add_argument("--method", choices=["nst", "reinhard"], default="nst")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--style-weight", type=float, default=1e4)
    ap.add_argument("--ssim-threshold", type=float, default=0.6)
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="data/synthetic_manifest.csv")
    ap.add_argument("--out-csv", default="data/metadata_with_synthetic.csv")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()

    if a.merge_only:
        return merge(a.metadata, a.manifest, a.out_csv)

    rng = np.random.default_rng(a.seed)
    df = load_metadata(a.metadata)
    dev = df[(df.split == "dev") & (df.synthetic == 0)]

    if a.style_dir:
        refs = sorted(glob.glob(os.path.join(a.style_dir, "*")))
        style_pool = [(p, a.style_fitzpatrick) for p in refs]
    else:
        iv = dev[dev.fitz_group == "IV-VI"]
        style_pool = [(os.path.join(a.image_root, p), f) for p, f in zip(iv.image_path, iv.fitzpatrick)]
    if not style_pool:
        sys.exit("No Fitzpatrick IV–VI style references found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = VGGFeatures(pretrained=not a.no_pretrained).to(device) if a.method == "nst" else None
    out_dir = os.path.join(a.image_root, "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sev in SEVERITY:
        src = dev[(dev.severity == sev) & (dev.fitz_group == "I-III")]
        n_iv = int(((dev.severity == sev) & (dev.fitz_group == "IV-VI")).sum())
        n_need = max(0, int(round(a.ratio * len(src))) - n_iv)
        print(f"{sev}: I–III={len(src)}  IV–VI={n_iv}  -> generating {n_need}")
        if n_need == 0 or len(src) == 0:
            continue
        picks = np.concatenate([rng.permutation(len(src)) for _ in range(n_need // len(src) + 1)])[:n_need]
        for j, i in enumerate(tqdm(picks, desc=sev)):
            r = src.iloc[i]
            content = cv2.imread(os.path.join(a.image_root, r.image_path))
            sp, sf = style_pool[rng.integers(len(style_pool))]
            style = cv2.imread(sp)
            if a.method == "nst":
                out = neural_style_transfer(content, style, net, size=a.size, steps=a.steps,
                                            style_weight=a.style_weight, device=device)
                out = cv2.resize(out, content.shape[1::-1], interpolation=cv2.INTER_CUBIC)
            else:
                out = reinhard_transfer(content, style)
            name = f"syn_{sev}_{j:04d}_{os.path.splitext(os.path.basename(r.image_path))[0]}.png"
            cv2.imwrite(os.path.join(out_dir, name), out)
            s = edge_ssim(content, out)
            row = r.to_dict()
            row.update({"image_path": os.path.join("synthetic", name), "synthetic": 1,
                        "source_image": r.image_path, "style_image": sp,
                        "fitzpatrick": sf, "fitz_group": "IV-VI",
                        "edge_ssim": s, "qc_flag": s < a.ssim_threshold, "review_status": "pending"})
            rows.append(row)

    man = pd.DataFrame(rows)
    man.to_csv(a.manifest, index=False)
    print(f"manifest -> {a.manifest} ({len(man)} images, {int(man.qc_flag.sum()) if len(man) else 0} flagged by SSIM)")
    merge(a.metadata, a.manifest, a.out_csv)


if __name__ == "__main__":
    main()
