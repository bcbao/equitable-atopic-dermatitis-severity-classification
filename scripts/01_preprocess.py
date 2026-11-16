#!/usr/bin/env python
"""Methods 2.4 — preprocess every raw image once and cache the crops.

  python scripts/01_preprocess.py --metadata data/metadata_raw.csv \
      --raw-root data/raw --out-root data/preprocessed --out-csv data/metadata.csv

If the CSV has roi_x, roi_y, roi_w, roi_h (manual boxes) they are used;
otherwise the automatic a*-channel ROI is used. image_path in the output CSV
points to the cached PNG (relative to --out-root).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import pandas as pd
from tqdm import tqdm

from adpipe.preprocess import preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", default="data/preprocessed")
    ap.add_argument("--out-csv", default="data/metadata.csv")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--no-hair-removal", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.metadata)
    os.makedirs(a.out_root, exist_ok=True)
    new_paths, failed = [], []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        src = os.path.join(a.raw_root, r["image_path"])
        img = cv2.imread(src)
        if img is None:
            failed.append(src)
            new_paths.append(None)
            continue
        roi = tuple(r[c] for c in ("roi_x", "roi_y", "roi_w", "roi_h")) if "roi_x" in df.columns else None
        out = preprocess_image(img, roi=roi, max_side=a.max_side, hair_removal=not a.no_hair_removal)
        rel = os.path.splitext(r["image_path"])[0] + ".png"
        dst = os.path.join(a.out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out)
        new_paths.append(rel)
    df["raw_image_path"] = df["image_path"]
    df["image_path"] = new_paths
    df = df[df.image_path.notna()]
    df.to_csv(a.out_csv, index=False)
    print(f"{len(df)} images written to {a.out_root}; metadata -> {a.out_csv}")
    if failed:
        print(f"WARNING: {len(failed)} images could not be read, e.g. {failed[:3]}")


if __name__ == "__main__":
    main()
