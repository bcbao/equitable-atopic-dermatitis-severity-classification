#!/usr/bin/env python
"""Section 2.8: generate Fitzpatrick IV–VI-toned synthetic images per split.

For each fold k (1–5), every source image is a Fitzpatrick I–III image of
that split's training subset, and the generator is Stable Diffusion v1.5
with the LoRA adapter trained on the same split (./lora_fold_k):

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights("./lora_fold_k")
    pipe(prompt="a clinical photo of skin, Fitzpatrick type V", image=src,
         strength=0.4, guidance_scale=7.5, num_inference_steps=50, generator=seed 42)

  python scripts/02b_generate_synthetic.py --metadata data/metadata.csv \
      --image-root data/preprocessed --lora-root . --ratio 1.0

Writes
  data/synthetic_manifest.csv       one row per synthetic image (fold, source, QC columns)
  data/metadata_with_synthetic.csv  real rows + kept synthetic rows (column syn_fold)

Dermatologist review: set review_status to "rejected" in the manifest for
images that fail the checklist (erythema extent, excoriation/lichenification/
scale texture, no colour bleeding/halo/distortion), then rerun with
--merge-only. Synthetic rows keep the source patient's ID and are used only
in the training subset of their own fold (see data.fold_frames).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from adpipe.data import load_metadata, make_patient_folds
from adpipe.generative import (BASE_MODEL, IMG2IMG_PROMPT, lora_dir, n_to_generate, pick_sources,
                               split_training_frames)
from adpipe.qc import edge_ssim


def merge(meta_csv, manifest_csv, out_csv):
    base = pd.read_csv(meta_csv)
    if "synthetic" in base:
        base = base[base["synthetic"].fillna(0).astype(int) == 0]
    man = pd.read_csv(manifest_csv)
    keep = man[man.review_status.astype(str).str.lower() != "rejected"]
    cols = [c for c in keep.columns if c in base.columns] + ["synthetic", "syn_fold", "source_image"]
    out = pd.concat([base, keep[list(dict.fromkeys(cols))]], ignore_index=True)
    out["synthetic"] = out["synthetic"].fillna(0).astype(int)
    out.to_csv(out_csv, index=False)
    print(f"{len(keep)}/{len(man)} synthetic images kept -> {out_csv}")
    print(man.assign(kept=man.review_status.astype(str).str.lower() != "rejected")
             .groupby(["syn_fold", "severity"])["kept"].agg(["size", "sum"]).rename(
                 columns={"size": "generated", "sum": "retained"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--image-root", default="data/preprocessed")
    ap.add_argument("--lora-root", default=".", help="folder containing lora_fold_1 ... lora_fold_5")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="target IV–VI : I–III image ratio per severity class in each training subset")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--prompt", default=IMG2IMG_PROMPT)
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ssim-flag", type=float, default=0.6,
                    help="edge-SSIM below this value is flagged for dermatologist review")
    ap.add_argument("--manifest", default="data/synthetic_manifest.csv")
    ap.add_argument("--out-csv", default="data/metadata_with_synthetic.csv")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only print how many images would be made")
    a = ap.parse_args()

    if a.merge_only:
        return merge(a.metadata, a.manifest, a.out_csv)

    df = load_metadata(a.metadata)
    dev = df[(df.split == "dev")].reset_index(drop=True)
    folds = make_patient_folds(dev, a.n_folds, a.split_seed)
    rng = np.random.default_rng(a.seed)

    plan = {}
    for k in [int(f) for f in a.folds.split(",")]:
        train = split_training_frames(dev, folds, k - 1)
        plan[k] = (train, n_to_generate(train, a.ratio))
        print(f"fold {k}: training images {len(train)}; to generate {plan[k][1]}")
    if a.dry_run:
        return

    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(a.base_model, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for k, (train, need) in plan.items():
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora_dir(a.lora_root, k))  # fold-specific LoRA adapter
        out_dir = os.path.join(a.image_root, "synthetic", f"fold{k}")
        os.makedirs(out_dir, exist_ok=True)
        for sev, n in need.items():
            pool = train[(train.severity == sev) & (train.fitz_group == "I-III")]
            for j, r in pick_sources(pool, n, rng).iterrows():
                src = Image.open(os.path.join(a.image_root, r.image_path)).convert("RGB").resize((a.size, a.size))
                gen = torch.Generator("cuda").manual_seed(a.seed + j)
                out = pipe(prompt=a.prompt, image=src, strength=a.strength, guidance_scale=a.guidance,
                           num_inference_steps=a.steps, generator=gen).images[0]
                name = f"syn_f{k}_{sev}_{j:04d}_{os.path.splitext(os.path.basename(r.image_path))[0]}.png"
                out.save(os.path.join(out_dir, name))
                s = edge_ssim(cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))
                row = r.to_dict()
                row.update({"image_path": os.path.join("synthetic", f"fold{k}", name), "synthetic": 1,
                            "syn_fold": k, "source_image": r.image_path, "fitzpatrick": 5, "fitz_group": "IV-VI",
                            "edge_ssim": s, "qc_flag": s < a.ssim_flag, "review_status": "pending"})
                rows.append(row)
        print(f"fold {k}: generated {sum(need.values())} images")
    man = pd.DataFrame(rows)
    man.to_csv(a.manifest, index=False)
    print(f"manifest -> {a.manifest}; review it, mark rejected images, then rerun with --merge-only")
    merge(a.metadata, a.manifest, a.out_csv)


if __name__ == "__main__":
    main()
