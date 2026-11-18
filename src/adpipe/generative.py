"""Section 2.8: skin-tone-targeted generative augmentation (shared helpers).

Method used in the study
  * Base model: Stable Diffusion v1.5 (latent diffusion), Hugging Face
    diffusers 0.25.0, PyTorch 2.1.0.
  * One LoRA adapter per cross-validation split, trained only on the
    Fitzpatrick IV–VI images of that split's training subset
    (rank 16, alpha 16, 1,200 steps, learning rate 1e-4, batch size 2,
    512 × 512 pixels).
  * Synthetic images: img2img translation of each Fitzpatrick I–III
    training image of the same split with the split's LoRA
    (prompt "a clinical photo of skin, Fitzpatrick type V", strength 0.4,
    guidance 7.5, 50 steps, float16, seed 42).

Folds are numbered 1–5 (as in Supplementary Table S1); the adapter of fold k
is saved in <lora-root>/lora_fold_k.

Scripts: scripts/02a_train_lora.py, scripts/02b_generate_synthetic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMG2IMG_PROMPT = "a clinical photo of skin, Fitzpatrick type V"

ROMAN = {4: "IV", 5: "V", 6: "VI"}
MORPHOLOGY_TEXT = {
    "xerotic": "dry scaly skin",
    "erythematous": "erythematous patches",
    "excoriated": "excoriations",
    "lichenified": "lichenified plaques",
    "exudative": "exudative crusted lesions",
}


def lora_caption(row: pd.Series) -> str:
    """Training caption for one Fitzpatrick IV–VI image.

    Clinical photograph of AD on Fitzpatrick IV–VI skin, realistic texture,
    inflammatory morphology and natural lighting, plus a lesion-specific
    descriptor when the `phenotype` column is filled. A non-empty `caption`
    column in the metadata overrides the template.
    """
    if "caption" in row and isinstance(row["caption"], str) and row["caption"].strip():
        return row["caption"].strip()
    try:
        fitz = ROMAN.get(int(float(row.get("fitzpatrick"))), "IV-VI")
    except (TypeError, ValueError):
        fitz = "IV-VI"
    parts = [f"a clinical photograph of atopic dermatitis on Fitzpatrick type {fitz} skin"]
    morph = str(row.get("phenotype", "")).strip().lower()
    if morph in MORPHOLOGY_TEXT:
        parts.append(MORPHOLOGY_TEXT[morph])
    parts += ["inflammatory morphology", "realistic skin texture", "natural clinical lighting"]
    return ", ".join(parts)


def split_training_frames(dev: pd.DataFrame, folds, k: int) -> pd.DataFrame:
    """Real images of the training subset of split k (validation fold excluded)."""
    train_pids, _ = folds[k]
    real = dev[dev.synthetic == 0]
    return real[real.patient_id.isin(train_pids)].reset_index(drop=True)


def n_to_generate(train: pd.DataFrame, ratio: float):
    """Per severity class: how many synthetic images bring the IV–VI : I–III
    image ratio of the training subset up to `ratio`."""
    out = {}
    for sev, g in train.groupby("severity"):
        n13 = int((g.fitz_group == "I-III").sum())
        n46 = int((g.fitz_group == "IV-VI").sum())
        out[sev] = max(0, int(round(ratio * n13)) - n46) if n13 else 0
    return out


def pick_sources(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n source images, cycling through the pool before repeating."""
    if n == 0 or len(pool) == 0:
        return pool.iloc[0:0]
    order = np.concatenate([rng.permutation(len(pool)) for _ in range(n // len(pool) + 1)])[:n]
    return pool.iloc[order].reset_index(drop=True)


def lora_dir(root: str, fold_number: int) -> str:
    """Folder of the LoRA adapter for fold 1–5, e.g. ./lora_fold_1."""
    import os
    return os.path.join(root, f"lora_fold_{fold_number}")
