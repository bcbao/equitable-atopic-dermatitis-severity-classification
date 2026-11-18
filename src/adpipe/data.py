"""Methods 2.3, 2.5, 2.7, 2.9 — metadata, patient-level splits, datasets.

Expected metadata CSV columns (see data/metadata_template.csv):
  image_path      path to the *preprocessed* image (relative to --image-root)
  patient_id      patient identifier (splits are made on this)
  severity        mild | moderate | severe
  fitzpatrick     1..6   (used for fairness strata and metadata fusion)
  split           dev | test   (test = held-out external site/devices)
  site, device    acquisition site / device (domain labels for 2.10)
  age, sex, anatomic_site, disease_duration_months, atopic_history
  phenotype       optional dominant morphology (xerotic, erythematous,
                  excoriated, lichenified, exudative) — only for Fig 3D
  synthetic       0/1 (1 = generated image, Section 2.8; training only)
  syn_fold        for synthetic rows: split (1–5) the image was generated for
  source_image    for synthetic rows: image it was generated from
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from . import FITZ_GROUPS, SEVERITY

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["severity"] = df["severity"].str.strip().str.lower()
    bad = set(df["severity"]) - set(SEVERITY)
    if bad:
        raise ValueError(f"Unknown severity labels: {bad}")
    df["label"] = df["severity"].map({s: i for i, s in enumerate(SEVERITY)}).astype(int)
    if "fitz_group" not in df.columns:
        df["fitz_group"] = np.where(df["fitzpatrick"].astype(float) <= 3, "I-III", "IV-VI")
    if "synthetic" not in df.columns:
        df["synthetic"] = 0
    df["synthetic"] = df["synthetic"].fillna(0).astype(int)
    if "split" not in df.columns:
        df["split"] = "dev"
    df["patient_id"] = df["patient_id"].astype(str)
    site = _as_category(df["site"]) if "site" in df else "site"
    device = _as_category(df["device"]) if "device" in df else "device"
    df["domain"] = site + "|" + device
    return df.reset_index(drop=True)


def check_no_leakage(dev: pd.DataFrame, test: pd.DataFrame):
    overlap = set(dev.patient_id) & set(test.patient_id)
    if overlap:
        raise RuntimeError(f"{len(overlap)} patients appear in both dev and test: {list(overlap)[:5]}")
    if (test.synthetic == 1).any():
        raise RuntimeError("Synthetic images found in the test set.")


# --------------------------------------------------------------------------- #
# Patient-level 5-fold CV (2.5)
# --------------------------------------------------------------------------- #
def make_patient_folds(dev: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """StratifiedGroupKFold on real images; strata = severity x Fitzpatrick group.
    Returns list of (train_patient_ids, val_patient_ids)."""
    real = dev[dev.synthetic == 0].reset_index(drop=True)
    strata = real["label"].astype(str) + "_" + real["fitz_group"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, va in sgkf.split(real, strata, groups=real["patient_id"]):
        folds.append((set(real.patient_id.iloc[tr]), set(real.patient_id.iloc[va])))
    return folds


def split_inner_val(train: pd.DataFrame, frac: float, seed: int):
    """Carve an inner validation set (by patient) out of the training fold for
    early stopping and temperature scaling, so the outer fold stays untouched."""
    real = train[train.synthetic == 0]
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    _, iv = next(gss.split(real, groups=real["patient_id"]))
    inner_pids = set(real.patient_id.iloc[iv])
    inner = real[real.patient_id.isin(inner_pids)]
    # synthetic images derived from inner-val patients are dropped entirely
    rest = train[~train.patient_id.isin(inner_pids)]
    return rest.reset_index(drop=True), inner.reset_index(drop=True)


def fold_frames(dev: pd.DataFrame, train_pids, val_pids, use_synthetic: bool, fold_number=None):
    """Training and validation frames of one split.

    Synthetic images keep their source patient's ID and carry the number of
    the split they were generated for (column syn_fold, 1–5). They enter only
    the training subset of that split and never a validation fold."""
    train = dev[dev.patient_id.isin(train_pids)]
    if not use_synthetic:
        train = train[train.synthetic == 0]
    elif "syn_fold" in train.columns and fold_number is not None:
        syn_ok = pd.to_numeric(train["syn_fold"], errors="coerce") == fold_number
        train = train[(train.synthetic == 0) | syn_ok]
    val = dev[dev.patient_id.isin(val_pids) & (dev.synthetic == 0)]
    return train.reset_index(drop=True), val.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Clinical metadata encoder (2.9)
# --------------------------------------------------------------------------- #
def _as_category(series: pd.Series) -> pd.Series:
    """Robust string categories: NaN/'' -> 'unknown', 4.0 -> '4'."""
    def norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
            return "unknown"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v).strip().lower()
    return series.map(norm).astype(object)


class MetaEncoder:
    """Standardise numeric variables (median imputation) and one-hot encode
    categorical variables (missing -> 'unknown'). Fit on training data only."""

    def __init__(self, numeric=None, categorical=None):
        self.numeric = list(numeric or [])
        self.categorical = list(categorical or [])
        self.stats = {}
        self.levels = {}

    def fit(self, df: pd.DataFrame):
        for c in self.numeric:
            v = pd.to_numeric(df[c], errors="coerce")
            med = float(v.median()) if v.notna().any() else 0.0
            v = v.fillna(med)
            self.stats[c] = (med, float(v.mean()), float(v.std() or 1.0))
        for c in self.categorical:
            vals = _as_category(df[c])
            self.levels[c] = sorted(set(vals) | {"unknown"})
        return self

    @property
    def dim(self):
        return len(self.numeric) + sum(len(v) for v in self.levels.values())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            med, mu, sd = self.stats[c]
            v = pd.to_numeric(df[c], errors="coerce").fillna(med).to_numpy(np.float32)
            cols.append(((v - mu) / sd)[:, None])
        for c in self.categorical:
            vals = _as_category(df[c])
            vals = vals.where(vals.isin(self.levels[c]), "unknown")
            oh = np.zeros((len(df), len(self.levels[c])), np.float32)
            idx = vals.map({l: i for i, l in enumerate(self.levels[c])}).to_numpy()
            oh[np.arange(len(df)), idx] = 1.0
            cols.append(oh)
        if not cols:
            return np.zeros((len(df), 0), np.float32)
        return np.concatenate(cols, axis=1).astype(np.float32)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"numeric": self.numeric, "categorical": self.categorical,
                       "stats": self.stats, "levels": self.levels}, f, indent=2)

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        enc = cls(d["numeric"], d["categorical"])
        enc.stats = {k: tuple(v) for k, v in d["stats"].items()}
        enc.levels = d["levels"]
        return enc


class LabelEncoder:
    """Maps strings to ints; unseen values -> -1 (ignored by the loss)."""

    def __init__(self, values=None):
        self.classes = sorted(set(values)) if values is not None else []

    def __len__(self):
        return len(self.classes)

    def transform(self, values):
        m = {c: i for i, c in enumerate(self.classes)}
        return np.array([m.get(v, -1) for v in values], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Dataset / transforms / sampler
# --------------------------------------------------------------------------- #
def build_transforms(size: int, train: bool):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:  # 2.7 standard augmentation, training only
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class ADDataset(Dataset):
    def __init__(self, df, image_root, size, train, meta=None, domains=None,
                 phenotypes=None, phenotype_classes=None):
        self.df = df.reset_index(drop=True)
        self.root = image_root
        self.tf = build_transforms(size, train)
        self.labels = self.df["label"].to_numpy()
        self.meta = meta if meta is not None else np.zeros((len(df), 0), np.float32)
        self.domains = domains if domains is not None else -np.ones(len(df), np.int64)
        if phenotype_classes and "phenotype" in self.df.columns:
            m = {p: i for i, p in enumerate(phenotype_classes)}
            self.pheno = _as_category(self.df["phenotype"]).map(m).fillna(-1).astype(int).to_numpy()
        else:
            self.pheno = -np.ones(len(df), np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import os
        img = Image.open(os.path.join(self.root, self.df.at[i, "image_path"])).convert("RGB")
        return {
            "image": self.tf(img),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "meta": torch.from_numpy(self.meta[i]),
            "domain": torch.tensor(self.domains[i], dtype=torch.long),
            "pheno": torch.tensor(self.pheno[i], dtype=torch.long),
            "idx": i,
        }


def class_balanced_sampler(labels: np.ndarray, seed: int = 0):
    """2.7 class-balanced batch sampling: each class drawn with equal probability."""
    counts = np.bincount(labels, minlength=len(SEVERITY)).astype(float)
    w = 1.0 / counts[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels),
                                 replacement=True, generator=g)


def describe(df: pd.DataFrame, name: str = "") -> str:
    ct = pd.crosstab(df["severity"], df["fitz_group"]).reindex(index=SEVERITY, columns=FITZ_GROUPS, fill_value=0)
    return (f"{name}: {len(df)} images / {df.patient_id.nunique()} patients "
            f"(synthetic={int(df.synthetic.sum())})\n{ct}")
