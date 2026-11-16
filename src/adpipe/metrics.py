"""Methods 2.11 — discrimination, calibration, reclassification and fairness metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from . import FITZ_GROUPS, SEVERITY

K = len(SEVERITY)
PROB_COLS = [f"p_{s}" for s in SEVERITY]
LOGIT_COLS = [f"logit_{s}" for s in SEVERITY]


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)


def macro_auroc(y, probs):
    """One-vs-rest macro AUROC; classes absent from y are skipped."""
    aucs = per_class_auroc(y, probs)
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def per_class_auroc(y, probs):
    out = {}
    for k, s in enumerate(SEVERITY):
        yk = (np.asarray(y) == k).astype(int)
        out[s] = roc_auc_score(yk, probs[:, k]) if 0 < yk.sum() < len(yk) else np.nan
    return out


def sens_spec(y, pred):
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    out = {}
    for k, s in enumerate(SEVERITY):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[f"sens_{s}"] = tp / (tp + fn) if tp + fn else np.nan
        out[f"spec_{s}"] = tn / (tn + fp) if tn + fp else np.nan
    return out


def ece(y, probs, n_bins: int = 15):
    """Top-label expected calibration error (Guo et al., 2017), equal-width bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability_table(y, probs, n_bins: int = 15) -> pd.DataFrame:
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "confidence": conf[m].mean() if m.any() else np.nan,
                     "accuracy": correct[m].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def brier(y, probs):
    """Multiclass Brier score: mean over samples of Σ_k (p_k − y_k)²."""
    oh = np.eye(K)[np.asarray(y)]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def nri_multiclass(y, pred_old, pred_new):
    """Categorical NRI for a multiclass outcome (new vs old model).

    For each true class k: (proportion moved wrong→right) − (proportion moved
    right→wrong); NRI = average over classes. Range −1..1.
    Please state this definition in the Methods when reporting NRI.
    """
    y, po, pn = map(np.asarray, (y, pred_old, pred_new))
    parts = []
    for k in range(K):
        m = y == k
        if not m.any():
            continue
        up = np.mean((po[m] != k) & (pn[m] == k))
        down = np.mean((po[m] == k) & (pn[m] != k))
        parts.append(up - down)
    return float(np.mean(parts))


def all_metrics(y, probs) -> dict:
    y = np.asarray(y)
    pred = probs.argmax(1)
    d = {
        "n": len(y),
        "macro_f1": macro_f1(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_auroc": macro_auroc(y, probs),
        "ece": ece(y, probs),
        "brier": brier(y, probs),
    }
    d.update({f"auroc_{k}": v for k, v in per_class_auroc(y, probs).items()})
    d.update(sens_spec(y, pred))
    return d


def fairness_metrics(df: pd.DataFrame, prob_cols=PROB_COLS) -> dict:
    """Per Fitzpatrick group metrics + absolute gaps (I–III minus IV–VI)."""
    out = {}
    per = {}
    for g in FITZ_GROUPS:
        sub = df[df.fitz_group == g]
        if len(sub) == 0:
            continue
        per[g] = all_metrics(sub.label.to_numpy(), sub[prob_cols].to_numpy())
        for k, v in per[g].items():
            out[f"{g}:{k}"] = v
    if len(per) == 2:
        a, b = per[FITZ_GROUPS[0]], per[FITZ_GROUPS[1]]
        for k in ["macro_f1", "macro_auroc", "ece", "accuracy"]:
            out[f"gap_{k}"] = abs(a[k] - b[k])
    return out


# --------------------------------------------------------------------------- #
# Temperature scaling (Guo et al., 2017)
# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200,
                    t_min: float = 0.05, t_max: float = 20.0) -> float:
    """Single temperature T minimising validation NLL, clipped to [t_min, t_max]
    (a T at the bound usually means the validation set is too small or the
    model has not learned yet)."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().item())
    if not np.isfinite(t):
        return 1.0
    return float(np.clip(t, t_min, t_max))


def softmax_np(logits, t: float = 1.0):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
