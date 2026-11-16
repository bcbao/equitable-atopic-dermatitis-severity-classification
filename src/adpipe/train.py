"""Methods 2.6–2.10 — training engine for one fold.

Progressive fine-tuning: `freeze_epochs` with frozen backbone, then up to
`finetune_epochs` with everything trainable. AdamW(lr, wd), batch 32,
dropout 0.3, early stopping on validation macro-F1 (patience 10, counted in
the fine-tuning phase). LDAM-DRW with re-weighting from `drw_start_epoch`.
Domain-adversarial head with GRL coefficient ramped 0 → grl_max.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import SEVERITY
from .data import ADDataset, LabelEncoder, MetaEncoder, class_balanced_sampler
from .losses import LDAMLoss, drw_weights
from .metrics import LOGIT_COLS, PROB_COLS, fit_temperature, macro_f1, softmax_np
from .models import INPUT_SIZE, ADNet


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grl_coeff(progress: float, grl_max: float) -> float:
    """DANN schedule: 0 → grl_max smoothly over training (Ganin & Lempitsky, 2015)."""
    return grl_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


class FoldContext:
    """Everything fitted on the training side of a fold (encoders, class counts)."""

    def __init__(self, cfg, train_df):
        self.cfg = cfg
        if cfg["use_metadata"]:
            self.meta_enc = MetaEncoder(cfg["meta_numeric"], cfg["meta_categorical"]).fit(train_df)
        else:
            self.meta_enc = MetaEncoder([], []).fit(train_df)
        self.dom_enc = LabelEncoder(train_df["domain"]) if cfg["domain_adaptation"] else LabelEncoder([])
        self.cls_counts = np.bincount(train_df["label"], minlength=len(SEVERITY)).clip(min=1)

    def dataset(self, df, image_root, train):
        size = INPUT_SIZE[self.cfg["backbone"]]
        meta = self.meta_enc.transform(df)
        dom = self.dom_enc.transform(df["domain"]) if len(self.dom_enc) else None
        phen = self.cfg["phenotype_classes"] if self.cfg.get("aux_phenotype") else None
        ds = ADDataset(df, image_root, size, train, meta=meta, domains=dom, phenotype_classes=phen)
        ds._pheno_classes = list(phen or [])
        return ds

    def build_model(self):
        c = self.cfg
        return ADNet(c["backbone"], n_classes=len(SEVERITY), meta_dim=self.meta_enc.dim,
                     hidden=c["hidden"], dropout=c["dropout"],
                     cosine_classifier=(c["loss"] == "ldam"), scale=c["ldam_s"],
                     n_domains=len(self.dom_enc),
                     n_phenotypes=len(c["phenotype_classes"]) if c.get("aux_phenotype") else 0,
                     pretrained=c["pretrained"])


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    logits, pheno, idx = [], [], []
    for b in loader:
        out = model(b["image"].to(device), b["meta"].to(device))
        logits.append(out["logits"].float().cpu().numpy())
        if "pheno_logits" in out:
            pheno.append(torch.softmax(out["pheno_logits"].float(), 1).cpu().numpy())
        idx.append(b["idx"].numpy())
    logits = np.concatenate(logits)
    df = loader.dataset.df.iloc[np.concatenate(idx)].copy()
    df[LOGIT_COLS] = logits
    df[PROB_COLS] = softmax_np(logits)
    if pheno:
        ph = np.concatenate(pheno)
        df[[f"ph_{p}" for p in model_pheno_classes(loader)]] = ph
    return df.reset_index(drop=True)


def model_pheno_classes(loader):
    return getattr(loader.dataset, "_pheno_classes", [])


def _loader(ds, cfg, train, sampler=None):
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(train and sampler is None),
                      sampler=sampler, num_workers=cfg["num_workers"], drop_last=False,
                      pin_memory=torch.cuda.is_available())


def train_fold(cfg, train_df, val_df, image_root, out_dir, device, log=print):
    """Train one model. `val_df` is used for early stopping and temperature
    scaling (an inner validation split, see run_cv). Returns (model, ctx, T, history)."""
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg["seed"])
    ctx = FoldContext(cfg, train_df)
    tr_ds = ctx.dataset(train_df, image_root, train=True)
    va_ds = ctx.dataset(val_df, image_root, train=False)
    sampler = class_balanced_sampler(tr_ds.labels, cfg["seed"]) if cfg["balanced_sampler"] else None
    tr_dl = _loader(tr_ds, cfg, True, sampler)
    va_dl = _loader(va_ds, cfg, False)

    model = ctx.build_model().to(device)
    if cfg["loss"] == "ldam":
        crit = LDAMLoss(ctx.cls_counts, max_m=cfg["ldam_max_m"], s=cfg["ldam_s"]).to(device)
    else:
        crit = None
    w_drw = drw_weights(ctx.cls_counts, cfg["drw_beta"]).to(device)

    total_epochs = cfg["freeze_epochs"] + cfg["finetune_epochs"]
    steps_total = total_epochs * len(tr_dl)
    step = 0
    best_f1, best_state, wait, history = -1.0, None, 0, []
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def make_opt():
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    model.set_backbone_trainable(False)
    opt = make_opt()
    for epoch in range(total_epochs):
        if epoch == cfg["freeze_epochs"]:
            model.set_backbone_trainable(True)
            opt = make_opt()
            wait = 0  # patience is counted within the fine-tuning phase
        weight = w_drw if (cfg["drw"] and epoch >= cfg["drw_start_epoch"]) else None
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for b in tr_dl:
            x, y = b["image"].to(device), b["label"].to(device)
            meta = b["meta"].to(device)
            lam = grl_coeff(step / max(steps_total, 1), cfg["grl_max"])
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x, meta, grl_lambda=lam)
                logits = out["logits"].float()
                loss = crit(logits, y, weight) if crit is not None else F.cross_entropy(logits, y, weight=weight)
                if "domain_logits" in out:
                    d = b["domain"].to(device)
                    if (d >= 0).any():
                        loss = loss + cfg["domain_loss_weight"] * F.cross_entropy(
                            out["domain_logits"].float(), d, ignore_index=-1)
                if "pheno_logits" in out:
                    ph = b["pheno"].to(device)
                    if (ph >= 0).any():
                        loss = loss + cfg["pheno_loss_weight"] * F.cross_entropy(
                            out["pheno_logits"].float(), ph, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            n += len(y)
            step += 1

        pv = predict(model, va_dl, device)
        f1 = macro_f1(pv.label, pv[PROB_COLS].to_numpy().argmax(1))
        phase = "frozen" if epoch < cfg["freeze_epochs"] else "finetune"
        history.append({"epoch": epoch + 1, "phase": phase, "train_loss": run_loss / max(n, 1),
                        "val_macro_f1": f1, "grl_lambda": lam, "drw": weight is not None,
                        "sec": time.time() - t0})
        log(f"  ep {epoch + 1:3d} [{phase}] loss {run_loss / max(n, 1):.4f}  val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state, wait = f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if phase == "finetune" and wait >= cfg["patience"]:
                log(f"  early stop at epoch {epoch + 1} (best val macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    pv = predict(model, va_dl, device)
    T = fit_temperature(pv[LOGIT_COLS].to_numpy(), pv.label.to_numpy()) if cfg["temperature_scaling"] else 1.0
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "temperature": T,
                "cls_counts": ctx.cls_counts.tolist(), "domains": ctx.dom_enc.classes},
               os.path.join(out_dir, "model.pt"))
    ctx.meta_enc.to_json(os.path.join(out_dir, "meta_encoder.json"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    return model, ctx, T, history


def predict_df(model, ctx, df, image_root, device, T=1.0):
    ds = ctx.dataset(df, image_root, train=False)
    out = predict(model, _loader(ds, ctx.cfg, False), device)
    out[PROB_COLS] = softmax_np(out[LOGIT_COLS].to_numpy(), T)
    out["temperature"] = T
    return out
