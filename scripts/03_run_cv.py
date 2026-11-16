#!/usr/bin/env python
"""Train one backbone × variant with patient-level 5-fold CV, then predict the
held-out external test set.

  python scripts/03_run_cv.py --backbone mobilenet_v2 --variant full
  python scripts/03_run_cv.py --backbone vgg16 --variant baseline

Outputs in runs/<backbone>__<variant>/ :
  cv_predictions.csv      out-of-fold predictions on the development set
  test_predictions.csv    held-out test predictions (ensemble of fold models or refit)
  fold*/model.pt, history.csv, meta_encoder.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import yaml

from adpipe.config import load_config
from adpipe.data import (check_no_leakage, describe, fold_frames, load_metadata,
                         make_patient_folds, split_inner_val)
from adpipe.metrics import PROB_COLS, all_metrics
from adpipe.train import get_device, predict_df, train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone", required=True, choices=["vgg16", "inception_v3", "mobilenet_v2"])
    ap.add_argument("--variant", default="full")
    ap.add_argument("--folds", default=None, help="comma list, e.g. 0,1 (default: all)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    a = ap.parse_args()

    cfg = load_config(a.config, a.variant, a.backbone, a.set)
    run_dir = os.path.join(cfg["output_dir"], f"{cfg['backbone']}__{cfg['variant']}")
    os.makedirs(run_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(run_dir, "config_used.yaml"), "w"), sort_keys=False)
    logf = open(os.path.join(run_dir, "train.log"), "a")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    device = get_device(cfg["device"])
    df = load_metadata(cfg["metadata_csv"])
    dev, test = df[df.split == "dev"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    check_no_leakage(dev, test)
    log(f"device={device}\n{describe(dev, 'development')}\n{describe(test, 'held-out test')}")

    folds = make_patient_folds(dev, cfg["n_folds"], cfg["seed"])
    which = range(len(folds)) if a.folds is None else [int(x) for x in a.folds.split(",")]
    cv_parts, test_parts = [], []
    for k in which:
        tr_pids, va_pids = folds[k]
        train, outer_val = fold_frames(dev, tr_pids, va_pids, cfg["use_synthetic"])
        if cfg["inner_val_frac"] > 0:
            train, inner_val = split_inner_val(train, cfg["inner_val_frac"], cfg["seed"] + k)
        else:
            inner_val = outer_val
        log(f"\n=== fold {k} ===  train {len(train)} (synthetic {int(train.synthetic.sum())}) | "
            f"inner-val {len(inner_val)} | outer-val {len(outer_val)}")
        fdir = os.path.join(run_dir, f"fold{k}")
        model, ctx, T, _ = train_fold(cfg, train, inner_val, cfg["image_root"], fdir, device, log)
        pv = predict_df(model, ctx, outer_val, cfg["image_root"], device, T)
        pv["fold"] = k
        cv_parts.append(pv)
        m = all_metrics(pv.label, pv[PROB_COLS].to_numpy())
        log(f"fold {k}: macro-F1 {m['macro_f1']:.4f}  macro-AUROC {m['macro_auroc']:.4f}  "
            f"ECE {m['ece']:.4f}  T={T:.3f}")
        if len(test) and cfg["test_mode"] == "ensemble":
            pt = predict_df(model, ctx, test, cfg["image_root"], device, T)
            pt["fold"] = k
            pt.to_csv(os.path.join(fdir, "test_predictions.csv"), index=False)
            test_parts.append(pt)

    cv = pd.concat(cv_parts, ignore_index=True)
    cv.to_csv(os.path.join(run_dir, "cv_predictions.csv"), index=False)

    if len(test):
        if cfg["test_mode"] == "refit":
            train_all = dev if cfg["use_synthetic"] else dev[dev.synthetic == 0]
            train_all, inner = split_inner_val(train_all.reset_index(drop=True), cfg["inner_val_frac"] or 0.15, cfg["seed"])
            log("\n=== refit on full development set ===")
            model, ctx, T, _ = train_fold(cfg, train_all, inner, cfg["image_root"],
                                          os.path.join(run_dir, "refit"), device, log)
            ens = predict_df(model, ctx, test, cfg["image_root"], device, T)
        else:
            ens = test_parts[0].copy()
            prob_stack = np.stack([p[PROB_COLS].to_numpy() for p in test_parts])
            ens[PROB_COLS] = prob_stack.mean(0)
            phc = [c for c in ens.columns if c.startswith("ph_")]
            if phc:
                ens[phc] = np.stack([p[phc].to_numpy() for p in test_parts]).mean(0)
            ens = ens.drop(columns=[c for c in ens.columns if c.startswith("logit_")] + ["fold", "temperature"])
        ens.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
        m = all_metrics(ens.label, ens[PROB_COLS].to_numpy())
        log(f"\nheld-out test: macro-F1 {m['macro_f1']:.4f}  macro-AUROC {m['macro_auroc']:.4f}  ECE {m['ece']:.4f}")
    log(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
