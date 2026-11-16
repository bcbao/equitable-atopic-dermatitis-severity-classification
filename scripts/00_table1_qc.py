#!/usr/bin/env python
"""Table 1 — quality-control and cohort-construction summary.

  python scripts/00_table1_qc.py --qc-log data/qc_log.csv --metadata data/metadata.csv \
      --out results/table1_qc.csv

qc_log.csv: one row per *collected* image with columns
  image_id, patient_id, excluded (0/1), exclusion_reason
exclusion_reason values (one per excluded image):
  duplicate | blur | illumination | no_active_lesion | uncertain_diagnosis | missing_metadata
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from adpipe.data import load_metadata

REASONS = [("duplicate", "Duplicate / near-duplicate images"),
           ("blur", "Severe blur / out-of-focus"),
           ("illumination", "Poor illumination / overexposure"),
           ("no_active_lesion", "No active AD lesion / post-treatment only"),
           ("uncertain_diagnosis", "Non-AD or uncertain diagnosis"),
           ("missing_metadata", "Missing severity or Fitzpatrick metadata")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-log", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="results/table1_qc.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    qc = pd.read_csv(a.qc_log)
    md = load_metadata(a.metadata)
    real = md[md.synthetic == 0]
    dev, test = real[real.split == "dev"], real[real.split == "test"]
    exc = qc[qc.excluded == 1]
    rows = [("Initial images collected", len(qc)),
            ("Images excluded after QC", len(exc)),
            ("Final analytic images", len(real)),
            ("Final patients", real.patient_id.nunique())]
    rows += [(lab, int((exc.exclusion_reason == key).sum())) for key, lab in REASONS]
    rows += [(f"{s.capitalize()} AD images retained", int((real.severity == s).sum()))
             for s in ("mild", "moderate", "severe")]
    rows += [("Fitzpatrick I–III images retained", int((real.fitz_group == "I-III").sum())),
             ("Fitzpatrick IV–VI images retained", int((real.fitz_group == "IV-VI").sum())),
             ("Development images / patients", f"{len(dev)} / {dev.patient_id.nunique()}"),
             ("Held-out test images / patients", f"{len(test)} / {test.patient_id.nunique()}"),
             ("Patient overlap across splits", len(set(dev.patient_id) & set(test.patient_id))),
             ("Synthetic images in validation/test sets", int((md[md.split == "test"].synthetic == 1).sum()))]
    out = pd.DataFrame(rows, columns=["QC domain", "Result"])
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    if len(qc) - len(exc) != len(real):
        print(f"\nWARNING: qc_log keeps {len(qc) - len(exc)} images but metadata has {len(real)}")
    other = set(exc.exclusion_reason) - {k for k, _ in REASONS}
    if other:
        print(f"WARNING: unrecognised exclusion reasons: {other}")


if __name__ == "__main__":
    main()
