#!/usr/bin/env python
"""Create a tiny fake dataset (random 'skin' images) to smoke-test the pipeline.
NOT real data — only checks that every script runs end-to-end."""
import os
import sys

import cv2
import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "dummy"
n_pat = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(0)
os.makedirs(f"{out}/raw", exist_ok=True)
SKIN = {1: (205, 220, 245), 2: (185, 205, 235), 3: (160, 185, 220), 4: (110, 145, 190), 5: (80, 110, 150), 6: (55, 75, 105)}
PHEN = {"mild": ["xerotic", "erythematous"], "moderate": ["excoriated", "lichenified"], "severe": ["lichenified", "exudative"]}
rows, qc = [], []
for p in range(n_pat):
    fitz = int(rng.choice([1, 2, 3, 3, 4, 5, 6]))
    sev = rng.choice(["mild", "moderate", "severe"], p=[0.42, 0.41, 0.17])
    split = "test" if p >= int(n_pat * 0.8) else "dev"
    site = "siteB" if split == "test" else "siteA"
    for j in range(int(rng.integers(2, 4))):
        h, w = int(rng.integers(500, 700)), int(rng.integers(500, 700))
        img = np.full((h, w, 3), SKIN[fitz], np.uint8)
        img = cv2.add(img, rng.normal(0, 8, img.shape).astype(np.int8), dtype=cv2.CV_8U)
        k = {"mild": 3, "moderate": 8, "severe": 16}[sev]
        for _ in range(k):  # red blotches + dark scratches ~ severity
            c = (int(rng.integers(100, w - 100)), int(rng.integers(100, h - 100)))
            cv2.circle(img, c, int(rng.integers(15, 60)), (60, 60, 200), -1)
            cv2.line(img, c, (c[0] + int(rng.integers(-60, 60)), c[1] + int(rng.integers(-60, 60))), (30, 30, 90), 2)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        name = f"P{p:03d}_{j}.jpg"
        cv2.imwrite(f"{out}/raw/{name}", img)
        rows.append({"image_path": name, "patient_id": f"P{p:03d}", "severity": sev, "fitzpatrick": fitz,
                     "split": split, "site": site, "device": rng.choice(["iphone", "samsung", "canon"]),
                     "age": rng.integers(2, 70) if rng.random() > 0.1 else np.nan,
                     "sex": rng.choice(["F", "M"]), "anatomic_site": rng.choice(["face", "arm", "leg", "trunk"]),
                     "disease_duration_months": rng.integers(1, 120),
                     "atopic_history": rng.choice(["yes", "no", ""]), "phenotype": rng.choice(PHEN[sev])})
        qc.append({"image_id": name, "patient_id": f"P{p:03d}", "excluded": 0, "exclusion_reason": ""})
for i, r in enumerate(["duplicate", "blur", "illumination", "no_active_lesion", "uncertain_diagnosis", "missing_metadata"]):
    qc.append({"image_id": f"X{i}.jpg", "patient_id": "PX", "excluded": 1, "exclusion_reason": r})
pd.DataFrame(rows).to_csv(f"{out}/metadata_raw.csv", index=False)
pd.DataFrame(qc).to_csv(f"{out}/qc_log.csv", index=False)
print(f"{len(rows)} fake images in {out}/raw")
