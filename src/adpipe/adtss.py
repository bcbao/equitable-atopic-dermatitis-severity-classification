"""AD Topographic Severity Score (Results, Fig 2E).

AD-TSS = (0·P_mild + 1·P_moderate + 2·P_severe) / 2   ∈ [0, 1]
Categories (scores rounded to 2 decimals):
  mild-likely 0.00–0.33 · moderate-likely 0.34–0.66 · severe-likely 0.67–1.00
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ["mild-likely", "moderate-likely", "severe-likely"]


def ad_tss(probs: np.ndarray) -> np.ndarray:
    probs = np.atleast_2d(probs)
    return (0 * probs[:, 0] + 1 * probs[:, 1] + 2 * probs[:, 2]) / 2.0


def ad_tss_category(score) -> np.ndarray:
    s = np.round(np.atleast_1d(score), 2)
    return np.where(s <= 0.33, CATEGORIES[0], np.where(s <= 0.66, CATEGORIES[1], CATEGORIES[2]))
