"""Methods 2.7 — LDAM loss with Deferred Re-Weighting (Cao et al., NeurIPS 2019)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss.

    Expects *scaled* logits (s * cos θ) from ADNet. Margin for class j is
    m_j ∝ n_j^(-1/4), largest margin = max_m. The margin is applied as s*m_j.
    """

    def __init__(self, cls_num_list, max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=float)))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float32))
        self.s = s

    def forward(self, logits, target, weight=None):
        onehot = F.one_hot(target, logits.size(1)).bool()
        margin = (self.m_list[target] * self.s)[:, None]
        out = torch.where(onehot, logits - margin, logits)
        return F.cross_entropy(out, target, weight=weight)


def drw_weights(cls_num_list, beta: float = 0.9999) -> torch.Tensor:
    """Class-balanced weights from the effective number of samples (Cui et al., 2019),
    switched on after `drw_start_epoch` (deferred re-weighting)."""
    n = np.asarray(cls_num_list, dtype=float)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)
