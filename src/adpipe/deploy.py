"""Methods 2.12 — load a trained checkpoint for export/deployment."""
from __future__ import annotations

import os

import torch

from . import SEVERITY
from .data import MetaEncoder
from .models import ADNet


class DeployWrapper(torch.nn.Module):
    """image (+ meta) -> calibrated class probabilities. Training-only heads are dropped."""

    def __init__(self, model: ADNet, temperature: float):
        super().__init__()
        self.model = model
        self.t = float(temperature)

    def forward(self, image, meta):
        return torch.softmax(self.model(image, meta)["logits"] / self.t, dim=1)


def load_checkpoint(fold_dir: str, map_location="cpu"):
    ck = torch.load(os.path.join(fold_dir, "model.pt"), map_location=map_location, weights_only=False)
    cfg = ck["cfg"]
    meta_enc = MetaEncoder.from_json(os.path.join(fold_dir, "meta_encoder.json"))
    model = ADNet(cfg["backbone"], n_classes=len(SEVERITY), meta_dim=meta_enc.dim, hidden=cfg["hidden"],
                  dropout=cfg["dropout"], cosine_classifier=(cfg["loss"] == "ldam"), scale=cfg["ldam_s"],
                  n_domains=len(ck.get("domains", [])),
                  n_phenotypes=len(cfg["phenotype_classes"]) if cfg.get("aux_phenotype") else 0,
                  pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta_enc, cfg, ck.get("temperature", 1.0)
