"""Methods 2.6, 2.9, 2.10 — backbones, late metadata fusion, domain-adversarial head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

INPUT_SIZE = {"vgg16": 224, "mobilenet_v2": 224, "inception_v3": 299}


def build_backbone(name: str, pretrained: bool = True):
    """ImageNet-pretrained backbone with its classifier removed.
    Returns (module, feature_dim)."""
    name = name.lower()
    if name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[6] = nn.Identity()  # keep fc6/fc7 (+ their dropout) -> 4096-d
        return m, 4096
    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Identity()  # 1280-d
        return m, 1280
    if name == "inception_v3":
        if pretrained:
            m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
        else:
            m = tvm.inception_v3(weights=None, aux_logits=True, init_weights=True)
        m.aux_logits = False
        m.AuxLogits = None
        m.fc = nn.Identity()  # 2048-d
        return m, 2048
    raise ValueError(f"Unknown backbone {name}")


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd: float):
    return _GradReverse.apply(x, lambd)


class NormedLinear(nn.Module):
    """Cosine classifier used with LDAM (Cao et al., 2019)."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_f, out_f).uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5))

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class ADNet(nn.Module):
    """CNN image features ⊕ encoded clinical metadata -> 2 FC layers -> 3 classes.

    Optional heads (training only):
      * domain head behind a gradient-reversal layer (site/device), 2.10
      * phenotype head (dominant morphology), used only to build Fig 3D
    Logits are already multiplied by `scale` when the cosine classifier is used,
    so softmax(logits) gives probabilities for both CE and LDAM.
    """

    def __init__(self, backbone: str, n_classes=3, meta_dim=0, hidden=256, dropout=0.3,
                 cosine_classifier=False, scale=30.0, n_domains=0, n_phenotypes=0,
                 pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.meta_dim = meta_dim
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + meta_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.cosine = cosine_classifier
        self.scale = scale if cosine_classifier else 1.0
        self.cls = NormedLinear(hidden, n_classes) if cosine_classifier else nn.Linear(hidden, n_classes)
        self.domain_head = (nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
                                          nn.Dropout(dropout), nn.Linear(hidden, n_domains))
                            if n_domains > 1 else None)
        self.pheno_head = nn.Linear(hidden, n_phenotypes) if n_phenotypes > 0 else None
        self.feat_drop = nn.Dropout(dropout)

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x, meta=None, grl_lambda: float = 0.0):
        f = self.backbone(x)
        if isinstance(f, tuple):  # safety for inception outputs
            f = f[0]
        z = self.feat_drop(f)
        if self.meta_dim > 0:
            z = torch.cat([z, meta], dim=1)
        h = self.fuse(z)
        out = {"logits": self.cls(h) * self.scale, "features": f}
        if self.domain_head is not None and self.training:
            out["domain_logits"] = self.domain_head(grad_reverse(f, grl_lambda))
        if self.pheno_head is not None:
            out["pheno_logits"] = self.pheno_head(h)
        return out
