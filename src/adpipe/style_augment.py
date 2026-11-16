"""Methods 2.8 — skin-tone-targeted generative augmentation.

Neural style transfer (Gatys et al., 2016) renders Fitzpatrick I–III training
images with the skin tone/texture statistics of Fitzpatrick IV–VI reference
skin, while a strong content loss preserves lesion morphology.

Morphology preservation:
  * style is taken only from shallow VGG19 layers (conv1_1, conv2_1, conv3_1),
    which carry colour and fine texture, not lesion shape;
  * content loss on conv3_2 + conv4_2 with high weight;
  * optimisation starts from the content image.
A structural-similarity check on edge maps flags outputs whose morphology
drifted; flagged and sampled outputs go to dermatologist review.

`reinhard_transfer` is a fast, non-neural fallback (LAB mean/std matching).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models as tvm

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# indices in torchvision vgg19.features
_LAYERS = {"conv1_1": 0, "conv2_1": 5, "conv3_1": 10, "conv3_2": 12, "conv4_1": 19, "conv4_2": 21}


class VGGFeatures(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features[:22]
        for i, m in enumerate(vgg):  # avg-pool gives smoother results (Gatys)
            if isinstance(m, torch.nn.MaxPool2d):
                vgg[i] = torch.nn.AvgPool2d(2, 2)
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.idx = {v: k for k, v in _LAYERS.items()}

    def forward(self, x):
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        out = {}
        for i, m in enumerate(self.vgg):
            x = m(x)
            if i in self.idx:
                out[self.idx[i]] = x
        return out


def _gram(f):
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def _to_tensor(img_bgr, size, device):
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).contiguous().to(device)


def _to_bgr(t):
    rgb = (t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def neural_style_transfer(content_bgr, style_bgr, net: VGGFeatures, size=384, steps=300,
                          content_weight=1.0, style_weight=1e4, tv_weight=1e-5,
                          style_layers=("conv1_1", "conv2_1", "conv3_1"),
                          content_layers=("conv3_2", "conv4_2"), device="cpu"):
    c = _to_tensor(content_bgr, size, device)
    s = _to_tensor(style_bgr, size, device)
    with torch.no_grad():
        fc, fs = net(c), net(s)
        c_tgt = {l: fc[l] for l in content_layers}
        s_tgt = {l: _gram(fs[l]) for l in style_layers}
    x = c.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=steps, lr=1.0)

    def closure():
        opt.zero_grad()
        f = net(x)
        lc = sum(F.mse_loss(f[l], c_tgt[l]) for l in content_layers)
        ls = sum(F.mse_loss(_gram(f[l]), s_tgt[l]) for l in style_layers)
        tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        loss = content_weight * lc + style_weight * ls + tv_weight * tv
        loss.backward()
        return loss

    opt.step(closure)
    return _to_bgr(x.detach())


def reinhard_transfer(content_bgr, style_bgr):
    """Fast LAB colour statistics transfer (Reinhard et al., 2001)."""
    c = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = cv2.cvtColor(cv2.resize(style_bgr, content_bgr.shape[1::-1]), cv2.COLOR_BGR2LAB).astype(np.float32)
    cm, cs = c.reshape(-1, 3).mean(0), c.reshape(-1, 3).std(0) + 1e-6
    sm, ss = s.reshape(-1, 3).mean(0), s.reshape(-1, 3).std(0) + 1e-6
    out = (c - cm) / cs * ss + sm
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def edge_ssim(a_bgr, b_bgr, size=256):
    """SSIM between Sobel edge maps of two images (morphology preservation check)."""
    def edges(img):
        g = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)  # remove global tone difference
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return cv2.magnitude(gx, gy)
    x, y = edges(a_bgr), edges(b_bgr)
    rng = max(x.max(), y.max()) or 1.0
    C1, C2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x ** 2
    syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y ** 2
    sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return float(ssim.mean())
