"""Quality-control helpers for synthetic images (Section 2.8)."""
from __future__ import annotations

import cv2
import numpy as np


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
