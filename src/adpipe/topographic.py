"""Section 2.9 / Fig 2 A–D: topographic representation of lesion morphology.

Used for visualisation only; nothing here enters the classifier.

The skin surface is approximated by the log-luminance of the image
(CIELAB L*). Working in log space makes local contrast independent of the
baseline skin tone and illumination (a multiplicative change in brightness
becomes a constant offset), so light and dark skin are treated alike.

  height map  z(x, y) = fine-scale log-luminance minus coarse-scale
                         log-luminance (detrended relief, unitless)
  gradient    g(x, y) = magnitude of the Sobel gradient of the fine-scale
                         log-luminance (local surface gradient, a.u.)
  key points  local maxima of g within a 9 × 9 window whose value exceeds
              a fixed threshold (default 0.04), i.e. points of locally steep
              relief such as scale edges, excoriations and lichenified folds

Because the threshold is fixed rather than relative to each image, smooth
skin yields few key points and rough, lichenified or excoriated skin yields
many. Each key point is drawn in 3D at (x, y, z) and coloured and sized by g.
Summary descriptors quantify density, gradient strength and clustering.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def surface_maps(img_bgr, size=600, sigma_fine=1.5, sigma_coarse=15.0):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    logL = np.log(L + 0.05)
    fine = cv2.GaussianBlur(logL, (0, 0), sigma_fine)
    coarse = cv2.GaussianBlur(logL, (0, 0), sigma_coarse)
    height = fine - coarse
    gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return img, height, grad


def keypoints(img_bgr, size=600, grad_threshold=0.04, nms_window=9, border=8):
    img, height, grad = surface_maps(img_bgr, size)
    local_max = grad >= cv2.dilate(grad, np.ones((nms_window, nms_window), np.uint8))
    mask = local_max & (grad > grad_threshold)
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    return {"x": xs.astype(float), "y": ys.astype(float), "z": height[ys, xs], "g": grad[ys, xs],
            "height": height, "grad": grad, "size": size, "threshold": grad_threshold}


def topographic_descriptors(kp) -> dict:
    n = len(kp["x"])
    area = kp["size"] ** 2
    d = {"n_keypoints": n, "density_per_10k_px": 1e4 * n / area,
         "grad_mean": float(kp["grad"].mean()), "grad_p95": float(np.percentile(kp["grad"], 95)),
         "relief_sd": float(kp["height"].std()),
         "kp_grad_mean": float(kp["g"].mean()) if n else float("nan")}
    if n > 3:
        tree = cKDTree(np.c_[kp["x"], kp["y"]])
        dist, _ = tree.query(np.c_[kp["x"], kp["y"]], k=2)
        nn = dist[:, 1].mean()
        expected = 0.5 / np.sqrt(n / area)  # Clark–Evans expectation under complete spatial randomness
        d.update({"nn_dist_mean": float(nn), "clark_evans_R": float(nn / expected)})
    return d


def plot_topographic(ax3d, kp, vmax=None, zlim=0.3, cmap="YlOrRd", title="3D topographic map with key points"):
    g = kp["g"]
    vmax = vmax or max(float(np.percentile(g, 99)) if len(g) else 0.1, 1e-6)
    sizes = 6 + 110 * np.clip(g / vmax, 0, 1) ** 1.5
    sc = ax3d.scatter(kp["x"], kp["y"], kp["z"], c=g, s=sizes, cmap=cmap, vmin=0, vmax=vmax,
                      alpha=0.8, edgecolors="none", depthshade=True)
    ax3d.set_xlim(0, kp["size"])
    ax3d.set_ylim(0, kp["size"])
    ax3d.set_zlim(-zlim, zlim)
    ax3d.set_xlabel("X (px)")
    ax3d.set_ylabel("Y (px)")
    ax3d.set_zlabel("Relief (a.u.)", labelpad=2)
    ax3d.set_title(title, fontsize=9)
    ax3d.view_init(elev=22, azim=-55)
    return sc
