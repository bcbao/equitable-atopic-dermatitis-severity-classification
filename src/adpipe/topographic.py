"""Results, Fig 2 A–D — topographic network representation of AD images.

The image luminance is treated as a surface. Key points are detected on the
surface (FAST corners — their number grows with texture/irregularity), and for
each key point we record:
  x, y  pixel position
  z     local surface height = detrended luminance (fine relief, about ±0.05)
  g     local surface gradient magnitude (a.u.), used for colour and size
Summary descriptors (key-point density, mean/95th-percentile gradient,
nearest-neighbour clustering) quantify "denser clustering and stronger
gradients" with increasing severity.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def surface_maps(img_bgr, size=600, sigma_fine=1.5, sigma_coarse=15.0):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    fine = cv2.GaussianBlur(L, (0, 0), sigma_fine)
    coarse = cv2.GaussianBlur(L, (0, 0), sigma_coarse)
    height = fine - coarse  # detrended relief (removes skin-tone/illumination level)
    gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return img, height, grad


def keypoints(img_bgr, size=600, fast_threshold=12, max_points=2500, seed=0):
    img, height, grad = surface_maps(img_bgr, size)
    # contrast-normalise so detection depends on texture, not on skin tone
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
    kps = fast.detect(L, None)
    pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    xi, yi = pts[:, 0].astype(int).clip(0, size - 1), pts[:, 1].astype(int).clip(0, size - 1)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": height[yi, xi], "g": grad[yi, xi],
            "height": height, "grad": grad, "size": size}


def topographic_descriptors(kp) -> dict:
    n = len(kp["x"])
    area = kp["size"] ** 2
    d = {"n_keypoints": n, "density_per_10k_px": 1e4 * n / area,
         "grad_mean": float(kp["grad"].mean()), "grad_p95": float(np.percentile(kp["grad"], 95)),
         "relief_sd": float(kp["height"].std()),
         "kp_grad_mean": float(kp["g"].mean()) if n else np.nan}
    if n > 3:
        tree = cKDTree(np.c_[kp["x"], kp["y"]])
        dist, _ = tree.query(np.c_[kp["x"], kp["y"]], k=2)
        nn = dist[:, 1].mean()
        expected = 0.5 / np.sqrt(n / area)  # Clark–Evans expectation under randomness
        d.update({"nn_dist_mean": float(nn), "clark_evans_R": float(nn / expected)})
    return d


def plot_topographic(ax3d, kp, vmax=None, cmap="YlOrRd", title="Topographic Network Map: 3D with Key Points"):
    g = kp["g"]
    vmax = vmax or max(float(np.percentile(g, 99)), 1e-6)
    sizes = 6 + 120 * np.clip(g / vmax, 0, 1) ** 1.5
    sc = ax3d.scatter(kp["x"], kp["y"], kp["z"], c=g, s=sizes, cmap=cmap, vmin=0, vmax=vmax,
                      alpha=0.8, edgecolors="none", depthshade=True)
    ax3d.set_xlabel("X-axis")
    ax3d.set_ylabel("Y-axis")
    ax3d.set_zlabel("Z-axis")
    ax3d.set_title(title, fontsize=9)
    ax3d.view_init(elev=22, azim=-55)
    return sc
