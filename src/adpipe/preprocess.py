"""Methods 2.4 — Preprocessing.

Shades-of-Gray colour constancy, DullRazor hair/artifact removal and anatomic
ROI localisation. Diffuse AD lesions are *cropped* (bounding box + margin),
not segmented, so surrounding clinically relevant skin is kept.
Final resizing to 224/299 + ImageNet normalisation happens in data.py.
"""
from __future__ import annotations

import cv2
import numpy as np


def shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004), Minkowski norm p."""
    img = img_bgr.astype(np.float32)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-8) * np.sqrt(3.0)
    out = img / (illum[None, None, :] + 1e-8)
    return np.clip(out, 0, 255).astype(np.uint8)


def dullrazor(img_bgr: np.ndarray, kernel_size: int = 17, thresh: int = 10,
              inpaint_radius: int = 6, max_mask_frac: float = 0.15):
    """DullRazor-style hair/artifact removal (Lee et al., 1997).

    Black-hat morphology finds thin dark structures (hair, pen marks), which are
    inpainted. Thin dark excoriations/crusts can also be picked up, so if the
    mask covers more than `max_mask_frac` of the image the original is returned
    unchanged (guard against erasing lesion morphology).
    Returns (clean_image, mask).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    if mask.mean() / 255.0 > max_mask_frac:
        return img_bgr.copy(), np.zeros_like(mask)
    clean = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return clean, mask


def auto_roi(img_bgr: np.ndarray, margin: float = 0.15, min_area_frac: float = 0.01):
    """Automatic lesion ROI: Otsu on the smoothed CIELAB a* (redness) channel.

    Returns a square bbox (x, y, w, h) around all inflamed regions plus margin.
    Falls back to the full image if nothing is found. Prefer manual ROI boxes
    (roi_x, roi_y, roi_w, roi_h columns) when available.
    """
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = cv2.GaussianBlur(lab[..., 1], (0, 0), sigmaX=max(H, W) / 200 + 1)
    _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area_frac * H * W]
    if not cnts:
        return square_bbox((0, 0, W, H), W, H)
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    dx, dy = int(w * margin), int(h * margin)
    return square_bbox((x - dx, y - dy, w + 2 * dx, h + 2 * dy), W, H)


def square_bbox(bbox, W: int, H: int):
    """Expand bbox to a square (as far as the image allows) and clip to bounds."""
    x, y, w, h = bbox
    side = min(max(w, h), W, H)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, int(side), int(side)


def preprocess_image(img_bgr: np.ndarray, roi=None, max_side: int = 512,
                     sog_p: int = 6, hair_removal: bool = True) -> np.ndarray:
    """Full preprocessing chain. Output is a square crop resized to `max_side`
    (kept larger than 299 so both 224 and 299 inputs can be derived later)."""
    img = shades_of_gray(img_bgr, p=sog_p)
    if hair_removal:
        img, _ = dullrazor(img)
    H, W = img.shape[:2]
    if roi is None or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in roi):
        x, y, w, h = auto_roi(img)
    else:
        x, y, w, h = square_bbox(tuple(int(v) for v in roi), W, H)
    crop = img[y:y + h, x:x + w]
    return cv2.resize(crop, (max_side, max_side), interpolation=cv2.INTER_AREA)
