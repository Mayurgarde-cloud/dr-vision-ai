"""Heuristic *demo* estimator.

Used ONLY when no trained model (``models/dr_model.joblib``) or PyTorch is available, so the
website remains fully usable out of the box. It counts dark-red blobs (haemorrhages /
microaneurysms) and bright blobs (exudates / cotton-wool spots) and maps those counts to a
severity grade. The mapping was fitted on the *synthetic* images produced by
``ml/synthetic.py`` and has NO clinical validity – every result produced in this mode is
flagged ``mode="demo"`` in the API and shown with a banner in the UI.

Train a real model with ``python -m ml.train`` to switch to the ResNet-50 + DenseNet-121 pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

from ml.preprocess import crop_fundus, load_rgb

_S = 256

# severity ≈ B0 + B·[log1p(n_microaneurysm), log1p(n_blot), log1p(n_exudate), log1p(n_cotton)]
# (least-squares fit on synthetic images – see scripts/calibrate_demo.py)
_B0 = -0.4033
_B = np.array([0.3915, 0.1971, 0.4325, 0.4825])


def _components(mask: np.ndarray, min_area: int, max_area: int, max_aspect: float, min_fill: float):
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        fill = area / float(w * h)
        if aspect <= max_aspect and fill >= min_fill:
            out.append((float(cents[i][0]), float(cents[i][1]), int(area)))
    return out


def lesion_features(rgb: np.ndarray) -> dict:
    img = cv2.resize(crop_fundus(rgb), (_S, _S), interpolation=cv2.INTER_AREA)
    g = img[..., 1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fov = (gray > 14).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    k = int(_S * 0.05)
    inner = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

    # optic disc = brightest large blob → excluded from lesion search
    smooth = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 7)
    smooth[inner == 0] = 0
    _, _, _, (dx, dy) = cv2.minMaxLoc(smooth)
    disc = np.zeros_like(inner)
    cv2.circle(disc, (dx, dy), int(_S * 0.14), 1, -1)

    bg = cv2.medianBlur(g, 31).astype(np.float32)
    gf = g.astype(np.float32)
    valid = (inner > 0) & (disc == 0)

    dark = ((bg - gf) > 20) & valid
    bright = ((gf - bg) > 26) & valid

    dark_c = _components(dark, 2, 400, 2.2, 0.5)
    bright_c = _components(bright, 2, 900, 2.6, 0.4)
    ma = [c for c in dark_c if c[2] < 14]
    blot = [c for c in dark_c if c[2] >= 14]
    exu = [c for c in bright_c if c[2] < 60]
    cws = [c for c in bright_c if c[2] >= 60]
    return {
        "n_ma": len(ma), "n_blot": len(blot), "n_exu": len(exu), "n_cws": len(cws),
        "points": [(x, y, a) for x, y, a in dark_c + bright_c],
        "image": img, "fov": fov,
    }


def _severity(feat: dict) -> float:
    x = np.log1p([feat["n_ma"], feat["n_blot"], feat["n_exu"], feat["n_cws"]])
    return float(np.clip(_B0 + _B @ x, 0.0, 4.0))


def _heatmap(feat: dict) -> np.ndarray:
    h = np.zeros((_S, _S), np.float32)
    for x, y, a in feat["points"]:
        cv2.circle(h, (int(x), int(y)), 3, float(min(a, 60)) ** 0.5, -1)
    h = cv2.GaussianBlur(h, (0, 0), _S * 0.05)
    if h.max() < 1e-6:  # nothing found → gentle macular focus
        yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float32)
        h = np.exp(-(((xx - _S * 0.55) ** 2 + (yy - _S * 0.52) ** 2) / (2 * (_S * 0.14) ** 2)))
    h = h * (feat["fov"] > 0)
    return h / (h.max() + 1e-6)


def predict(pil_or_array) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities[5], heatmap in [0,1] of shape (256, 256))."""
    feat = lesion_features(load_rgb(pil_or_array))
    sev = _severity(feat)
    ks = np.arange(5)
    logits = -((ks - sev) ** 2) / (2 * 0.62 ** 2)
    p = np.exp(logits - logits.max())
    p = 0.97 * p / p.sum() + 0.03 / 5
    return p / p.sum(), _heatmap(feat)
