"""Image preprocessing shared by training and inference (no PyTorch dependency)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def load_rgb(src: ImageLike) -> np.ndarray:
    """Load an image path / PIL image / array as an RGB uint8 array."""
    if isinstance(src, np.ndarray):
        return src.astype(np.uint8)
    img = src if isinstance(src, Image.Image) else Image.open(src)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def crop_fundus(rgb: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Remove black margins around the circular fundus and pad to a square."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(ys) < 100:  # nothing recognisable – keep as is
        crop = rgb
    else:
        crop = rgb[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return canvas


def apply_clahe(rgb: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the L channel."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def prepare_image(src: ImageLike, size: int = 224, clahe: bool = True) -> np.ndarray:
    """Crop → (optional) CLAHE → resize to ``size`` x ``size``. Returns RGB uint8."""
    rgb = crop_fundus(load_rgb(src))
    if clahe:
        rgb = apply_clahe(rgb)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip / rotation / brightness-contrast jitter for a prepared (square) fundus image."""
    out = rgb
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1, :]
    h, w = out.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-25, 25)), 1.0)
    out = cv2.warpAffine(np.ascontiguousarray(out), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    alpha, beta = float(rng.uniform(0.85, 1.15)), float(rng.uniform(-12, 12))
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
