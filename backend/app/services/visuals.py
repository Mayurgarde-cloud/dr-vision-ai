"""Heat-map rendering helpers."""
from __future__ import annotations

import cv2
import numpy as np


def colorize(cam01: np.ndarray, size: int) -> np.ndarray:
    """[0,1] activation map -> RGB JET heat-map of ``size`` x ``size``."""
    cam = cv2.resize(np.clip(cam01, 0, 1).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(rgb, 1 - alpha, heat_rgb, alpha, 0)


def looks_like_fundus(rgb: np.ndarray) -> bool:
    """Cheap plausibility check: dark corners and red-dominant colouring."""
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    corners = np.concatenate([small[:8, :8].reshape(-1, 3), small[:8, -8:].reshape(-1, 3),
                              small[-8:, :8].reshape(-1, 3), small[-8:, -8:].reshape(-1, 3)])
    dark_corners = corners.mean() < 45
    r, g = small[..., 0].mean(), small[..., 1].mean()
    red_dominant = r > 1.25 * g
    return bool(dark_corners and red_dominant)
