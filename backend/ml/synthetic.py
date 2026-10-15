"""Synthetic retinal fundus image generator.

This produces *illustrative* fundus-like images (optic disc, macula, vessel tree and
severity-dependent lesions). They are used for:

* the hero / severity-class artwork on the website,
* the optional 231-image demo dataset (``scripts/make_sample_dataset.py``),
* smoke-testing the training / inference pipeline without a real dataset.

They are NOT clinical images and must never be used to judge real-world accuracy.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

# Lesion counts per severity (at 512 px; scaled for other sizes).
_LESIONS = {
    0: dict(ma=(0, 1), blot=(0, 0), exu=(0, 0), cws=(0, 0), neo=0, pre=0.0),
    1: dict(ma=(5, 9), blot=(0, 1), exu=(0, 1), cws=(0, 0), neo=0, pre=0.0),
    2: dict(ma=(8, 14), blot=(7, 12), exu=(5, 10), cws=(0, 1), neo=0, pre=0.0),
    3: dict(ma=(10, 18), blot=(22, 32), exu=(16, 26), cws=(3, 6), neo=0, pre=0.15),
    4: dict(ma=(12, 20), blot=(30, 46), exu=(24, 40), cws=(4, 8), neo=4, pre=0.8),
}


def _noise(h: int, w: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (n.std() + 1e-6)


def _blend(img: np.ndarray, alpha: np.ndarray, color) -> None:
    a = np.clip(alpha, 0, 1)[..., None]
    img *= 1.0 - a
    img += a * np.asarray(color, dtype=np.float32)


def _draw_vessel(mask, p0, angle, length, width, rng, depth, S, bias):
    """Random-walk vessel with occasional branching."""
    x, y = p0
    step = max(2.0, 0.012 * S)
    n_steps = int(length / step)
    pts = [(x, y)]
    for i in range(n_steps):
        angle += rng.normal(0, 0.10) + bias * 0.012
        x += math.cos(angle) * step
        y += math.sin(angle) * step
        pts.append((x, y))
        if depth < 2 and i > 6 and rng.random() < 0.07:
            _draw_vessel(mask, (x, y), angle + rng.choice([-1, 1]) * rng.uniform(0.45, 0.8),
                         length * 0.5, width * 0.62, rng, depth + 1, S, bias)
    for i in range(len(pts) - 1):
        w = max(1, int(round(width * (1.0 - 0.55 * i / max(1, len(pts) - 1)))))
        cv2.line(mask, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])), 255, w, cv2.LINE_AA)


def make_fundus(severity: int = 0, seed: int = 0, size: int = 512, transparent: bool = False) -> Image.Image:
    """Return a synthetic fundus-like PIL image for the given DR severity (0-4)."""
    severity = int(max(0, min(4, severity)))
    rng = np.random.default_rng(seed * 7919 + severity * 104729 + 13)
    S = int(size)
    k = S / 512.0
    cx = cy = S / 2.0
    R = 0.47 * S
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = (xx - cx) / R, (yy - cy) / R
    r = np.sqrt(dx ** 2 + dy ** 2)

    tint = rng.uniform(-0.04, 0.04, 3).astype(np.float32)
    center = np.array([0.88, 0.40, 0.10], np.float32) + tint
    edge = np.array([0.42, 0.13, 0.04], np.float32) + tint * 0.4
    t = np.clip(r, 0, 1)[..., None] ** 2
    img = center * (1 - t) + edge * t

    # Choroidal mottling + fine grain
    img *= (1.0 + 0.07 * _noise(S, S, S / 38, rng))[..., None]
    img *= (1.0 + 0.025 * _noise(S, S, 1.0, rng))[..., None]

    # Macula (slightly darker) + fovea
    mx, my = 0.16, 0.04
    d2 = (dx - mx) ** 2 + (dy - my) ** 2
    img *= (1.0 - 0.32 * np.exp(-d2 / (2 * 0.075 ** 2)))[..., None]
    _blend(img, 0.25 * np.exp(-d2 / (2 * 0.012 ** 2)), (0.75, 0.30, 0.08))

    # Optic disc
    ox, oy = -0.52, -0.04
    do = np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)
    _blend(img, 0.35 * np.exp(-do ** 2 / (2 * 0.20 ** 2)), (0.98, 0.55, 0.20))
    _blend(img, np.clip((0.115 - do) / 0.02, 0, 1) * 0.95, (0.98, 0.80, 0.42))
    _blend(img, np.exp(-do ** 2 / (2 * 0.045 ** 2)) * 0.8, (1.0, 0.93, 0.70))

    # Vessel tree
    vm = np.zeros((S, S), np.uint8)
    dpx = (cx + ox * R, cy + oy * R)
    for ang_deg, ln, w, bias in [(-62, 0.95, 4.2, 1), (-24, 1.15, 3.6, 1), (22, 1.15, 3.6, -1), (60, 0.95, 4.2, -1),
                                 (-150, 0.55, 3.0, -1), (150, 0.55, 3.0, 1)]:
        ang = math.radians(ang_deg + rng.uniform(-8, 8))
        _draw_vessel(vm, dpx, ang, ln * R, max(1.0, w * k), rng, 0, S, bias)
    vm_f = cv2.GaussianBlur(vm.astype(np.float32) / 255.0, (0, 0), 0.9 * max(k, 0.6))
    vm_f[do < 0.11] *= 0.2
    _blend(img, 0.80 * vm_f, (0.40, 0.05, 0.04))
    _blend(img, 0.18 * cv2.GaussianBlur(vm_f, (0, 0), 2.0 * max(k, 0.6)), (0.95, 0.55, 0.30))

    # Lesions
    cfg = _LESIONS[severity]

    def rand_pos(min_do=0.17, rmax=0.86):
        for _ in range(60):
            a, rr = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0.01, rmax ** 2))
            px, py = rr * math.cos(a), rr * math.sin(a)
            if math.hypot(px - ox, py - oy) > min_do:
                return cx + px * R, cy + py * R
        return cx, cy

    def count(rng_pair):
        lo, hi = rng_pair
        return int(rng.integers(lo, hi + 1))

    ma_m = np.zeros((S, S), np.float32)
    for _ in range(count(cfg["ma"])):
        x, y = rand_pos()
        cv2.circle(ma_m, (int(x), int(y)), max(1, int(round(rng.uniform(1.2, 2.2) * k))), 1.0, -1, cv2.LINE_AA)
    _blend(img, 0.9 * cv2.GaussianBlur(ma_m, (0, 0), 0.6), (0.30, 0.03, 0.03))

    blot_m = np.zeros((S, S), np.float32)
    for _ in range(count(cfg["blot"])):
        x, y = rand_pos()
        cv2.circle(blot_m, (int(x), int(y)), max(2, int(round(rng.uniform(3.5, 8.5) * k))), 1.0, -1, cv2.LINE_AA)
    _blend(img, 0.85 * cv2.GaussianBlur(blot_m, (0, 0), 1.1 * max(k, 0.7)), (0.33, 0.03, 0.03))

    exu_m = np.zeros((S, S), np.float32)
    n_exu = count(cfg["exu"])
    for i in range(n_exu):
        if i % 3 == 0:   # ring/cluster around the macula
            a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0.10, 0.32)
            x, y = cx + (mx + rr * math.cos(a)) * R, cy + (my + rr * math.sin(a)) * R
        else:
            x, y = rand_pos()
        cv2.circle(exu_m, (int(x), int(y)), max(1, int(round(rng.uniform(2, 5) * k))), 1.0, -1, cv2.LINE_AA)
    _blend(img, 0.9 * cv2.GaussianBlur(exu_m, (0, 0), 0.9 * max(k, 0.7)), (1.0, 0.92, 0.55))

    cws_m = np.zeros((S, S), np.float32)
    for _ in range(count(cfg["cws"])):
        x, y = rand_pos(0.25, 0.7)
        cv2.ellipse(cws_m, (int(x), int(y)), (int(rng.uniform(9, 16) * k) + 1, int(rng.uniform(5, 9) * k) + 1),
                    rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
    _blend(img, 0.55 * cv2.GaussianBlur(cws_m, (0, 0), 3.0 * max(k, 0.7)), (0.98, 0.90, 0.72))

    if cfg["neo"]:
        nv = np.zeros((S, S), np.uint8)
        for _ in range(cfg["neo"]):
            ang = rng.uniform(-math.pi, math.pi)
            _draw_vessel(nv, (dpx[0] + 0.06 * R * math.cos(ang), dpx[1] + 0.06 * R * math.sin(ang)), ang,
                         0.35 * R, max(1.0, 1.6 * k), rng, 1, S, rng.choice([-3, 3]))
        _blend(img, 0.85 * cv2.GaussianBlur(nv.astype(np.float32) / 255, (0, 0), 0.7), (0.75, 0.05, 0.05))
    if rng.random() < cfg["pre"]:
        x, y = rand_pos(0.3, 0.6)
        pm = np.zeros((S, S), np.float32)
        cv2.ellipse(pm, (int(x), int(y)), (int(rng.uniform(22, 34) * k) + 2, int(rng.uniform(14, 22) * k) + 2),
                    rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        _blend(img, 0.8 * cv2.GaussianBlur(pm, (0, 0), 3.0 * max(k, 0.7)), (0.25, 0.02, 0.02))

    # Vignette + circular field-of-view mask
    img *= (1.0 - 0.28 * np.clip(r, 0, 1) ** 3)[..., None]
    fov = np.clip((1.0 - r) / 0.015, 0, 1)
    img = np.clip(img, 0, 1) * fov[..., None]
    rgb = (img * 255).astype(np.uint8)
    if rng.random() < 0.3:
        rgb = rgb[:, ::-1].copy()
        fov = fov[:, ::-1]
    if transparent:
        rgba = np.dstack([rgb, (fov * 255).astype(np.uint8)])
        return Image.fromarray(rgba, "RGBA")
    return Image.fromarray(rgb, "RGB")
