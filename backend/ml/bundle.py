"""Saved model bundle helpers (scaler → PCA → AdaBoost/Random-Forest, plus a CAM head)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import NUM_CLASSES


def load_bundle(path: Path | str) -> dict:
    return joblib.load(path)


def save_bundle(bundle: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def predict_proba(bundle: dict, fused_features: np.ndarray) -> np.ndarray:
    """Fused 3072-d features -> (N, 5) class probabilities (classes unseen in training get 0)."""
    features = np.asarray(fused_features)
    if features.ndim != 2 or features.shape[1] != 3072:
        raise ValueError(f"Expected fused features with shape (N, 3072), got {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("Fused features contain NaN or infinite values.")
    z = bundle["pca"].transform(bundle["scaler"].transform(features))
    clf = bundle["clf"]
    classes = np.asarray(clf.classes_)
    if classes.ndim != 1 or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("Classifier classes must be a one-dimensional array of integer class indices.")
    if len(np.unique(classes)) != len(classes) or np.any((classes < 0) | (classes >= NUM_CLASSES)):
        raise ValueError(f"Classifier classes must be unique indices from 0 to {NUM_CLASSES - 1}.")
    proba = np.asarray(clf.predict_proba(z), dtype=np.float64)
    if proba.shape != (features.shape[0], len(classes)):
        raise ValueError(f"Classifier returned scores with shape {proba.shape}; expected {(features.shape[0], len(classes))}.")
    if not np.isfinite(proba).all() or np.any(proba < 0):
        raise ValueError("Classifier scores contain NaN, infinite, or negative values.")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("Classifier scores must sum to 1 for each input image.")
    full = np.zeros((proba.shape[0], NUM_CLASSES), dtype=np.float64)
    full[:, classes.astype(int)] = proba
    return full


def cam_from_map(bundle: dict, res_map: np.ndarray, class_index: int) -> np.ndarray:
    """Class activation map for one image from the ResNet-50 layer4 map.

    ``res_map`` has shape (2048, h, w). The linear head stored in the bundle was fitted on the
    globally-pooled ResNet-50 features, so CAM_c(x, y) = sum_k W[c, k] * A_k(x, y) – which is
    mathematically what Grad-CAM reduces to for a global-average-pool + linear classifier.
    Returns an (h, w) array normalised to [0, 1].
    """
    w = np.asarray(bundle["cam_W"])[class_index]
    cam = np.tensordot(w, res_map, axes=([0], [0]))
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam
