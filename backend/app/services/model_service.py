"""Inference service: trained ResNet-50 + DenseNet-121 → PCA → AdaBoost/RF pipeline, or demo fallback."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .. import config
from ml import features as feat_mod
from ml.bundle import cam_from_map, load_bundle, predict_proba
from ml.preprocess import crop_fundus, load_rgb, prepare_image

from . import demo_model

log = logging.getLogger("dr_vision.model")
DISPLAY_SIZE = 384


@dataclass
class Prediction:
    probabilities: np.ndarray   # shape (5,), sums to 1
    class_index: int
    confidence: float           # 0-1
    heatmap: np.ndarray         # (DISPLAY_SIZE, DISPLAY_SIZE) in [0, 1]
    display_rgb: np.ndarray     # (DISPLAY_SIZE, DISPLAY_SIZE, 3) uint8 – the cropped, un-enhanced fundus
    mode: str                   # "trained" | "demo"


class ModelService:
    def __init__(self) -> None:
        self.mode = "demo"
        self.reason: Optional[str] = None
        self.reason_code: Optional[str] = None   # no_model | no_torch | load_error | random_weights
        self.bundle: Optional[dict] = None
        self.extractor = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self.mode, self.bundle, self.extractor = "demo", None, None
        self.reason_code = None
        if not config.MODEL_PATH.exists():
            self.reason_code = "no_model"
            self.reason = "No trained model found. Train one with: python -m ml.train"
        elif not feat_mod.torch_available():
            self.reason_code = "no_torch"
            self.reason = "PyTorch / torchvision are not installed, so the trained model cannot run."
        else:
            try:
                self.bundle = load_bundle(config.MODEL_PATH)
                meta = self.bundle["meta"]
                if meta.get("classifier"):
                    config.MODEL_CLASSIFIER = str(meta["classifier"])
                self.extractor = feat_mod.FeatureExtractor(pretrained=bool(meta.get("pretrained", True)))
                self.mode, self.reason = "trained", None
                if not meta.get("pretrained", True):
                    self.reason_code = "random_weights"
                    self.reason = "Model was trained with random backbone weights (smoke-test only)."
                log.info("Loaded trained model from %s", config.MODEL_PATH)
            except Exception as exc:  # corrupt / incompatible pickle, missing weights, ...
                log.exception("Could not load trained model")
                self.bundle, self.extractor = None, None
                self.reason_code = "load_error"
                self.reason = "The trained model could not be loaded. Check server logs for details."
        if self.mode == "demo":
            log.warning("Running in DEMO mode: %s", self.reason)

    @property
    def is_trained(self) -> bool:
        return self.mode == "trained"

    @property
    def image_size(self) -> int:
        return int(self.bundle["meta"]["image_size"]) if self.bundle else config.IMAGE_SIZE

    # --------------------------------------------------------------- inference
    def predict(self, img: Image.Image) -> Prediction:
        rgb = load_rgb(img)
        display = cv2.resize(crop_fundus(rgb), (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)

        if self.is_trained:
            meta = self.bundle["meta"]
            prepared = prepare_image(rgb, meta["image_size"], meta["clahe"])
            with self._lock:
                res_map, den_map = self.extractor.forward_maps(prepared[None])
            probs = predict_proba(self.bundle, self.extractor.fuse(res_map, den_map))[0]
            cls = int(probs.argmax())
            cam = cam_from_map(self.bundle, res_map[0], cls)
            heat = cv2.resize(cam.astype(np.float32), (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
            mode = "trained"
        else:
            probs, heat = demo_model.predict(rgb)
            heat = cv2.resize(heat, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
            cls = int(np.argmax(probs))
            mode = "demo"

        return Prediction(probabilities=np.asarray(probs, dtype=float), class_index=cls,
                          confidence=float(probs[cls]), heatmap=np.clip(heat, 0, 1),
                          display_rgb=display, mode=mode)


service = ModelService()
