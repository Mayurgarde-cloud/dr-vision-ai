"""Deep feature extraction with ResNet-50 and DenseNet-121 (ImageNet pre-trained).

Both backbones are used as frozen feature extractors. For every image we keep

* the global-average-pooled ResNet-50 vector   (2048-d),
* the global-average-pooled DenseNet-121 vector (1024-d),

and fuse them by concatenation (3072-d). The ResNet-50 ``layer4`` feature map is also
returned so a class-activation map (Grad-CAM equivalent) can be computed.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

from .preprocess import prepare_image

RES_DIM, DENSE_DIM = 2048, 1024
FUSED_DIM = RES_DIM + DENSE_DIM
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


class FeatureExtractor:
    def __init__(self, pretrained: bool = True, device: str | None = None):
        import torch
        from torchvision import models

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            # Large default OpenMP pools can make single-image inference slower
            # through oversubscription, especially on shared Windows hosts.
            threads = max(1, int(os.getenv("DR_TORCH_THREADS", min(4, os.cpu_count() or 1))))
            torch.set_num_threads(threads)
        if not pretrained:
            # Random weights must be identical in every process (train / evaluate / serve),
            # otherwise the saved classifier would see a different feature space.
            torch.manual_seed(0)
        res_w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        den_w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=res_w)
        densenet = models.densenet121(weights=den_w)
        self.resnet_body = torch.nn.Sequential(*list(resnet.children())[:-2]).eval().to(self.device)
        self.dense_body = densenet.features.eval().to(self.device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.pretrained = pretrained

    # ------------------------------------------------------------------ helpers
    def to_tensor(self, batch_uint8: np.ndarray):
        """NHWC uint8 RGB -> normalised NCHW float tensor on the device."""
        t = self.torch.from_numpy(batch_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (t - self.mean) / self.std

    def forward_maps(self, batch_uint8: np.ndarray):
        """Return (resnet_map [B,2048,h,w], densenet_map [B,1024,h,w]) as numpy arrays."""
        F = self.torch.nn.functional
        with self.torch.no_grad():
            x = self.to_tensor(batch_uint8)
            res = self.resnet_body(x)
            den = F.relu(self.dense_body(x), inplace=False)
        return res.cpu().numpy(), den.cpu().numpy()

    @staticmethod
    def fuse(res_map: np.ndarray, den_map: np.ndarray) -> np.ndarray:
        return np.concatenate([res_map.mean(axis=(2, 3)), den_map.mean(axis=(2, 3))], axis=1)

    def extract_batch(self, batch_uint8: np.ndarray) -> np.ndarray:
        res, den = self.forward_maps(batch_uint8)
        return self.fuse(res, den)

    def extract_images(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Features for an (N, H, W, 3) uint8 array of already-prepared images."""
        feats = [self.extract_batch(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, FUSED_DIM), dtype=np.float32)

    def extract_paths(self, paths: Sequence[str], size: int = 224, clahe: bool = True, batch_size: int = 16,
                      progress: bool = True) -> np.ndarray:
        feats = []
        n = len(paths)
        for i in range(0, n, batch_size):
            imgs = np.stack([prepare_image(p, size, clahe) for p in paths[i:i + batch_size]])
            feats.append(self.extract_batch(imgs))
            if progress:
                print(f"  extracted {min(i + batch_size, n)}/{n}", end="\r", flush=True)
        if progress:
            print()
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, FUSED_DIM), dtype=np.float32)
