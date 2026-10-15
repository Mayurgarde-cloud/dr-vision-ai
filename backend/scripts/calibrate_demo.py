"""Fit the demo estimator's severity mapping on synthetic images.

    python -m scripts.calibrate_demo        (from the backend folder)

Prints coefficients to paste into app/services/demo_model.py (_B0 / _B) and the resulting
accuracy on a held-out set of synthetic images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import demo_model as dm  # noqa: E402
from ml.synthetic import make_fundus  # noqa: E402


def feats(seed0: int, n: int, size: int):
    X, y = [], []
    for sev in range(5):
        for i in range(n):
            f = dm.lesion_features(np.asarray(make_fundus(sev, seed=seed0 + i, size=size)))
            X.append(np.log1p([f["n_ma"], f["n_blot"], f["n_exu"], f["n_cws"]]))
            y.append(sev)
    return np.array(X), np.array(y, float)


def main():
    Xtr, ytr = feats(1000, 40, 512)
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    print("_B0 =", round(float(coef[0]), 4))
    print("_B  = np.array(%s)" % [round(float(c), 4) for c in coef[1:]])
    dm._B0, dm._B = float(coef[0]), coef[1:]
    for size in (512, 700, 300):
        Xte, yte = feats(5000, 25, size)
        pred = np.clip(dm._B0 + Xte @ dm._B, 0, 4).round()
        print(f"size {size}: exact acc {np.mean(pred == yte):.2f}, within-1 {np.mean(np.abs(pred - yte) <= 1):.2f}")


if __name__ == "__main__":
    main()
