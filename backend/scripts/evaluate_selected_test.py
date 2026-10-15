"""Run the one-time, independent test evaluation for a selected bundle.

Run only after validation selection is complete. This script reads the test cache,
does not fit any model component, and writes a report beside the selected bundle.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from ml import NUM_CLASSES
from ml.bundle import load_bundle, predict_proba
from ml.dataset import assert_content_disjoint
from ml.metrics import compute_metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/experiment_aug1"))
    parser.add_argument("--splits", type=Path, default=Path("models/splits.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/experiment_svm_v1"))
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model_dir / "dr_model.joblib")
    labels = json.loads(args.splits.read_text(encoding="utf-8"))["labels"]
    with np.load(args.cache_dir / "features_test.npz", allow_pickle=False) as z:
        paths = [str(Path(p).resolve()) for p in z["paths"].tolist()]
        features = z["feats"].astype(np.float32)
        cache_meta = (bool(z["pretrained"]), int(z["size"]), bool(z["clahe"]))
    meta = bundle["meta"]
    if cache_meta != (bool(meta["pretrained"]), int(meta["image_size"]), bool(meta["clahe"])):
        raise ValueError("Test feature metadata does not match selected model preprocessing.")
    if set(paths) & set(json.loads(args.splits.read_text(encoding="utf-8"))["train"] +
                        json.loads(args.splits.read_text(encoding="utf-8"))["val"]):
        raise ValueError("Test paths overlap train/validation paths.")
    split_data = json.loads(args.splits.read_text(encoding="utf-8"))
    assert_content_disjoint(split_data["train"] + split_data["val"], paths,
                            "training/validation", "test")
    y_test = np.asarray([labels[p] for p in paths], dtype=np.int64)
    started = time.perf_counter()
    probabilities = predict_proba(bundle, features)
    inference_seconds = time.perf_counter() - started
    metrics = compute_metrics(y_test, probabilities.argmax(axis=1))
    metrics.update({
        "bundle_version": meta.get("model_version", "unknown"),
        "evaluation": "independent test, evaluated once after validation model selection",
        "config": {
            "pretrained": bool(meta["pretrained"]), "clahe": bool(meta["clahe"]),
            "image_size": int(meta["image_size"]), "classifier": meta.get("classifier"),
            "pca_components": int(meta["pca_components"]),
            "C": getattr(getattr(bundle["clf"], "estimator", None), "C", None),
            "class_weight": getattr(getattr(bundle["clf"], "estimator", None), "class_weight", None),
        },
        "split": {"train": int(meta["n_train"]), "val": int(meta["n_val"]), "test": int(len(y_test))},
        "inference_seconds": inference_seconds,
        "inference_ms_per_image": inference_seconds * 1000 / len(y_test),
        "log_loss": float(log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - np.eye(NUM_CLASSES)[y_test]) ** 2, axis=1))),
    })
    out = args.model_dir / "test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved independent test report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
