"""Re-evaluate the saved model on the held-out test split and refresh ``models/metrics.json``.

    python -m ml.evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml.bundle import load_bundle, predict_proba  # noqa: E402
from ml.dataset import assert_content_disjoint, read_csv_labels  # noqa: E402
from ml.features import FeatureExtractor, torch_available  # noqa: E402
from ml.metrics import compute_metrics  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(BACKEND_DIR / "models" / "candidate"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-csv", help="Optional external test labels, supplied only after model selection.")
    ap.add_argument("--test-image-dir", help="Image directory for --test-csv.")
    args = ap.parse_args(argv)
    mdir = Path(args.model_dir)

    report_path = mdir / "test_metrics.json"
    if report_path.exists():
        print(f"Test evaluation already exists at {report_path}; refusing to reuse the test set.")
        return 2
    metrics_path = mdir / "metrics.json"
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if "n_samples" in previous and "confusion_matrix" in previous:
            print("Metrics already contain test results; refusing to evaluate the test set again.")
            return 2

    if not torch_available():
        print("PyTorch / torchvision are required for evaluation.")
        return 2
    if not (mdir / "dr_model.joblib").exists() or not (mdir / "splits.json").exists():
        print("No trained model / splits found. Run `python -m ml.train` first.")
        return 1

    bundle = load_bundle(mdir / "dr_model.joblib")
    splits = json.loads((mdir / "splits.json").read_text())
    meta = bundle["meta"]
    if args.test_csv:
        if not args.test_image_dir:
            print("--test-image-dir is required with --test-csv.")
            return 2
        test_df = read_csv_labels(args.test_csv, args.test_image_dir)
        paths = test_df["path"].tolist()
        y_true = test_df["label"].to_numpy(dtype=int)
        train_val = set(splits.get("train", []) + splits.get("val", []))
        if train_val & set(paths):
            print("Test paths overlap the training or validation split.")
            return 1
    else:
        paths = splits.get("test", [])
        y_true = np.array([splits["labels"][p] for p in paths])
    missing = [p for p in paths if not Path(p).is_file()]
    if not paths or missing:
        print(f"Test images are missing ({len(missing)} of {len(paths)}). No partial test evaluation was run.")
        return 1
    try:
        assert_content_disjoint(splits.get("train", []) + splits.get("val", []), paths,
                                "training/validation", "test")
    except ValueError as exc:
        print(f"Refusing test evaluation: {exc}")
        return 1
    extractor = FeatureExtractor(pretrained=meta["pretrained"])
    extraction_started = time.perf_counter()
    feats = extractor.extract_paths(paths, size=meta["image_size"], clahe=meta["clahe"], batch_size=args.batch_size)
    extraction_seconds = time.perf_counter() - extraction_started
    inference_started = time.perf_counter()
    probabilities = predict_proba(bundle, feats)
    classifier_seconds = time.perf_counter() - inference_started
    y_pred = probabilities.argmax(1)

    metrics = compute_metrics(y_true, y_pred)
    prev = previous if metrics_path.exists() else {}
    for key in ("validation_metrics", "split", "config"):
        if key in prev:
            metrics[key] = prev[key]
    metrics.update({
        "evaluation": "independent test, evaluated once after validation model selection",
        "bundle_version": meta.get("model_version", "unknown"),
        "selected_for_test": True,
        "feature_extraction_seconds": extraction_seconds,
        "classifier_inference_seconds": classifier_seconds,
        "classifier_inference_ms_per_image": classifier_seconds * 1000 / len(paths),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(5)))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - np.eye(5)[y_true]) ** 2, axis=1))),
    })
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test images: {len(paths)}")
    print(f"Accuracy {metrics['accuracy']:.3f} | macro precision {metrics['macro_precision']:.3f} | "
          f"macro recall {metrics['macro_recall']:.3f} | macro F1 {metrics['macro_f1']:.3f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
