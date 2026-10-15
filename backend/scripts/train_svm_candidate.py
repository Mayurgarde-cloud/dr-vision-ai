"""Train and select an RBF-SVM using cached train/validation features only.

The test cache is deliberately never opened by this script. Candidate bundle and
validation artifacts are written into a new experiment directory, not models/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml import CLASS_NAMES, NUM_CLASSES
from ml.bundle import save_bundle
from ml.metrics import compute_metrics
from ml.train import fit_cam_head
from ml.dataset import assert_content_disjoint, assert_content_labels_consistent


def load_features(root: Path, split: str, label_map: dict[str, int]):
    with np.load(root / f"features_{split}.npz", allow_pickle=False) as z:
        paths = [str(Path(p).resolve()) for p in z["paths"].tolist()]
        feats = z["feats"].astype(np.float32)
        metadata = (bool(z["pretrained"]), int(z["size"]), bool(z["clahe"]))
    return paths, feats, np.asarray([label_map[p] for p in paths], dtype=np.int64), metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/experiment_aug1"))
    parser.add_argument("--splits", type=Path, default=Path("models/splits.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/experiment_svm_v1"))
    args = parser.parse_args(argv)

    labels = json.loads(args.splits.read_text(encoding="utf-8"))["labels"]
    train_paths, x_train, y_train, train_meta = load_features(args.cache_dir, "train", labels)
    val_paths, x_val, y_val, val_meta = load_features(args.cache_dir, "val", labels)
    if train_meta != val_meta or x_train.shape[1] != 3072:
        raise ValueError("Train/validation feature metadata or dimensions do not match.")
    if set(train_paths) & set(val_paths):
        raise ValueError("Train and validation features overlap.")
    assert_content_labels_consistent(train_paths, y_train)
    assert_content_labels_consistent(val_paths, y_val)
    assert_content_disjoint(train_paths, val_paths, "training features", "validation features")
    if set(np.unique(y_train)) != set(range(NUM_CLASSES)):
        raise ValueError("Training features must contain all five labels.")

    pretrained, image_size, clahe = train_meta
    results = []
    best = None
    # A small, explicit validation-only search around the already promising RBF-SVM.
    for components in (128, 256):
        scaler = StandardScaler().fit(x_train)
        pca = PCA(n_components=components, random_state=42).fit(scaler.transform(x_train))
        z_train = pca.transform(scaler.transform(x_train))
        z_val = pca.transform(scaler.transform(x_val))
        for c_value in (1.0, 2.0, 4.0):
            for weight in ("balanced", None):
                clf = CalibratedClassifierCV(
                    SVC(C=c_value, kernel="rbf", gamma="scale", class_weight=weight, random_state=42),
                    method="sigmoid", cv=5, ensemble=False,
                )
                started = time.perf_counter()
                clf.fit(z_train, y_train)
                training_seconds = time.perf_counter() - started
                infer_started = time.perf_counter()
                probabilities = clf.predict_proba(z_val)
                inference_seconds = time.perf_counter() - infer_started
                full = np.zeros((len(y_val), NUM_CLASSES), dtype=np.float64)
                full[:, clf.classes_.astype(int)] = probabilities
                pred = full.argmax(axis=1)
                metrics = compute_metrics(y_val, pred)
                one_hot = np.eye(NUM_CLASSES)[y_val]
                row = {
                    "classifier": "RBF-SVM", "pca_components": components,
                    "C": c_value, "gamma": "scale", "class_weight": weight,
                    "validation": metrics,
                    "validation_log_loss": float(log_loss(y_val, full, labels=list(range(NUM_CLASSES)))),
                    "validation_brier_multiclass": float(np.mean(np.sum((full - one_hot) ** 2, axis=1))),
                    "training_seconds": training_seconds,
                    "validation_inference_seconds": inference_seconds,
                    "validation_inference_ms_per_image": inference_seconds * 1000 / len(y_val),
                }
                results.append(row)
                print(f"PCA={components}, C={c_value}, weights={weight}: "
                      f"macro-F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
                # Selection priority: macro-F1, macro-recall, then accuracy.
                key = (metrics["macro_f1"], metrics["macro_recall"], metrics["accuracy"])
                if best is None or key > best[0]:
                    best = (key, row, scaler, pca, clf)

    _, selected, scaler, pca, clf = best
    # CAM explanation is a separate linear surrogate trained on training images only.
    from ml.features import RES_DIM
    cam_w, cam_b = fit_cam_head(x_train[:, :RES_DIM], y_train, seed=42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler": scaler, "pca": pca, "clf": clf, "cam_W": cam_w, "cam_b": cam_b,
        "meta": {
            "bundle_schema": 2, "model_version": "svm-v1",
            "classifier": "RBF-SVM", "image_size": image_size, "clahe": clahe,
            "probability_calibration": "sigmoid, internal stratified five-fold CV on training split",
            "pretrained": pretrained, "classes": CLASS_NAMES,
            "pca_components": int(pca.n_components_), "selected_on": "validation macro F1",
            "validation_macro_f1": selected["validation"]["macro_f1"],
            "n_train": len(y_train), "n_val": len(y_val),
        },
    }
    save_bundle(bundle, args.out_dir / "dr_model.joblib")
    (args.out_dir / "validation_results.json").write_text(
        json.dumps({"selection_metric": "macro_f1", "selected": selected,
                    "candidates": results}, indent=2), encoding="utf-8")
    print(f"Selected PCA={selected['pca_components']} C={selected['C']} "
          f"class_weight={selected['class_weight']}; bundle saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
