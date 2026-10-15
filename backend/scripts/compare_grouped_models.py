"""Compare classifiers on a content-deduplicated, train-only validation split.

This audit deliberately does not open the official validation or test labels for
scoring and never writes a model bundle. Their image bytes are hashed only to
exclude train rows that duplicate either official holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from PIL import Image

from ml.dataset import read_csv_labels
from ml.metrics import compute_metrics


def _hash_images(directory: Path) -> dict[str, str]:
    output = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        with Image.open(p) as image:
            image.verify()
        output[p.stem] = hashlib.sha256(p.read_bytes()).hexdigest()
    return output


def _validation_scores(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    one_hot = label_binarize(y_true, classes=np.arange(5))
    return {
        "macro_ovr_roc_auc": float(roc_auc_score(y_true, probabilities, labels=np.arange(5),
                                                    multi_class="ovr", average="macro")),
        "macro_average_precision": float(average_precision_score(one_hot, probabilities, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(5))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - np.eye(5)[y_true]) ** 2, axis=1))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\Diabetic Dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/experiment_aug1"))
    parser.add_argument("--output", type=Path, default=Path("models/grouped_validation_audit.json"))
    args = parser.parse_args(argv)

    root = args.dataset_root
    split_specs = {
        "train": (root / "train_1.csv", root / "train_images" / "train_images"),
        "validation": (root / "valid.csv", root / "val_images" / "val_images"),
        "test": (root / "test.csv", root / "test_images" / "test_images"),
    }
    split_data = {}
    for name, (csv_path, image_dir) in split_specs.items():
        label_rows = read_csv_labels(csv_path, image_dir)
        labels = {Path(row.path).stem: int(row.label) for row in label_rows.itertuples()}
        hashes = _hash_images(image_dir)
        missing = sorted(set(labels) - set(hashes))
        if missing:
            raise ValueError(f"{name}: {len(missing)} labeled images are missing; examples {missing[:5]}.")
        extras = sorted(set(hashes) - set(labels))
        if extras:
            raise ValueError(f"{name}: {len(extras)} unlabeled image file(s); examples {extras[:5]}.")
        split_data[name] = {ident: (labels.get(ident), hashes[ident]) for ident in hashes if ident in labels}

    content_rows: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for split_name, rows in split_data.items():
        for ident, (label, digest) in rows.items():
            content_rows[digest].append((split_name, ident, label))
    duplicate_groups = [rows for rows in content_rows.values() if len(rows) > 1]
    cross_split_groups = [rows for rows in duplicate_groups if len({row[0] for row in rows}) > 1]
    all_conflicting_content_groups = [rows for rows in duplicate_groups if len({row[2] for row in rows}) > 1]

    # Exclude train images with byte-identical copies in either supplied holdout.
    holdout_hashes = {digest for name in ("validation", "test")
                      for _label, digest in split_data[name].values()}

    cache = args.cache_dir / "features_train.npz"
    with np.load(cache, allow_pickle=False) as archive:
        cache_paths = [str(Path(p).resolve()) for p in archive["paths"].tolist()]
        features = archive["feats"].astype(np.float32)
        if features.shape != (len(cache_paths), 3072) or not np.isfinite(features).all():
            raise ValueError(f"Invalid cached feature matrix: {features.shape}.")
        if not bool(archive["pretrained"]) or int(archive["size"]) != 224 or not bool(archive["clahe"]):
            raise ValueError("Feature cache does not match the expected pretrained, CLAHE, 224 px pipeline.")

    # Map source IDs to actual current content and resolve label conflicts before splitting.
    train_labels = {ident: label for ident, (label, _digest) in split_data["train"].items()}
    cache_index = {Path(p).stem: i for i, p in enumerate(cache_paths)}
    by_digest: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for ident, digest in _hash_images(split_specs["train"][1]).items():
        if ident in train_labels and ident in cache_index:
            by_digest[digest].append((ident, train_labels[ident]))
    external_overlap_groups = sum(any(digest in holdout_hashes for _id, _lab in group)
                                  for digest, group in by_digest.items())
    train_conflicting_groups = {digest for digest, group in by_digest.items()
                                if len({label for _ident, label in group}) > 1}

    selected = []
    for digest, group in by_digest.items():
        if digest in holdout_hashes or digest in train_conflicting_groups:
            continue
        ident, label = sorted(group)[0]  # one representative per exact-content group
        selected.append((ident, digest, label, cache_index[ident]))
    if len({x[2] for x in selected}) != 5:
        raise ValueError("Content-clean training pool does not contain all five classes.")

    ids = np.asarray([x[0] for x in selected])
    groups = np.asarray([x[1] for x in selected])
    y = np.asarray([x[2] for x in selected], dtype=np.int64)
    X = features[[x[3] for x in selected]]
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                                    random_state=42).split(X, y))
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise AssertionError("Exact-content groups overlap the train-only validation split.")

    results = []
    candidates = []
    for components in (128, 256):
        scaler = StandardScaler().fit(X[train_idx])
        pca = PCA(n_components=components, random_state=42).fit(scaler.transform(X[train_idx]))
        z_train = pca.transform(scaler.transform(X[train_idx]))
        z_val = pca.transform(scaler.transform(X[val_idx]))
        candidates.extend([
            (f"RBF-SVM PCA{components} balanced C=1", CalibratedClassifierCV(
                SVC(C=1, kernel="rbf", class_weight="balanced", random_state=42), method="sigmoid", cv=5)),
            (f"RBF-SVM PCA{components} balanced C=2", CalibratedClassifierCV(
                SVC(C=2, kernel="rbf", class_weight="balanced", random_state=42), method="sigmoid", cv=5)),
        ])
        if components == 128:
            candidates.extend([
                ("RandomForest PCA128 balanced", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                          max_features="sqrt", min_samples_leaf=2,
                                                                          n_jobs=-1, random_state=42)),
                ("ExtraTrees PCA128 balanced", ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                                                      max_features="sqrt", min_samples_leaf=2,
                                                                      n_jobs=-1, random_state=42)),
                ("AdaBoost-RandomForest PCA128", AdaBoostClassifier(
                    estimator=RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                                                     n_jobs=-1, random_state=42), n_estimators=10,
                    learning_rate=0.8, random_state=42)),
            ])
        for name, classifier in candidates[-(2 if components == 256 else 5):]:
            classifier.fit(z_train, y[train_idx])
            probs_native = classifier.predict_proba(z_val)
            probs = np.zeros((len(val_idx), 5), dtype=np.float64)
            probs[:, classifier.classes_.astype(int)] = probs_native
            y_pred = probs.argmax(axis=1)
            results.append({
                "model": name,
                "validation": compute_metrics(y[val_idx], y_pred),
                "score_metrics": _validation_scores(y[val_idx], probs),
                "score_note": "Sigmoid calibrated SVM scores" if "SVM" in name else "Classifier predict_proba scores; calibration not assessed",
            })

    results.sort(key=lambda row: row["validation"]["macro_f1"], reverse=True)
    report = {
        "evaluation": "content-deduplicated internal validation split from supplied training CSV only",
        "selection_metric": "validation macro F1",
        "seed": 42,
        "official_validation_or_test_labels_used_for_scoring": False,
        "source_integrity": {
            "train_files": len(split_data["train"]),
            "official_validation_files": len(split_data["validation"]),
            "official_test_files": len(split_data["test"]),
            "train_groups_overlapping_official_holdouts_excluded": external_overlap_groups,
            "conflicting_label_groups_excluded": len(train_conflicting_groups),
            "remaining_unique_train_images": len(selected),
            "train_rows": len(train_idx),
            "internal_validation_rows": len(val_idx),
            "class_distribution_after_filter": {str(i): int(np.sum(y == i)) for i in range(5)},
            "exact_duplicate_content_groups_all_sources": len(duplicate_groups),
            "exact_duplicate_content_groups_cross_source_splits": len(cross_split_groups),
            "exact_duplicate_content_groups_with_label_conflicts": len(all_conflicting_content_groups),
        },
        "candidates_ranked": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["source_integrity"],
                      "best_candidate": results[0]["model"],
                      "best_macro_f1": results[0]["validation"]["macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
