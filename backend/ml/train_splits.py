
"""Train a candidate using explicit train/validation CSVs; never read the test split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml import CLASS_NAMES, NUM_CLASSES
from ml.bundle import predict_proba, save_bundle
from ml.dataset import (assert_content_disjoint, assert_content_labels_consistent,
                        read_csv_labels, summarize)
from ml.features import RES_DIM, FeatureExtractor, torch_available
from ml.metrics import compute_metrics
from ml.preprocess import augment, prepare_image
from ml.train import get_features, fit_cam_head


def load_split(csv_path, image_dir, name):
    df = read_csv_labels(csv_path, image_dir)
    if df.empty:
        raise ValueError(
            f"{name}: no matching images found. Check CSV and image directory."
        )

    paths = [str(Path(p).resolve()) for p in df["path"].tolist()]
    labels = df["label"].to_numpy(dtype=int)

    if len(paths) != len(labels):
        raise ValueError(f"{name}: image and label counts do not match.")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name}: duplicate image paths found.")

    print(f"{name}: {len(paths)} images | "
          f"class counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return paths, labels


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train/evaluate using separate labelled CSV splits."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--val-image-dir", required=True)
    parser.add_argument("--out-dir", default=str(BACKEND_DIR / "models" / "candidate_train_splits"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ada-estimators", type=int, default=10)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--rf-depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="Diabetic Retinopathy Dataset")
    parser.add_argument("--dataset-source", default=None)
    args = parser.parse_args(argv)

    if not torch_available():
        print("PyTorch and torchvision are required.")
        return 2

    if args.image_size < 32 or args.batch_size < 1 or args.augment < 0:
        print("Invalid image size, batch size or augmentation count.")
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    try:
        train_paths, y_train = load_split(
            args.train_csv, args.train_image_dir, "TRAIN"
        )
        val_paths, y_val = load_split(
            args.val_csv, args.val_image_dir, "VALIDATION"
        )
    except (ValueError, OSError) as exc:
        print(f"Dataset error: {exc}")
        return 1

    # Guard against accidental overlap between splits.
    train_set, val_set = set(train_paths), set(val_paths)
    if train_set & val_set:
        print("ERROR: image paths overlap between dataset splits.")
        return 1
    try:
        assert_content_labels_consistent(train_paths, y_train)
        assert_content_labels_consistent(val_paths, y_val)
        assert_content_disjoint(train_paths, val_paths, "training", "validation")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not set(np.unique(y_train)).issuperset(range(NUM_CLASSES)):
        print("ERROR: training split must contain all five classes (0-4).")
        return 1

    clahe = not args.no_clahe
    pretrained = not args.no_pretrained
    extractor = FeatureExtractor(pretrained=pretrained)

    print("\nExtracting training features...")
    X_train = get_features(
        train_paths, extractor, args.image_size, clahe,
        out / "features_train.npz", args.batch_size
    )
    print("\nExtracting validation features...")
    X_val = get_features(
        val_paths, extractor, args.image_size, clahe,
        out / "features_val.npz", args.batch_size
    )

    X_fit, y_fit = X_train, y_train

    if args.augment:
        rng = np.random.default_rng(args.seed)
        print(f"Augmenting training images x{args.augment}...")
        augmented_images = np.stack([
            augment(prepare_image(p, args.image_size, clahe), rng)
            for p in train_paths
            for _ in range(args.augment)
        ])
        augmented_features = extractor.extract_images(
            augmented_images, batch_size=args.batch_size
        )
        X_fit = np.concatenate([X_train, augmented_features])
        y_fit = np.concatenate([
            y_train, np.repeat(y_train, args.augment)
        ])

    # Fit preprocessing only on training data.
    scaler = StandardScaler().fit(X_fit)
    n_comp = min(
        args.pca_components, len(X_fit) - 1, X_fit.shape[1]
    )
    if n_comp < 2:
        print("ERROR: not enough training samples for PCA.")
        return 1

    pca = PCA(n_components=n_comp, random_state=args.seed)
    Z_train = pca.fit_transform(scaler.transform(X_fit))

    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=args.rf_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    clf = AdaBoostClassifier(
        estimator=rf,
        n_estimators=args.ada_estimators,
        learning_rate=0.8,
        random_state=args.seed,
    )
    print("\nTraining classifier...")
    clf.fit(Z_train, y_fit)

    cam_W, cam_b = fit_cam_head(X_fit[:, :RES_DIM], y_fit, args.seed)

    bundle = {
        "scaler": scaler,
        "pca": pca,
        "clf": clf,
        "cam_W": cam_W,
        "cam_b": cam_b,
        "meta": {
            "image_size": args.image_size,
            "clahe": clahe,
            "pretrained": pretrained,
            "classes": CLASS_NAMES,
            "pca_components": int(n_comp),
            "augment": args.augment,
            "n_train": int(len(train_paths)),
            "n_val": int(len(val_paths)),
            "trained_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
    }

    val_prob = predict_proba(
        bundle, X_val
    )
    val_pred = val_prob.argmax(axis=1)

    val_metrics = compute_metrics(y_val, val_pred)

    split_counts = {
        "train": len(train_paths),
        "val": len(val_paths),
    }
    config = {
        "pretrained": pretrained,
        "clahe": clahe,
        "image_size": args.image_size,
        "pca_components": int(n_comp),
        "augment": args.augment,
        "ada_estimators": args.ada_estimators,
        "rf_trees": args.rf_trees,
        "rf_depth": args.rf_depth,
        "seed": args.seed,
    }

    # Keep the format expected by the existing evaluator and app.
    metrics = {"validation_metrics": val_metrics, "selected_for_test": False}
    metrics["split"] = split_counts
    metrics["config"] = config

    # Save the bundle last, after successful training and evaluation.
    save_bundle(bundle, out / "dr_model.joblib")

    all_paths = train_paths + val_paths
    all_labels = np.concatenate([y_train, y_val])
    split_info = {
        "train": train_paths,
        "val": val_paths,
        "labels": {
            p: int(label) for p, label in zip(all_paths, all_labels)
        },
    }
    (out / "splits.json").write_text(
        json.dumps(split_info, indent=2), encoding="utf-8"
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    info = {
        "name": args.dataset_name,
        "source": args.dataset_source,
        "total": len(all_paths),
        "classes": {
            str(i): int(np.sum(all_labels == i))
            for i in range(NUM_CLASSES)
        },
        "split": split_counts,
        "created_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    print("\n--- Validation results ---")
    print(
        f"Accuracy: {val_metrics['accuracy']:.3f} | "
        f"Macro-F1: {val_metrics['macro_f1']:.3f}"
    )
    print("\nNo test data was evaluated. Use ml.evaluate only after selecting this candidate.")

    print(f"\nSplit sizes: {split_counts}")
    print(f"Elapsed time: {time.time() - started:.1f} seconds")
    print(f"Saved model and reports to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
