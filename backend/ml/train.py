"""Train the DR severity classifier.

Pipeline
--------
    fundus image → crop / CLAHE / resize 224 → ResNet-50 (2048-d) + DenseNet-121 (1024-d)
                 → feature fusion (3072-d) → StandardScaler → PCA → AdaBoost(Random Forest) → DR grade 0-4

Run from the ``backend`` folder::

    python -m ml.train                                   # uses data/dataset/<class folders>
    python -m ml.train --csv train.csv --image-dir imgs  # APTOS-style CSV
    python -m ml.train --pca-components 128 --ada-estimators 20

Outputs (in ``models/candidate/`` by default): a candidate bundle and validation metrics.
Test images are not opened until ``python -m ml.evaluate --model-dir <candidate>``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from ml.bundle import predict_proba, save_bundle  # noqa: E402
from ml.dataset import (assert_content_labels_consistent, image_content_hash, read_csv_labels,
                        scan_dataset, summarize, unique_content_indices)  # noqa: E402
from ml.preprocess import augment, prepare_image  # noqa: E402
from ml.features import RES_DIM, FeatureExtractor, torch_available  # noqa: E402
from ml.metrics import compute_metrics  # noqa: E402


def stratified_split(labels: np.ndarray, val_size: float, test_size: float, seed: int, groups=None):
    idx = np.arange(len(labels))
    if set(np.unique(labels)) != set(range(NUM_CLASSES)):
        raise ValueError("Training data must contain all five DR classes.")
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    if groups.shape != labels.shape:
        raise ValueError("Image content groups must align with the label array.")
    per_class_groups = [len(set(groups[labels == cls])) for cls in range(NUM_CLASSES)]
    if min(per_class_groups) < 2:
        raise ValueError("Each class needs at least two distinct image-content groups to split.")
    test_folds = min(max(2, round(1 / test_size)), min(per_class_groups))
    first = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    trainval, test = next(first.split(idx, labels, groups))
    val_fraction = val_size / (1 - test_size)
    val_fold_limit = min(len(set(groups[trainval][labels[trainval] == cls])) for cls in range(NUM_CLASSES))
    val_folds = min(max(2, round(1 / val_fraction)), val_fold_limit)
    second = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    train_rel, val_rel = next(second.split(trainval, labels[trainval], groups[trainval]))
    train, val = trainval[train_rel], trainval[val_rel]
    if (set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test])
            or set(groups[val]) & set(groups[test])):
        raise AssertionError("Exact-content image groups crossed a dataset split.")
    return train, val, test


def get_features(paths: list[str], extractor: FeatureExtractor, size: int, clahe: bool, cache: Path,
                 batch_size: int) -> np.ndarray:
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                cached_paths = [str(Path(p).resolve()) for p in z["paths"].tolist()]
                cached_feats = z["feats"]
                compatible = (bool(z["pretrained"]) == extractor.pretrained
                              and int(z["size"]) == size and bool(z["clahe"]) == clahe
                              and cached_feats.shape == (len(cached_paths), 3072)
                              and np.isfinite(cached_feats).all())
                cache_index = {p: i for i, p in enumerate(cached_paths)}
                requested = [str(Path(p).resolve()) for p in paths]
                if compatible and all(p in cache_index for p in requested):
                    if "source_hashes" in z.files:
                        expected_hashes = dict(zip(cached_paths, z["source_hashes"].tolist()))
                        for path in requested:
                            if image_content_hash(path) != expected_hashes[path]:
                                raise ValueError(f"Cached source image changed since feature extraction: {path}")
                    else:
                        print(f"Reusing legacy cache {cache} without per-image source hashes; verified by path and pipeline metadata.")
                    print(f"Using cached features from {cache}")
                    return cached_feats[[cache_index[p] for p in requested]]
        except Exception:
            pass
    print(f"Extracting ResNet-50 + DenseNet-121 features for {len(paths)} images on {extractor.device} ...")
    feats = extractor.extract_paths(paths, size=size, clahe=clahe, batch_size=batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.asarray(paths), feats=feats, pretrained=extractor.pretrained, size=size,
             clahe=clahe, source_hashes=np.asarray([image_content_hash(p) for p in paths]))
    return feats


def fit_cam_head(res_feats: np.ndarray, y: np.ndarray, seed: int):
    """Linear head on pooled ResNet-50 features, used to draw class-activation maps."""
    lr = LogisticRegression(max_iter=3000, C=0.5, random_state=seed)
    lr.fit(res_feats, y)
    W = np.zeros((NUM_CLASSES, res_feats.shape[1]), dtype=np.float32)
    b = np.zeros(NUM_CLASSES, dtype=np.float32)
    coef, icpt, classes = lr.coef_, lr.intercept_, lr.classes_
    if len(classes) == 2:  # sklearn returns a single row for binary problems
        coef, icpt = np.vstack([-coef[0], coef[0]]), np.array([-icpt[0], icpt[0]])
    for row, c in enumerate(classes):
        W[int(c)], b[int(c)] = coef[row], icpt[row]
    return W, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(BACKEND_DIR / "data" / "dataset"))
    ap.add_argument("--csv", help="CSV with image ids and labels (APTOS style)")
    ap.add_argument("--image-dir", help="Folder with the images referenced by --csv")
    ap.add_argument("--out-dir", default=str(BACKEND_DIR / "models" / "candidate"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--no-clahe", action="store_true", help="disable CLAHE contrast enhancement")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random backbone weights (only for offline smoke tests – accuracy will be poor)")
    ap.add_argument("--augment", type=int, default=0,
                    help="extra randomly augmented copies of every TRAIN image (flip/rotate/brightness)")
    ap.add_argument("--pca-components", type=int, default=128)
    ap.add_argument("--ada-estimators", type=int, default=15)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=8)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--dataset-source", default=None)
    args = ap.parse_args(argv)

    if not torch_available():
        print("PyTorch / torchvision are required for training. Install them first (see README).")
        return 2

    out = Path(args.out_dir)
    if out.resolve() == (BACKEND_DIR / "models").resolve() and (out / "dr_model.joblib").exists():
        print("Refusing to overwrite the production bundle. Choose a separate candidate --out-dir.")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    df = read_csv_labels(args.csv, args.image_dir) if args.csv else scan_dataset(args.data_dir)
    if df.empty:
        print("No images found. Put images in class folders under data/dataset/ or use --csv/--image-dir.")
        return 1
    print(f"Found {len(df)} images: {summarize(df)['classes']}")
    paths, labels = df["path"].tolist(), df["label"].to_numpy(dtype=int)
    try:
        unique_indices, duplicates_removed = unique_content_indices(paths, labels)
    except ValueError as exc:
        print(f"Dataset integrity error: {exc}")
        return 1
    if duplicates_removed:
        print(f"Removed {duplicates_removed} repeated exact-content image row(s) before splitting.")
        paths = [paths[i] for i in unique_indices]
        labels = labels[unique_indices]
    clahe = not args.no_clahe

    content_hashes = np.asarray(assert_content_labels_consistent(paths, labels))
    tr, va, te = stratified_split(labels, args.val_size, args.test_size, args.seed, content_hashes)
    print(f"Split → train {len(tr)}  val {len(va)}  test {len(te)}")

    # Keep the test split to paths and labels; do not load its images or extract features here.
    t0 = time.time()
    extractor = FeatureExtractor(pretrained=not args.no_pretrained)
    train_paths = [paths[i] for i in tr]
    val_paths = [paths[i] for i in va]
    X_train = get_features(train_paths, extractor, args.image_size, clahe,
                           out / "features_train.npz", args.batch_size)
    X_val = get_features(val_paths, extractor, args.image_size, clahe,
                         out / "features_val.npz", args.batch_size)

    Xtr, ytr = X_train, labels[tr]
    if args.augment > 0:
        rng = np.random.default_rng(args.seed)
        print(f"Augmenting {len(tr)} training images x{args.augment} ...")
        aug = np.stack([augment(prepare_image(paths[i], args.image_size, clahe), rng)
                        for i in tr for _ in range(args.augment)])
        Xtr = np.concatenate([Xtr, extractor.extract_images(aug, args.batch_size)])
        ytr = np.concatenate([ytr, np.repeat(labels[tr], args.augment)])

    scaler = StandardScaler().fit(Xtr)
    n_comp = max(2, min(args.pca_components, len(Xtr) - 1, Xtr.shape[1]))
    pca = PCA(n_components=n_comp, random_state=args.seed).fit(scaler.transform(Xtr))
    ztr = pca.transform(scaler.transform(Xtr))
    print(f"PCA: 3072 → {n_comp} dims (explained variance {pca.explained_variance_ratio_.sum():.1%})")

    rf = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=args.rf_depth, class_weight="balanced",
                                n_jobs=-1, random_state=args.seed)
    clf = AdaBoostClassifier(estimator=rf, n_estimators=args.ada_estimators, learning_rate=0.8,
                             random_state=args.seed)
    clf.fit(ztr, ytr)

    cam_W, cam_b = fit_cam_head(Xtr[:, :RES_DIM], ytr, args.seed)

    bundle = {
        "scaler": scaler, "pca": pca, "clf": clf, "cam_W": cam_W, "cam_b": cam_b,
        "meta": {
            "image_size": args.image_size, "clahe": clahe, "pretrained": not args.no_pretrained,
            "classes": CLASS_NAMES, "pca_components": int(n_comp), "augment": args.augment,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    val_pred = predict_proba(bundle, X_val).argmax(1)
    val_metrics = compute_metrics(labels[va], val_pred)
    metrics = {"validation_metrics": val_metrics, "selected_for_test": False}
    metrics["val_accuracy"] = val_metrics["accuracy"]
    metrics["split"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}
    metrics["config"] = {"pretrained": not args.no_pretrained, "clahe": clahe, "pca_components": int(n_comp), "augment": args.augment,
                         "ada_estimators": args.ada_estimators, "rf_trees": args.rf_trees, "seed": args.seed}
    bundle["meta"].update({
        "classifier": "AdaBoost + Random Forest", "model_version": "candidate",
        "validation_macro_f1": val_metrics["macro_f1"],
    })
    save_bundle(bundle, out / "dr_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "splits.json").write_text(json.dumps({
        "train": [paths[i] for i in tr], "val": [paths[i] for i in va], "test": [paths[i] for i in te],
        "labels": {paths[i]: int(labels[i]) for i in np.concatenate([tr, va, te])},
    }))
    info = summarize(df)
    info.update({"split": metrics["split"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if args.dataset_name:
        info["name"] = args.dataset_name
    if args.dataset_source:
        info["source"] = args.dataset_source
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nValidation accuracy {val_metrics['accuracy']:.3f} | "
          f"macro-F1 {val_metrics['macro_f1']:.3f} | {time.time() - t0:.0f}s")
    print("Test images were not evaluated. Select the candidate before running ml.evaluate once.")
    print(f"Saved candidate bundle + validation metrics to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
