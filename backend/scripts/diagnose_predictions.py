"""Log safe, per-image diagnostics for saved predictions or labeled validation samples.

Run from ``backend`` with DR_MODEL_PATH set to the model being investigated.
Output is JSON Lines to stdout; no images or patient fields are logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app import config, database as db
from app.services.model_service import service
from ml.bundle import predict_proba
from ml.preprocess import prepare_image


def _diagnose(image: Image.Image, *, record_id=None, filename: str, image_path: Path,
              known_label=None, saved=None, previous_hash=None) -> str:
    raw_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    meta = service.bundle["meta"]
    prepared = prepare_image(image, meta["image_size"], meta["clahe"])
    with service._lock:
        res_map, den_map = service.extractor.forward_maps(prepared[None])
    features = service.extractor.fuse(res_map, den_map)
    scaled = service.bundle["scaler"].transform(features)
    projected = service.bundle["pca"].transform(scaled)
    classifier = service.bundle["clf"]
    class_scores = predict_proba(service.bundle, features)[0]
    calibrated = [float(v) for v in class_scores]
    predicted = int(np.argmax(class_scores))

    # For CalibratedClassifierCV, inspect the final fitted SVC used at inference.
    fitted = getattr(classifier, "calibrated_classifiers_", None)
    estimator = fitted[0].estimator if fitted else classifier
    decision = estimator.decision_function(projected) if hasattr(estimator, "decision_function") else None
    saved_scores = saved.get("probabilities") if saved else None
    saved_class = saved.get("class_index") if saved else None
    saved_delta = (float(np.max(np.abs(np.asarray(saved_scores) - class_scores)))
                   if saved_scores is not None else None)
    row = {
        "record_id": record_id,
        "filename": filename,
        "image_path": str(image_path.resolve()),
        "image_sha256": raw_hash,
        "differs_from_previous_image": raw_hash != previous_hash,
        "feature_shape": list(features.shape),
        "feature_sha256": hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(),
        "known_dataset_label": known_label,
        "saved_class": saved_class,
        "fresh_class": predicted,
        "saved_top_score": saved.get("confidence") if saved else None,
        "fresh_top_score": float(class_scores[predicted]),
        "saved_fresh_max_score_delta": saved_delta,
        "raw_svm_decision_scores": np.asarray(decision).reshape(-1).round(6).tolist() if decision is not None else None,
        "calibrated_class_scores": [round(v, 6) for v in calibrated],
    }
    print(json.dumps(row), flush=True)
    return raw_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", type=int, help="Saved prediction database IDs to recheck.")
    parser.add_argument("--known-validation", action="store_true",
                        help="Also infer one labeled validation image from each class (0-4).")
    args = parser.parse_args(argv)
    if not args.ids and not args.known_validation:
        parser.error("provide --ids and/or --known-validation")

    service.load()
    if not service.is_trained or service.bundle is None:
        raise RuntimeError("A trained model is required; check DR_MODEL_PATH and server logs.")

    previous_hash = None
    for record_id in args.ids or []:
        row = db.get_prediction(record_id)
        if not row:
            print(json.dumps({"record_id": record_id, "error": "record not found"}), flush=True)
            continue
        image_path = config.UPLOAD_DIR / row["image_file"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        previous_hash = _diagnose(image, record_id=record_id, filename=row["filename"],
                                  image_path=image_path, saved=row, previous_hash=previous_hash)

    if args.known_validation:
        split_path = config.MODEL_DIR / "splits.json"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        chosen = {}
        for path in splits["val"]:
            label = int(splits["labels"][path])
            if label not in chosen and Path(path).is_file():
                chosen[label] = path
            if len(chosen) == 5:
                break
        for label in range(5):
            path = Path(chosen[label])
            with Image.open(path) as source:
                image = source.convert("RGB")
            previous_hash = _diagnose(image, filename=path.name, image_path=path,
                                      known_label=label, previous_hash=previous_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
