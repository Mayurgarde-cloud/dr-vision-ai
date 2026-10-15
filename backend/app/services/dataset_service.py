
"""Dataset and evaluation information for Dashboard and Model Performance."""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from ml import CLASS_NAMES, NUM_CLASSES
from ml.dataset import scan_dataset, summarize

from .. import config


def _read_json(path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _split_counts(total: int) -> dict[str, int]:
    saved = _read_json(config.METRICS_PATH) or {}
    split = saved.get("split", {})
    if split and sum(split.values()) > 0:
        return {key: int(split.get(key, 0))
                for key in ("train", "val", "test")}

    test = math.ceil(total * 0.15)
    val = math.ceil((total - test) * (0.15 / 0.85))
    return {
        "train": total - test - val,
        "val": val,
        "test": test,
    }


def _class_summary(saved: dict, total: int) -> list[dict]:
    counts = saved.get("classes", {})
    result = []

    for i in range(NUM_CLASSES):
        count = int(counts.get(str(i), counts.get(i, 0)))
        result.append({
            "index": i,
            "name": CLASS_NAMES[i],
            "count": count,
            "percent": round(100 * count / total, 2) if total else 0.0,
        })

    return result


def dataset_info() -> dict[str, Any]:
    # Prefer the real dataset metadata used for model training.
    run_dir = (config.MODEL_PATH.parent if config.MODEL_PATH != config.MODEL_DIR / "dr_model.joblib"
               else config.MODEL_DIR)
    saved = (_read_json(run_dir / "dataset_info.json") or
             _read_json(config.MODEL_DIR / "dataset_info.json") or {})
    trained = _read_json(config.METRICS_PATH) or {}

    df = scan_dataset(config.DATA_DIR)
    meta = _read_json(config.DATA_DIR / "dataset.json") or {}

    if saved.get("total") and saved.get("classes"):
        total = int(saved["total"])
        classes = _class_summary(saved, total)
        name = saved.get("name", "Diabetic Retinopathy Dataset")
        source = saved.get("source") or "Local diabetic retinopathy dataset"
        status = "Training dataset"
    elif not df.empty:
        summary = summarize(df)
        total = int(summary["total"])
        classes = summary["classes"]
        name = meta.get("name", "Custom dataset")
        source = meta.get("source", f"Local folder: {config.DATA_DIR.name}/")
        status = "Loaded"
    else:
        total = 0
        classes = [
            {"index": i, "name": CLASS_NAMES[i], "count": 0, "percent": 0.0}
            for i in range(NUM_CLASSES)
        ]
        name = "Dataset not found"
        source = "No dataset metadata available"
        status = "Not found"

    split = _split_counts(total) if total else {
        "train": 0, "val": 0, "test": 0
    }

    def pct(value: int) -> int:
        return round(100 * value / total) if total else 0

    model_config = trained.get("config") or {}

    return {
        "name": name,
        "source": source,
        "total": total,
        "status": status,
        "classes": classes,
        "split": [
            {"name": "Training", "percent": pct(split["train"]),
             "count": split["train"]},
            {"name": "Validation", "percent": pct(split["val"]),
             "count": split["val"]},
            {"name": "Test", "percent": pct(split["test"]),
             "count": split["test"]},
        ],
        "split_is_actual": bool(trained.get("split")),
        "preprocessing": [
            {"label": f"Resize to {config.IMAGE_SIZE} x {config.IMAGE_SIZE}",
             "active": True},
            {"label": "Normalize (ImageNet mean/std)", "active": True},
            {"label": "CLAHE (optional)",
             "active": bool(model_config.get("clahe", config.USE_CLAHE))},
            {"label": "Data Augmentation (train only)",
             "active": bool(model_config.get("augment", 0))},
        ],
        "hint": (
            f"Add images to {config.DATA_DIR} or configure the training dataset."
            if status == "Not found" else None
        ),
    }


def model_performance() -> dict[str, Any]:
    metrics = _read_json(config.METRICS_PATH)
    classes = [
        {"index": i, "name": CLASS_NAMES[i]}
        for i in range(NUM_CLASSES)
    ]

    if not metrics:
        return {"evaluated": False, "classes": classes}

    return {**metrics, "evaluated": True, "classes": classes}
