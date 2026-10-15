"""Read-only endpoints for the dashboard, dataset and model-performance pages."""
from __future__ import annotations

from fastapi import APIRouter

from .. import config, database as db
from ..services import dataset_service
from ..services.model_service import service
from ml.features import torch_available

router = APIRouter(prefix="/api", tags=["insights"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "model_mode": service.mode, "model_loaded": service.is_trained,
            "torch_available": torch_available(), "note": service.reason,
            "reason_code": service.reason_code}


@router.get("/dashboard")
def dashboard() -> dict:
    ds = dataset_service.dataset_info()
    recent = db.list_predictions(limit=5)
    last = recent[0] if recent else None
    return {
        "dataset": {"status": ds["status"], "total": ds["total"]},
        "model": {
            "status": "Ready" if service.is_trained else "Demo Mode",
            "mode": service.mode,
            "detail": config.MODEL_CLASSIFIER if service.is_trained else "No trained model loaded",
            "note": service.reason,
        },
        "last_prediction": last["created_at"] if last else None,
        "total_predictions": db.count_predictions(),
        "class_distribution": ds["classes"],
        "recent": [{"id": r["id"], "class_index": r["class_index"], "class_name": r["class_name"],
                    "confidence": round(r["confidence"] * 100, 2), "created_at": r["created_at"]} for r in recent],
    }


@router.get("/dataset")
def dataset() -> dict:
    return dataset_service.dataset_info()


@router.get("/model/performance")
def model_performance() -> dict:
    out = dataset_service.model_performance()
    out["model_mode"] = service.mode
    return out


@router.get("/classes")
def classes() -> list[dict]:
    return [{"index": i, "name": n, "description": config.CLASS_SHORT[i]} for i, n in enumerate(config.CLASS_NAMES)]
