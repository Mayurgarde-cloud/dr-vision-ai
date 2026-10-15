"""Prediction endpoints: upload & classify, history, delete, PDF report."""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from .. import config, database as db
from ..schemas import ClassProbability, HistoryItem, MessageOut, ModelInfo, PredictionOut
from ..services import report, visuals
from ..services.model_service import service

router = APIRouter(prefix="/api", tags=["predictions"])


# ------------------------------------------------------------------ helpers
def _model_info(mode: str) -> ModelInfo:
    note = None
    if mode == "demo":
        note = "Demo mode – this result comes from a simple image-statistics heuristic, not a trained model."
    elif service.reason:
        note = service.reason
    meta = service.bundle.get("meta", {}) if service.bundle else {}
    score_description = ("SVM scores calibrated by internal five-fold training cross-validation; not a probability of actual disease."
                         if meta.get("probability_calibration") else
                         "Classifier output score; not a calibrated probability of actual disease.")
    return ModelInfo(backbones=config.MODEL_BACKBONES, classifier=meta.get("classifier", config.MODEL_CLASSIFIER),
                     image_size=f"{service.image_size} × {service.image_size}", mode=mode, mode_note=note,
                     version=meta.get("model_version", "production" if mode == "trained" else "demo"),
                     score_description=score_description)


def _to_out(row: dict) -> PredictionOut:
    probs = [ClassProbability(index=i, name=config.CLASS_NAMES[i], probability=round(p * 100, 2))
             for i, p in enumerate(row["probabilities"])]
    return PredictionOut(
        id=row["id"], filename=row["filename"], class_index=row["class_index"], class_name=row["class_name"],
        confidence=round(row["confidence"] * 100, 2), probabilities=probs,
        processing_time=round(row["processing_time"], 2), created_at=row["created_at"],
        image_url=f"/files/uploads/{row['image_file']}",
        heatmap_url=f"/files/heatmaps/{row['heatmap_file']}" if row.get("heatmap_file") else None,
        overlay_url=f"/files/heatmaps/{row['overlay_file']}" if row.get("overlay_file") else None,
        explanation=config.CLASS_EXPLANATIONS[row["class_index"]],
        warnings=row.get("warnings", []), model=_model_info(row["mode"]),
    )


def _to_history(row: dict) -> HistoryItem:
    return HistoryItem(id=row["id"], filename=row["filename"], class_index=row["class_index"],
                       class_name=row["class_name"], confidence=round(row["confidence"] * 100, 2),
                       created_at=row["created_at"], image_url=f"/files/uploads/{row['image_file']}",
                       mode=row["mode"], model_version=row.get("model_version", "unknown"))


def _remove_files(row: Optional[dict]) -> None:
    if not row:
        return
    for key, folder in (("image_file", config.UPLOAD_DIR), ("heatmap_file", config.HEATMAP_DIR),
                        ("overlay_file", config.HEATMAP_DIR)):
        if row.get(key):
            try:
                (Path(folder) / row[key]).unlink(missing_ok=True)
            except OSError:
                pass


def _read_upload(file: UploadFile) -> tuple[Image.Image, bytes]:
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS or (file.content_type or "").lower() not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Unsupported file type. Please upload a JPG, JPEG or PNG image.")
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large. The maximum size is 5 MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                                   # detects truncated / corrupt files
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(400, "The file could not be read as an image. It may be corrupted.")
    if min(img.size) < config.MIN_IMAGE_SIDE:
        raise HTTPException(400, f"Image is too small. Minimum size is {config.MIN_IMAGE_SIDE} × {config.MIN_IMAGE_SIDE} px.")
    return img, data


# ------------------------------------------------------------------ endpoints
@router.post("/predict", response_model=PredictionOut)
def predict(file: UploadFile = File(...)) -> PredictionOut:
    img, _ = _read_upload(file)
    started = time.perf_counter()
    try:
        pred = service.predict(img)
    except Exception as exc:  # pragma: no cover - defensive
        import logging
        logging.getLogger("dr_vision.predictions").exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Please try another image or contact the administrator.")
    elapsed = time.perf_counter() - started

    heat = visuals.colorize(pred.heatmap, pred.display_rgb.shape[0])
    over = visuals.overlay(pred.display_rgb, heat)

    uid = uuid.uuid4().hex
    img_name, heat_name, over_name = f"{uid}.jpg", f"{uid}_heatmap.png", f"{uid}_overlay.png"
    config.ensure_dirs()
    stored = img.copy()
    stored.thumbnail((1024, 1024))
    stored.save(config.UPLOAD_DIR / img_name, "JPEG", quality=90)
    Image.fromarray(heat).save(config.HEATMAP_DIR / heat_name)
    Image.fromarray(over).save(config.HEATMAP_DIR / over_name)

    warnings = []
    if not visuals.looks_like_fundus(np.asarray(img)):
        warnings.append("This image does not look like a typical retinal fundus photograph, so the result may be "
                        "unreliable.")

    pid = db.insert_prediction(
        filename=Path(file.filename or "upload").name[:200], image_file=img_name, heatmap_file=heat_name,
        overlay_file=over_name, class_index=pred.class_index, class_name=config.CLASS_NAMES[pred.class_index],
        confidence=pred.confidence, probabilities=[float(p) for p in pred.probabilities],
        processing_time=elapsed, mode=pred.mode,
        model_version=_model_info(pred.mode).version, warnings=warnings)
    return _to_out(db.get_prediction(pid))


@router.get("/predictions", response_model=list[HistoryItem])
def list_predictions(search: str = Query("", max_length=100), limit: int = Query(200, ge=1, le=1000)):
    return [_to_history(r) for r in db.list_predictions(search, limit)]


@router.get("/predictions/{pred_id}", response_model=PredictionOut)
def get_prediction(pred_id: int) -> PredictionOut:
    row = db.get_prediction(pred_id)
    if not row:
        raise HTTPException(404, "Prediction not found.")
    return _to_out(row)


@router.delete("/predictions/{pred_id}", response_model=MessageOut)
def delete_prediction(pred_id: int) -> MessageOut:
    row = db.delete_prediction(pred_id)
    if not row:
        raise HTTPException(404, "Prediction not found.")
    _remove_files(row)
    return MessageOut(message="Prediction deleted.", deleted=1)


@router.delete("/predictions", response_model=MessageOut)
def clear_history() -> MessageOut:
    rows = db.delete_all_predictions()
    for r in rows:
        _remove_files(r)
    return MessageOut(message="History cleared.", deleted=len(rows))


@router.get("/predictions/{pred_id}/report")
def download_report(pred_id: int) -> Response:
    row = db.get_prediction(pred_id)
    if not row:
        raise HTTPException(404, "Prediction not found.")
    row["score_description"] = _model_info(row["mode"]).score_description
    pdf = report.build_pdf(row)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="dr-vision-report-{pred_id}.pdf"'})
