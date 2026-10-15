"""Central configuration for the DR Vision AI backend."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Storage -------------------------------------------------------------------
STORAGE_DIR = Path(os.getenv("DR_STORAGE_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
HEATMAP_DIR = STORAGE_DIR / "heatmaps"
DB_PATH = Path(os.getenv("DR_DB_PATH", STORAGE_DIR / "dr_vision.sqlite3"))

# --- Dataset & model artefacts --------------------------------------------------
DATA_DIR = Path(os.getenv("DR_DATA_DIR", BASE_DIR / "data" / "dataset"))
MODEL_DIR = Path(os.getenv("DR_MODEL_DIR", BASE_DIR / "models"))
_MODEL_PATH_OVERRIDE = os.getenv("DR_MODEL_PATH")
MODEL_PATH = Path(_MODEL_PATH_OVERRIDE or MODEL_DIR / "dr_model.joblib")
MODEL_RUN_DIR = MODEL_PATH.parent if _MODEL_PATH_OVERRIDE else MODEL_DIR
_DEFAULT_METRICS = (MODEL_RUN_DIR / "test_metrics.json" if _MODEL_PATH_OVERRIDE else MODEL_DIR / "metrics.json")
METRICS_PATH = Path(os.getenv("DR_METRICS_PATH", _DEFAULT_METRICS))
SPLITS_PATH = MODEL_RUN_DIR / "splits.json"

# --- Upload validation ------------------------------------------------------------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGE_SIDE = 64

# --- Classes ------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
CLASS_SHORT = {
    0: "Healthy retina",
    1: "Early changes",
    2: "More visible signs",
    3: "Advanced damage",
    4: "High risk of vision loss",
}
CLASS_EXPLANATIONS = {
    0: "no visible signs of diabetic retinopathy in the retinal image",
    1: "early changes such as a small number of microaneurysms",
    2: "the presence of visible changes in the retinal image, such as microaneurysms or exudates, "
       "which are commonly associated with moderate DR",
    3: "extensive haemorrhages, venous beading or intraretinal abnormalities that are commonly associated with "
       "severe non-proliferative DR",
    4: "advanced changes such as abnormal new vessel growth (neovascularisation) that are commonly associated with "
       "proliferative DR",
}

# --- Model settings --------------------------------------------------------------------
IMAGE_SIZE = 224
USE_CLAHE = True
MODEL_BACKBONES = "ResNet-50 + DenseNet-121"
MODEL_CLASSIFIER = "AdaBoost + Random Forest"

# --- Misc --------------------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "DR_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
).split(",") if o.strip()]
FRONTEND_DIST = Path(os.getenv("DR_FRONTEND_DIST", BASE_DIR.parent / "frontend" / "dist"))


def ensure_dirs() -> None:
    for d in (STORAGE_DIR, UPLOAD_DIR, HEATMAP_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
