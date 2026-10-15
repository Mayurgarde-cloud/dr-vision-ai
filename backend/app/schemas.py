"""Pydantic response models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ClassProbability(BaseModel):
    index: int
    name: str
    probability: float  # 0-100 (percent)


class ModelInfo(BaseModel):
    backbones: str
    classifier: str
    image_size: str
    mode: str          # "trained" | "demo"
    mode_note: Optional[str] = None
    version: str = "unknown"
    score_description: str = "Classifier score; not a probability of actual disease."


class PredictionOut(BaseModel):
    id: int
    filename: str
    class_index: int
    class_name: str
    confidence: float  # percent
    probabilities: list[ClassProbability]
    processing_time: float
    created_at: str
    image_url: str
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    explanation: str
    warnings: list[str] = []
    model: ModelInfo


class HistoryItem(BaseModel):
    id: int
    filename: str
    class_index: int
    class_name: str
    confidence: float
    created_at: str
    image_url: str
    mode: str
    model_version: str = "unknown"


class MessageOut(BaseModel):
    message: str
    deleted: int = 0
