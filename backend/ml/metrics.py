"""Evaluation metrics shared by ``train.py`` and ``evaluate.py``."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import CLASS_NAMES, NUM_CLASSES


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    labels = list(range(NUM_CLASSES))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {"index": i, "name": CLASS_NAMES[i], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])}
            for i in labels
        ],
        "class_distribution": [int(s[i]) for i in labels],
        "n_samples": int(len(y_true)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
