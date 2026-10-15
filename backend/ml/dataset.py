"""Dataset discovery helpers (pure pandas – no PyTorch needed).

Supported layouts
-----------------
1. Class sub-folders (any of these folder names work)::

       data/dataset/0_No_DR/*.jpg      data/dataset/No_DR/*.jpg      data/dataset/0/*.jpg
       data/dataset/1_Mild/*.jpg       data/dataset/Mild/*.jpg       ...

2. A CSV of labels (e.g. APTOS 2019 ``train.csv`` with ``id_code,diagnosis``)::

       python -m ml.train --csv path/to/train.csv --image-dir path/to/train_images
"""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from . import CLASS_NAMES, NUM_CLASSES


def image_content_hash(path: Path | str) -> str:
    """SHA-256 of the original image bytes, used to keep duplicate images together."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_content_labels_consistent(paths, labels) -> list[str]:
    """Reject exact duplicate images whose supplied labels disagree."""
    if len(paths) != len(labels):
        raise ValueError("Image paths and labels have different lengths.")
    by_hash: dict[str, set[int]] = {}
    hashes = []
    for path, label in zip(paths, labels):
        digest = image_content_hash(path)
        hashes.append(digest)
        by_hash.setdefault(digest, set()).add(int(label))
    conflicts = [digest for digest, values in by_hash.items() if len(values) > 1]
    if conflicts:
        raise ValueError(f"Found {len(conflicts)} exact-content image group(s) with conflicting labels.")
    return hashes


def assert_content_disjoint(left_paths, right_paths, left_name="train", right_name="validation") -> None:
    """Fail when exact copies of an image occur in two nominally separate splits."""
    left_hashes = {image_content_hash(path) for path in left_paths}
    right_hashes = {image_content_hash(path) for path in right_paths}
    overlap = left_hashes & right_hashes
    if overlap:
        raise ValueError(f"{left_name}/{right_name} content leakage: {len(overlap)} exact duplicate image group(s).")


def unique_content_indices(paths, labels) -> tuple[list[int], int]:
    """Return one row per exact image, rejecting conflicting duplicate labels."""
    hashes = assert_content_labels_consistent(paths, labels)
    seen: set[str] = set()
    keep: list[int] = []
    for i, digest in enumerate(hashes):
        if digest not in seen:
            seen.add(digest)
            keep.append(i)
    return keep, len(paths) - len(keep)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def folder_to_label(name: str) -> Optional[int]:
    low = name.lower().strip()
    m = re.match(r"^([0-4])(?:[_\-\s]|$)", low)
    if m:
        return int(m.group(1))
    if "prolif" in low:
        return 4
    if "severe" in low:
        return 3
    if "moderate" in low:
        return 2
    if "mild" in low:
        return 1
    if low.startswith("no") or "healthy" in low or "normal" in low:
        return 0
    return None


def scan_dataset(data_dir: Path | str) -> pd.DataFrame:
    """Return a DataFrame(path, label) for a class-folder dataset. Empty if nothing found."""
    root = Path(data_dir)
    rows: list[tuple[str, int]] = []
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            label = folder_to_label(sub.name)
            if label is None:
                continue
            for f in sorted(sub.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS and f.is_file():
                    rows.append((str(f), label))
    return pd.DataFrame(rows, columns=["path", "label"])


def read_csv_labels(csv_path: Path | str, image_dir: Path | str) -> pd.DataFrame:
    """Read and validate an APTOS/EyePACS style label CSV.

    Do not silently discard bad rows: a partial dataset can change split balance
    and make training/evaluation results look valid when labels were omitted.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("id_code", "image", "image_id", "filename", "file", "path", "id") if c in cols), None)
    lab_col = next((cols[c] for c in ("diagnosis", "level", "label", "dr_grade", "grade", "class") if c in cols), None)
    if id_col is None or lab_col is None:
        raise ValueError(f"Could not find id/label columns in {csv_path}. Columns: {list(df.columns)}")
    if df.empty:
        raise ValueError(f"No labeled rows found in {csv_path}.")

    identifiers = df[id_col].astype(str).str.strip()
    if identifiers.eq("").any() or identifiers.str.lower().eq("nan").any():
        rows = (np.flatnonzero((identifiers == "") | identifiers.str.lower().eq("nan"))[:10] + 2).tolist()
        raise ValueError(f"Empty image IDs in {csv_path} at CSV row(s): {rows}.")
    duplicate_ids = identifiers[identifiers.duplicated(keep=False)].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate image IDs in {csv_path}: {duplicate_ids[:10]}.")

    numeric_labels = pd.to_numeric(df[lab_col], errors="coerce")
    bad_labels = numeric_labels.isna() | (numeric_labels % 1 != 0) | ~numeric_labels.between(0, NUM_CLASSES - 1)
    if bad_labels.any():
        examples = [(identifiers.iloc[i], str(df[lab_col].iloc[i]))
                    for i in np.flatnonzero(bad_labels.to_numpy())[:10]]
        raise ValueError(f"Invalid labels in {csv_path}; expected integer values 0-{NUM_CLASSES - 1}: {examples}.")

    image_dir = Path(image_dir)
    rows: list[tuple[str, int]] = []
    missing: list[str] = []
    for ident, lab in zip(identifiers, numeric_labels.astype(int)):
        p = image_dir / ident
        if not p.suffix:
            for ext in (".png", ".jpg", ".jpeg"):
                if (image_dir / f"{ident}{ext}").exists():
                    p = image_dir / f"{ident}{ext}"
                    break
        if p.is_file():
            rows.append((str(p), int(lab)))
        else:
            missing.append(ident)
    if missing:
        raise ValueError(f"Missing {len(missing)} image file(s) referenced by {csv_path}: {missing[:10]}.")
    return pd.DataFrame(rows, columns=["path", "label"])


def class_counts(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return [0] * NUM_CLASSES
    counts = df["label"].value_counts().to_dict()
    return [int(counts.get(i, 0)) for i in range(NUM_CLASSES)]


def summarize(df: pd.DataFrame) -> dict:
    counts = class_counts(df)
    total = int(sum(counts))
    return {
        "total": total,
        "classes": [
            {
                "index": i,
                "name": CLASS_NAMES[i],
                "count": counts[i],
                "percent": round(100.0 * counts[i] / total, 1) if total else 0.0,
            }
            for i in range(NUM_CLASSES)
        ],
    }
