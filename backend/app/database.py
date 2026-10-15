"""SQLite persistence for prediction history."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    image_file      TEXT NOT NULL,
    heatmap_file    TEXT,
    overlay_file    TEXT,
    class_index     INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    probabilities   TEXT NOT NULL,
    processing_time REAL NOT NULL,
    mode            TEXT NOT NULL,
    model_version   TEXT NOT NULL DEFAULT 'unknown',
    warnings        TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        if "model_version" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT NOT NULL DEFAULT 'unknown'")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["probabilities"] = json.loads(d["probabilities"])
    d["warnings"] = json.loads(d.get("warnings") or "[]")
    return d


def insert_prediction(*, filename: str, image_file: str, heatmap_file: Optional[str], overlay_file: Optional[str],
                      class_index: int, class_name: str, confidence: float, probabilities: list[float],
                      processing_time: float, mode: str, warnings: list[str],
                      model_version: str = "unknown") -> int:
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (filename, image_file, heatmap_file, overlay_file, class_index, class_name, "
            "confidence, probabilities, processing_time, mode, model_version, warnings, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (filename, image_file, heatmap_file, overlay_file, class_index, class_name, confidence,
             json.dumps(probabilities), processing_time, mode, model_version, json.dumps(warnings), created),
        )
        return int(cur.lastrowid)


def get_prediction(pred_id: int) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_predictions(search: str = "", limit: int = 200) -> list[dict[str, Any]]:
    sql, args = "SELECT * FROM predictions", []
    if search.strip():
        like = f"%{search.strip().lower()}%"
        sql += (" WHERE lower(class_name) LIKE ? OR lower(created_at) LIKE ? OR lower(filename) LIKE ?"
                " OR CAST(id AS TEXT) LIKE ?")
        args += [like, like, like, like]
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_predictions() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def delete_prediction(pred_id: int) -> Optional[dict[str, Any]]:
    row = get_prediction(pred_id)
    if row is None:
        return None
    with connect() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    return row


def delete_all_predictions() -> list[dict[str, Any]]:
    rows = list_predictions(limit=1_000_000)
    with connect() as conn:
        conn.execute("DELETE FROM predictions")
    return rows
