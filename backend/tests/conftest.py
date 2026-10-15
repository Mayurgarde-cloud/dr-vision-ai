import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient wired to a throw-away storage folder and DB, in demo mode."""
    from app import config
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "HEATMAP_DIR", tmp_path / "heatmaps")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "dr_model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "models" / "metrics.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "no_dataset")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
