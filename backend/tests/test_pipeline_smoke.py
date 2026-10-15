"""Train → evaluate → serve smoke test with random (offline) backbone weights.

It checks wiring and, importantly, that the three stages see the *same* feature space:
a regression test for a bug where unseeded random weights made evaluation disagree with training.
Accuracy values are meaningless here. Skipped automatically when PyTorch is not installed.
"""
import io
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from ml import evaluate, train  # noqa: E402
from ml.synthetic import make_fundus  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    data, out = root / "data", root / "models"
    for sev in range(5):
        d = data / f"{sev}_c"
        d.mkdir(parents=True)
        for i in range(8):
            make_fundus(sev, seed=i, size=128).save(d / f"{i}.jpg")
    rc = train.main(["--data-dir", str(data), "--out-dir", str(out), "--no-pretrained", "--pca-components", "16",
                     "--ada-estimators", "3", "--rf-trees", "20", "--batch-size", "8", "--augment", "1"])
    assert rc == 0
    return out


def test_artifacts_written(trained):
    for name in ("dr_model.joblib", "metrics.json", "splits.json", "dataset_info.json"):
        assert (trained / name).exists(), name
    m = json.loads((trained / "metrics.json").read_text())
    assert m["split"] == {"train": 28, "val": 6, "test": 6}
    assert len(m["validation_metrics"]["confusion_matrix"]) == 5
    assert "n_samples" not in m and not (trained / "features_test.npz").exists()


def test_evaluate_reproduces_training_metrics(trained):
    before = json.loads((trained / "metrics.json").read_text())
    assert evaluate.main(["--model-dir", str(trained), "--batch-size", "8"]) == 0
    after = json.loads((trained / "metrics.json").read_text())
    assert after["confusion_matrix"]
    assert after["validation_metrics"] == before["validation_metrics"]
    assert after["selected_for_test"] is True
    assert evaluate.main(["--model-dir", str(trained), "--batch-size", "8"]) == 2


def test_service_matches_batch_inference(trained, monkeypatch):
    from app import config
    from app.services.model_service import ModelService
    from ml.bundle import predict_proba

    monkeypatch.setattr(config, "MODEL_PATH", trained / "dr_model.joblib")
    svc = ModelService()
    svc.load()
    assert svc.is_trained
    paths = json.loads((trained / "splits.json").read_text())["test"][:3]
    meta = svc.bundle["meta"]
    batch = predict_proba(svc.bundle, svc.extractor.extract_paths(paths, size=meta["image_size"], clahe=meta["clahe"],
                                                                   progress=False))
    from PIL import Image
    for p, expected in zip(paths, batch):
        pred = svc.predict(Image.open(p))
        assert pred.mode == "trained"
        assert np.allclose(pred.probabilities, expected, atol=1e-6)
        assert pred.heatmap.shape == (384, 384) and 0 <= pred.heatmap.min() and pred.heatmap.max() <= 1
