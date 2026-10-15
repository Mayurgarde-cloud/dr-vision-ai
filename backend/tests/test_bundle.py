import numpy as np
import pytest

from ml.bundle import predict_proba


class _Identity:
    def transform(self, values):
        return values


class _Classifier:
    classes_ = np.array([4, 0, 2, 1, 3])

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, values):
        return self.scores[:len(values)]


def _bundle(scores):
    return {"scaler": _Identity(), "pca": _Identity(), "clf": _Classifier(scores)}


def test_probability_scores_map_by_classifier_class_index():
    raw = np.array([[0.05, 0.10, 0.15, 0.20, 0.50]])
    mapped = predict_proba(_bundle(raw), np.zeros((1, 3072), dtype=np.float32))
    assert mapped.shape == (1, 5)
    assert np.allclose(mapped, [[0.10, 0.20, 0.15, 0.50, 0.05]])
    assert np.allclose(mapped.sum(axis=1), 1)


@pytest.mark.parametrize("features,error", [
    (np.zeros((1, 3071)), "shape"),
    (np.full((1, 3072), np.nan), "NaN or infinite"),
    (np.full((1, 3072), np.inf), "NaN or infinite"),
])
def test_probability_pipeline_rejects_invalid_feature_vectors(features, error):
    with pytest.raises(ValueError, match=error):
        predict_proba(_bundle([[0.2] * 5]), features)


@pytest.mark.parametrize("scores,error", [
    ([[0.2, np.nan, 0.2, 0.2, 0.2]], "NaN, infinite"),
    ([[0.2, -0.1, 0.3, 0.3, 0.3]], "negative"),
    ([[0.2, 0.2, 0.2, 0.2, 0.1]], "sum to 1"),
])
def test_probability_pipeline_rejects_invalid_classifier_scores(scores, error):
    with pytest.raises(ValueError, match=error):
        predict_proba(_bundle(scores), np.zeros((1, 3072), dtype=np.float32))
