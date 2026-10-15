import numpy as np
import pytest

from ml.dataset import assert_content_disjoint, assert_content_labels_consistent, image_content_hash
from ml.train import stratified_split


def test_stratified_split_keeps_exact_image_copies_together(tmp_path):
    paths, labels = [], []
    for label in range(5):
        for index in range(8):
            path = tmp_path / f"class{label}_{index}.img"
            path.write_bytes(f"unique-image-{label}-{index}".encode())
            paths.append(path)
            labels.append(label)
            if index == 0:
                duplicate = tmp_path / f"class{label}_duplicate.img"
                duplicate.write_bytes(path.read_bytes())
                paths.append(duplicate)
                labels.append(label)

    groups = np.asarray(assert_content_labels_consistent(paths, labels))
    first = stratified_split(np.asarray(labels), 0.15, 0.15, 42, groups)
    second = stratified_split(np.asarray(labels), 0.15, 0.15, 42, groups)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    group_splits = [set(groups[idx]) for idx in first]
    assert not (group_splits[0] & group_splits[1])
    assert not (group_splits[0] & group_splits[2])
    assert not (group_splits[1] & group_splits[2])


def test_exact_duplicate_conflicting_labels_are_rejected(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same retinal image")
    b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="conflicting labels"):
        assert_content_labels_consistent([a, b], [1, 2])
    with pytest.raises(ValueError, match="content leakage"):
        assert_content_disjoint([a], [b])
