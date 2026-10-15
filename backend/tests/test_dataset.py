from pathlib import Path

import pandas as pd
import pytest

from ml.dataset import class_counts, folder_to_label, read_csv_labels, scan_dataset, summarize
from ml.synthetic import make_fundus


def test_folder_names_map_to_labels():
    cases = {"0": 0, "0_No_DR": 0, "No_DR": 0, "normal": 0, "1_Mild": 1, "Mild_DR": 1, "2": 2, "Moderate": 2,
             "3_Severe_DR": 3, "Severe": 3, "4_Proliferative_DR": 4, "Proliferative_DR": 4}
    for name, label in cases.items():
        assert folder_to_label(name) == label, name
    assert folder_to_label("misc") is None and folder_to_label("5") is None


def test_scan_and_summary(tmp_path):
    for sev, n in enumerate([3, 2, 0, 1, 1]):
        d = tmp_path / f"{sev}_class"
        d.mkdir()
        for i in range(n):
            make_fundus(sev, seed=i, size=96).save(d / f"{i}.jpg")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "x.jpg").write_bytes(b"x")
    df = scan_dataset(tmp_path)
    assert class_counts(df) == [3, 2, 0, 1, 1]
    s = summarize(df)
    assert s["total"] == 7 and s["classes"][0]["percent"] == round(100 * 3 / 7, 1)
    assert scan_dataset(tmp_path / "missing").empty


def test_csv_labels_aptos_style(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i, sev in enumerate([0, 2, 4]):
        make_fundus(sev, seed=i, size=96).save(imgs / f"id{i}.png")
    pd.DataFrame({"id_code": ["id0", "id1", "id2"], "diagnosis": [0, 2, 4]}).to_csv(tmp_path / "t.csv", index=False)
    df = read_csv_labels(tmp_path / "t.csv", imgs)
    assert sorted(df["label"]) == [0, 2, 4] and all(Path(p).exists() for p in df["path"])


@pytest.mark.parametrize("ids,labels,error", [
    (["id0", "missing"], [0, 1], "Missing 1 image"),
    (["id0", "id0"], [0, 1], "Duplicate image IDs"),
    (["id0", "id1"], [0, 5], "Invalid labels"),
    (["id0", "id1"], [0, 1.5], "Invalid labels"),
])
def test_csv_labels_reject_incomplete_or_invalid_rows(tmp_path, ids, labels, error):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    make_fundus(0, size=96).save(imgs / "id0.png")
    make_fundus(1, size=96).save(imgs / "id1.png")
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"id_code": ids, "diagnosis": labels}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match=error):
        read_csv_labels(csv, imgs)
