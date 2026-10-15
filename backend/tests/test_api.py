import io
import hashlib

from PIL import Image

from ml.synthetic import make_fundus


def _png_bytes(sev=2, seed=1, size=320, fmt="PNG"):
    buf = io.BytesIO()
    make_fundus(sev, seed=seed, size=size).save(buf, fmt)
    return buf.getvalue()


def _upload(client, data, name="eye.png", ctype="image/png"):
    return client.post("/api/predict", files={"file": (name, data, ctype)})


def test_health_reports_demo_mode(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["model_mode"] == "demo"


def test_predict_returns_full_result(client):
    r = _upload(client, _png_bytes())
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0 <= d["class_index"] <= 4
    assert len(d["probabilities"]) == 5
    assert abs(sum(p["probability"] for p in d["probabilities"]) - 100) < 0.5
    assert d["confidence"] == max(p["probability"] for p in d["probabilities"])
    assert [p["index"] for p in d["probabilities"]] == list(range(5))
    assert d["probabilities"][d["class_index"]]["probability"] == d["confidence"]
    assert d["model"]["mode"] == "demo" and d["model"]["mode_note"]
    assert d["model"]["version"] == "demo"
    assert "not a calibrated probability" in d["model"]["score_description"]
    for key in ("image_url", "heatmap_url", "overlay_url"):
        assert client.get(d[key]).status_code == 200


def test_distinct_uploads_keep_prediction_and_history_bound_to_request(client):
    first = _upload(client, _png_bytes(0, seed=11), "first.png").json()
    second = _upload(client, _png_bytes(4, seed=29), "second.png").json()

    assert first["id"] != second["id"]
    assert first["filename"] == "first.png"
    assert second["filename"] == "second.png"
    first_image = client.get(first["image_url"]).content
    second_image = client.get(second["image_url"]).content
    assert hashlib.sha256(first_image).digest() != hashlib.sha256(second_image).digest()

    for posted in (first, second):
        saved = client.get(f"/api/predictions/{posted['id']}")
        assert saved.status_code == 200
        saved = saved.json()
        assert saved["id"] == posted["id"]
        assert saved["filename"] == posted["filename"]
        assert saved["class_index"] == posted["class_index"]
        assert saved["probabilities"] == posted["probabilities"]


def test_demo_estimator_orders_severity(client):
    healthy = _upload(client, _png_bytes(0)).json()["class_index"]
    severe = _upload(client, _png_bytes(4)).json()["class_index"]
    assert severe > healthy


def test_history_search_get_delete_clear(client):
    ids = [_upload(client, _png_bytes(s, seed=s)).json()["id"] for s in (0, 2, 4)]
    hist = client.get("/api/predictions").json()
    assert [h["id"] for h in hist] == sorted(ids, reverse=True)
    assert all(h["model_version"] == "demo" for h in hist)
    name = hist[0]["class_name"]
    assert all(h["class_name"] == name for h in client.get("/api/predictions", params={"search": name}).json())
    assert client.get(f"/api/predictions/{ids[0]}").status_code == 200
    assert client.delete(f"/api/predictions/{ids[0]}").json()["deleted"] == 1
    assert client.get(f"/api/predictions/{ids[0]}").status_code == 404
    assert client.delete("/api/predictions").json()["deleted"] == 2
    assert client.get("/api/predictions").json() == []


def test_pdf_report(client):
    pid = _upload(client, _png_bytes()).json()["id"]
    r = client.get(f"/api/predictions/{pid}/report")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and len(r.content) > 3000
    assert client.get("/api/predictions/9999/report").status_code == 404


def test_rejects_bad_uploads(client):
    assert _upload(client, b"hello", "a.txt", "text/plain").status_code == 400
    assert _upload(client, b"not an image", "a.png", "image/png").status_code == 400
    assert _upload(client, b"", "a.png", "image/png").status_code == 400
    big = b"\x89PNG" + b"0" * (5 * 1024 * 1024 + 10)
    assert _upload(client, big).status_code == 413
    tiny = io.BytesIO()
    Image.new("RGB", (20, 20)).save(tiny, "PNG")
    assert _upload(client, tiny.getvalue()).status_code == 400
    trunc = _png_bytes()[:400]
    assert _upload(client, trunc).status_code == 400


def test_non_fundus_image_gets_warning(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (40, 120, 220)).save(buf, "JPEG")
    d = _upload(client, buf.getvalue(), "sky.jpg", "image/jpeg").json()
    assert d["warnings"]


def test_dashboard_dataset_and_performance_before_training(client):
    d = client.get("/api/dashboard").json()
    assert d["model"]["status"] == "Demo Mode" and d["total_predictions"] == 0
    _upload(client, _png_bytes())
    d = client.get("/api/dashboard").json()
    assert d["total_predictions"] == 1 and len(d["recent"]) == 1 and d["last_prediction"]
    ds = client.get("/api/dataset").json()
    assert ds["status"] == "Not found" and ds["total"] == 0 and len(ds["classes"]) == 5
    perf = client.get("/api/model/performance").json()
    assert perf["evaluated"] is False
