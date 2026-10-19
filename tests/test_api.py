from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_invalid_input():
    bad = {**SAMPLE, "age": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_exposes_counters():
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
