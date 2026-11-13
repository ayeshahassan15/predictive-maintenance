from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID = {
    "cycle": 1, "sensor_2": 641.82, "sensor_3": 1589.7, "sensor_4": 1400.6,
    "sensor_7": 554.36, "sensor_8": 2388.06, "sensor_9": 9046.19,
    "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.02,
    "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_17": 392,
    "sensor_20": 39.06, "sensor_21": 23.419,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict-rul", json={"readings": [VALID]})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_rul"] <= 125
    assert body["cycles_used"] == 1


def test_predict_missing_sensor():
    bad = {k: v for k, v in VALID.items() if k != "sensor_21"}
    r = client.post("/predict-rul", json={"readings": [bad]})
    assert r.status_code == 422


def test_predict_empty_readings():
    r = client.post("/predict-rul", json={"readings": []})
    assert r.status_code == 422