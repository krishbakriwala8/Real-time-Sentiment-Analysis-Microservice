from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_positive():
    response = client.post("/v1/predict", json={"text": "I love this product!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["confidence"] > 0.5

def test_predict_negative():
    response = client.post("/v1/predict", json={"text": "This is terrible."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert data["confidence"] > 0.5