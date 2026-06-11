"""
tests/test_sentiment.py
───────────────────────
Unit and integration tests covering:
  • Pydantic validation (schemas layer)
  • Model inference (model layer)
  • API endpoints (API layer) via TestClient
  • Benchmarking evaluator (benchmarking layer)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import (
    BatchSentimentRequest,
    SentimentRequest,
    SentimentResult,
)
from app.benchmarking.evaluator import BenchmarkConfig, SentimentEvaluator
from app.model.sentiment import InferenceResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """A SentimentModel mock that avoids downloading BERT during tests."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "test-model-v1"
    model.predict.return_value = InferenceResult(
        label="POSITIVE", score=0.98, latency_ms=45.0
    )
    model.predict_batch.return_value = [
        InferenceResult(label="POSITIVE", score=0.98, latency_ms=45.0),
        InferenceResult(label="NEGATIVE", score=0.91, latency_ms=42.0),
    ]
    return model


@pytest.fixture
def client(mock_model) -> Generator:
    """FastAPI TestClient with model and DB dependencies mocked out."""
    # Ensure routes is imported so patch targets resolve correctly
    import app.api.routes  # noqa: F401

    with (
        patch("app.model.sentiment.sentiment_model", mock_model),
        patch("app.api.routes.sentiment_model", mock_model),
        patch("app.api.routes.create_prediction") as mock_create,
        patch("app.api.routes.list_predictions") as mock_list,
        patch("app.api.routes.check_connection", return_value=True),
        # Prevent the lifespan from downloading the real BERT model or
        # connecting to a real database.
        patch("app.main.sentiment_model", mock_model),
        patch("app.main.create_tables", return_value=None),
        patch("app.main.close_db", return_value=None),
    ):
        # Make create_prediction return a minimal mock record
        async def _fake_create_prediction(db, **kwargs):
            rec = MagicMock()
            rec.id = uuid.uuid4()
            return rec

        mock_create.side_effect = _fake_create_prediction
        mock_list.return_value = ([], 0)

        from app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ─── Schema validation tests ───────────────────────────────────────────────────

class TestSchemas:
    def test_valid_request(self):
        req = SentimentRequest(text="Great product!")
        assert req.text == "Great product!"

    def test_strips_whitespace(self):
        req = SentimentRequest(text="  hello  ")
        assert req.text == "hello"

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            SentimentRequest(text="")

    def test_whitespace_only_raises(self):
        with pytest.raises(Exception):
            SentimentRequest(text="   ")

    def test_text_too_long_raises(self):
        with pytest.raises(Exception):
            SentimentRequest(text="a" * 5_001)

    def test_valid_batch_request(self):
        req = BatchSentimentRequest(texts=["Good", "Bad"])
        assert len(req.texts) == 2

    def test_batch_too_many_raises(self):
        with pytest.raises(Exception):
            BatchSentimentRequest(texts=["text"] * 65)

    def test_batch_empty_entry_raises(self):
        with pytest.raises(Exception):
            BatchSentimentRequest(texts=["Good", ""])

    def test_sentiment_result_score_range(self):
        result = SentimentResult(label="POSITIVE", score=0.95, latency_ms=50.0)
        assert 0.0 <= result.score <= 1.0


# ─── Model layer tests ────────────────────────────────────────────────────────

class TestModelLayer:
    def test_inference_result_fields(self):
        result = InferenceResult(label="POSITIVE", score=0.99, latency_ms=30.0)
        assert result.label == "POSITIVE"
        assert result.score == 0.99
        assert result.latency_ms == 30.0

    def test_model_predict_called(self, mock_model):
        result = mock_model.predict("I love this!")
        mock_model.predict.assert_called_once_with("I love this!")
        assert result.label == "POSITIVE"

    def test_batch_predict_called(self, mock_model):
        results = mock_model.predict_batch(["Good", "Bad"])
        assert len(results) == 2
        assert results[0].label == "POSITIVE"
        assert results[1].label == "NEGATIVE"

    def test_unloaded_model_raises(self):
        from app.model.sentiment import SentimentModel
        m = SentimentModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            m.predict("test")


# ─── API endpoint tests ────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/api/v1/predict", json={"text": "I love this product!"})
        assert resp.status_code == 200

    def test_predict_response_shape(self, client):
        resp = client.post("/api/v1/predict", json={"text": "Great!"})
        data = resp.json()
        assert "id" in data
        assert "result" in data
        assert data["result"]["label"] in ("POSITIVE", "NEGATIVE", "UNCERTAIN")
        assert 0.0 <= data["result"]["score"] <= 1.0
        assert data["result"]["latency_ms"] >= 0

    def test_predict_empty_text_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_missing_text_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_predict_too_long_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"text": "a" * 5_001})
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_returns_200(self, client):
        resp = client.post(
            "/api/v1/predict/batch", json={"texts": ["Good", "Bad"]}
        )
        assert resp.status_code == 200

    def test_batch_response_shape(self, client):
        resp = client.post(
            "/api/v1/predict/batch", json={"texts": ["Great!", "Terrible!"]}
        )
        data = resp.json()
        assert "batch_id" in data
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["total_latency_ms"] >= 0
        assert data["avg_latency_ms"] >= 0

    def test_batch_too_many_returns_422(self, client):
        resp = client.post(
            "/api/v1/predict/batch", json={"texts": ["text"] * 65}
        )
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_fields(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_connected" in data
        assert "uptime_seconds" in data


# ─── Benchmarking evaluator tests ─────────────────────────────────────────────

class TestEvaluator:
    def test_latency_stats_computed(self, mock_model):
        config = BenchmarkConfig(sample_size=10, target_latency_ms=200.0, include_charts=False)
        evaluator = SentimentEvaluator(mock_model, config)
        texts = ["test text"] * 10
        report = evaluator.run(texts)

        assert report.latency.mean >= 0
        assert report.latency.p95 >= report.latency.p50
        assert report.latency.p99 >= report.latency.p95
        assert 0.0 <= report.latency.pct_under_target <= 100.0

    def test_accuracy_computed_when_labels_provided(self, mock_model):
        config = BenchmarkConfig(sample_size=5, target_latency_ms=200.0, include_charts=False)
        mock_model.predict.return_value = InferenceResult(
            label="POSITIVE", score=0.98, latency_ms=45.0
        )
        evaluator = SentimentEvaluator(mock_model, config)
        texts = ["test"] * 5
        labels = [1, 1, 1, 1, 1]   # all POSITIVE — should be 100%

        report = evaluator.run(texts, labels=labels)
        assert report.accuracy is not None
        assert report.accuracy.accuracy == 1.0

    def test_label_mismatch_raises(self, mock_model):
        config = BenchmarkConfig(sample_size=3, target_latency_ms=200.0, include_charts=False)
        evaluator = SentimentEvaluator(mock_model, config)
        with pytest.raises(ValueError, match="labels length"):
            evaluator.run(["a", "b", "c"], labels=[1, 0])  # wrong length

    def test_passes_sla_flag(self, mock_model):
        """Model returns 45ms latency — should pass 200ms SLA."""
        config = BenchmarkConfig(sample_size=5, target_latency_ms=200.0, include_charts=False)
        evaluator = SentimentEvaluator(mock_model, config)
        report = evaluator.run(["text"] * 5)
        assert report.passes_sla is True

    def test_fails_sla_flag(self, mock_model):
        """Set a very tight 10ms target — 45ms latency should fail."""
        mock_model.predict.return_value = InferenceResult(
            label="POSITIVE", score=0.98, latency_ms=45.0
        )
        config = BenchmarkConfig(sample_size=5, target_latency_ms=10.0, include_charts=False)
        evaluator = SentimentEvaluator(mock_model, config)
        report = evaluator.run(["text"] * 5)
        assert report.passes_sla is False

    def test_empty_texts_raises(self, mock_model):
        config = BenchmarkConfig(sample_size=0, include_charts=False)
        evaluator = SentimentEvaluator(mock_model, config)
        with pytest.raises(ValueError, match="non-empty"):
            evaluator.run([])
