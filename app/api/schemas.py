"""
app/api/schemas.py
──────────────────
All Pydantic models used for request validation and response serialisation.
Keeping these in one place enforces a strict contract between callers and the
API — the "validation layer" called out in the resume bullet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Request schemas ───────────────────────────────────────────────────────────

class SentimentRequest(BaseModel):
    """Single-text sentiment request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5_000,
        description="Text to analyse. 1–5 000 characters.",
        examples=["I absolutely loved the new product launch!"],
    )

    @field_validator("text")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank after stripping whitespace")
        return v


class BatchSentimentRequest(BaseModel):
    """Up to 64 texts in one request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of 1–64 texts.",
    )

    @field_validator("texts")
    @classmethod
    def clean_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("texts list must not contain blank entries")
        return cleaned

    @model_validator(mode="after")
    def check_lengths(self) -> "BatchSentimentRequest":
        for t in self.texts:
            if len(t) > 5_000:
                raise ValueError("Each text must be ≤ 5 000 characters")
        return self


# ─── Response schemas ──────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    """Prediction for a single text."""

    label: Literal["POSITIVE", "NEGATIVE", "UNCERTAIN"]
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")
    latency_ms: float = Field(..., ge=0.0, description="Per-item inference time in ms")


class SentimentResponse(BaseModel):
    """Full response for a single-text request."""

    id: UUID
    text: str
    result: SentimentResult
    model_version: str
    timestamp: datetime


class BatchSentimentResponse(BaseModel):
    """Full response for a batch request."""

    batch_id: UUID
    results: list[SentimentResponse]
    total_latency_ms: float
    avg_latency_ms: float
    model_version: str
    timestamp: datetime


# ─── Database read schemas ─────────────────────────────────────────────────────

class PredictionRecord(BaseModel):
    """Shape of a stored prediction returned from the DB."""

    id: UUID
    text_snippet: str          # first 500 chars only
    label: str
    score: float
    latency_ms: float
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedPredictions(BaseModel):
    items: list[PredictionRecord]
    total: int
    page: int
    page_size: int


# ─── Health / meta schemas ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float
    model_version: str


class BenchmarkSummary(BaseModel):
    sample_size: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    accuracy: Optional[float]
    pct_under_target: float
    target_latency_ms: float
    timestamp: datetime
