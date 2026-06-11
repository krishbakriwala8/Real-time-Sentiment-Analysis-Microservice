"""
app/db/models.py
────────────────
SQLAlchemy ORM table definitions.

Tables
──────
predictions        — every inference request + result, for audit and analytics
benchmark_runs     — summary row for each benchmarking execution
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Predictions table ────────────────────────────────────────────────────────

class Prediction(Base):
    """Persists every inference call for audit, retraining, and analytics."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Store only the first 500 chars to keep rows small; full text is ephemeral
    text_snippet: Mapped[str] = mapped_column(String(500), nullable=False)
    full_text_length: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)     # POSITIVE / NEGATIVE / UNCERTAIN
    score: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    is_batch: Mapped[bool] = mapped_column(Boolean, default=False)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} label={self.label} "
            f"score={self.score:.3f} latency={self.latency_ms:.1f}ms>"
        )


# ─── Benchmark runs table ─────────────────────────────────────────────────────

class BenchmarkRun(Base):
    """One row per benchmark execution — used for longitudinal monitoring."""

    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    p50_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    p95_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    p99_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    max_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_under_target: Mapped[float] = mapped_column(Float, nullable=False)
    target_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=_utcnow,
    )
