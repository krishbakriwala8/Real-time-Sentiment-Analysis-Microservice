"""
app/db/crud.py
──────────────
All database read/write operations in one place.
Routes and services import these functions — they never write raw SQL elsewhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BenchmarkRun, Prediction


# ─── Predictions ──────────────────────────────────────────────────────────────

async def create_prediction(
    db: AsyncSession,
    *,
    text: str,
    label: str,
    score: float,
    latency_ms: float,
    model_version: str,
    is_batch: bool = False,
    batch_id: Optional[uuid.UUID] = None,
) -> Prediction:
    record = Prediction(
        id=uuid.uuid4(),
        text_snippet=text[:500],
        full_text_length=len(text),
        label=label,
        score=score,
        latency_ms=latency_ms,
        model_version=model_version,
        is_batch=is_batch,
        batch_id=batch_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    await db.flush()   # get DB-generated fields without committing
    return record


async def get_prediction(db: AsyncSession, prediction_id: uuid.UUID) -> Optional[Prediction]:
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    return result.scalar_one_or_none()


async def list_predictions(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    label_filter: Optional[str] = None,
) -> tuple[list[Prediction], int]:
    """Returns (items, total_count)."""
    query = select(Prediction).order_by(desc(Prediction.created_at))
    count_query = select(func.count()).select_from(Prediction)

    if label_filter:
        query = query.where(Prediction.label == label_filter.upper())
        count_query = count_query.where(Prediction.label == label_filter.upper())

    total = (await db.execute(count_query)).scalar_one()
    items = (
        await db.execute(query.offset((page - 1) * page_size).limit(page_size))
    ).scalars().all()

    return list(items), total


# ─── Benchmark runs ───────────────────────────────────────────────────────────

async def create_benchmark_run(
    db: AsyncSession,
    *,
    sample_size: int,
    mean_latency_ms: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    p99_latency_ms: float,
    max_latency_ms: float,
    pct_under_target: float,
    target_latency_ms: float,
    model_version: str,
    accuracy: Optional[float] = None,
    notes: Optional[str] = None,
) -> BenchmarkRun:
    run = BenchmarkRun(
        id=uuid.uuid4(),
        sample_size=sample_size,
        mean_latency_ms=mean_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        max_latency_ms=max_latency_ms,
        pct_under_target=pct_under_target,
        target_latency_ms=target_latency_ms,
        model_version=model_version,
        accuracy=accuracy,
        notes=notes,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.flush()
    return run


async def list_benchmark_runs(
    db: AsyncSession, limit: int = 10
) -> list[BenchmarkRun]:
    result = await db.execute(
        select(BenchmarkRun).order_by(desc(BenchmarkRun.created_at)).limit(limit)
    )
    return list(result.scalars().all())
