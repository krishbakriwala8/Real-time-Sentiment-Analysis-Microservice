"""
app/api/routes.py
─────────────────
All FastAPI route handlers.

Endpoints
─────────
POST  /predict          — single-text sentiment analysis
POST  /predict/batch    — batch inference (up to 64 texts)
GET   /predictions      — paginated history from DB
GET   /predictions/{id} — single prediction by ID
POST  /benchmark        — run a live benchmark and store results
GET   /benchmark/runs   — list past benchmark runs
GET   /health           — liveness + readiness check
GET   /metrics          — Prometheus-compatible metrics
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    BenchmarkSummary,
    HealthResponse,
    PaginatedPredictions,
    PredictionRecord,
    SentimentRequest,
    SentimentResponse,
    SentimentResult,
)
from app.benchmarking.evaluator import BenchmarkConfig, SentimentEvaluator
from app.db.crud import (
    create_benchmark_run,
    create_prediction,
    get_prediction,
    list_benchmark_runs,
    list_predictions,
)
from app.db.database import check_connection, get_db
from app.model.sentiment import sentiment_model
from app import state

router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_sentiment_response(
    text: str,
    label: str,
    score: float,
    latency_ms: float,
    record_id: uuid.UUID,
) -> SentimentResponse:
    return SentimentResponse(
        id=record_id,
        text=text,
        result=SentimentResult(label=label, score=score, latency_ms=latency_ms),
        model_version=sentiment_model.version,
        timestamp=datetime.now(timezone.utc),
    )


# ─── Sentiment endpoints ──────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Single-text sentiment analysis",
    tags=["Sentiment"],
)
async def predict(
    request: SentimentRequest,
    db: AsyncSession = Depends(get_db),
) -> SentimentResponse:
    """
    Analyse the sentiment of a single piece of text.
    Returns POSITIVE, NEGATIVE, or UNCERTAIN with a confidence score.
    Target latency: < 200 ms.
    """
    if not sentiment_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    result = sentiment_model.predict(request.text)
    logger.info(
        f"Prediction: label={result.label} score={result.score:.4f} "
        f"latency={result.latency_ms:.1f}ms"
    )

    record = await create_prediction(
        db,
        text=request.text,
        label=result.label,
        score=result.score,
        latency_ms=result.latency_ms,
        model_version=sentiment_model.version,
    )

    return _build_sentiment_response(
        text=request.text,
        label=result.label,
        score=result.score,
        latency_ms=result.latency_ms,
        record_id=record.id,
    )


@router.post(
    "/predict/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch sentiment analysis (up to 64 texts)",
    tags=["Sentiment"],
)
async def predict_batch(
    request: BatchSentimentRequest,
    db: AsyncSession = Depends(get_db),
) -> BatchSentimentResponse:
    """
    Analyse up to 64 texts in a single call.
    Each result includes per-item latency for benchmarking analysis.
    """
    if not sentiment_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    batch_id = uuid.uuid4()
    t0 = time.perf_counter()
    model_results = sentiment_model.predict_batch(request.texts)
    total_latency_ms = (time.perf_counter() - t0) * 1_000

    responses: list[SentimentResponse] = []
    for text, result in zip(request.texts, model_results):
        record = await create_prediction(
            db,
            text=text,
            label=result.label,
            score=result.score,
            latency_ms=result.latency_ms,
            model_version=sentiment_model.version,
            is_batch=True,
            batch_id=batch_id,
        )
        responses.append(
            _build_sentiment_response(
                text=text,
                label=result.label,
                score=result.score,
                latency_ms=result.latency_ms,
                record_id=record.id,
            )
        )

    avg_latency = total_latency_ms / len(responses) if responses else 0.0

    return BatchSentimentResponse(
        batch_id=batch_id,
        results=responses,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=avg_latency,
        model_version=sentiment_model.version,
        timestamp=datetime.now(timezone.utc),
    )


# ─── Prediction history ───────────────────────────────────────────────────────

@router.get(
    "/predictions",
    response_model=PaginatedPredictions,
    summary="Paginated prediction history",
    tags=["History"],
)
async def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: Optional[str] = Query(None, description="Filter by POSITIVE/NEGATIVE/UNCERTAIN"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedPredictions:
    items, total = await list_predictions(db, page=page, page_size=page_size, label_filter=label)
    return PaginatedPredictions(
        items=[PredictionRecord.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecord,
    summary="Fetch a single prediction by ID",
    tags=["History"],
)
async def get_prediction_by_id(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> PredictionRecord:
    record = await get_prediction(db, prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found",
        )
    return PredictionRecord.model_validate(record)


# ─── Benchmarking endpoints ────────────────────────────────────────────────────

# Diverse real-world sample texts for on-demand benchmarking
_BENCHMARK_TEXTS = [
    "This product is absolutely fantastic — worth every penny!",
    "Terrible experience. Broke after two days and support was useless.",
    "It's okay I guess. Nothing special but does the job.",
    "Outstanding quality and lightning-fast delivery. Will order again.",
    "I've had better. The taste was bland and portion sizes were tiny.",
    "The new update completely ruined the app. I want my old version back.",
    "Surprisingly good for the price. Highly recommend to anyone on a budget.",
    "Slow service, rude staff, and the food was cold. Zero stars.",
    "My expectations were exceeded. Truly a premium experience.",
    "Meh. Not what I expected but not terrible either.",
    "Five stars all the way — flawless from start to finish.",
    "Complete waste of money. I want a refund.",
    "Decent product, a few rough edges but overall happy with it.",
    "Love love love this! My whole family enjoys it every day.",
    "Instructions were confusing and half the parts were missing.",
    "Great concept but poor execution. Could be so much better.",
    "Arrived on time and in perfect condition. Exactly as described.",
    "The worst customer service I've ever encountered. Absolutely shocking.",
    "This is my third purchase and I keep coming back — says it all.",
    "Looks good on paper but falls short in real-world use.",
]


@router.post(
    "/benchmark",
    response_model=BenchmarkSummary,
    summary="Run a live benchmark and persist results",
    tags=["Benchmarking"],
)
async def run_benchmark(
    sample_size: int = Query(100, ge=10, le=500, description="Number of inference calls"),
    db: AsyncSession = Depends(get_db),
) -> BenchmarkSummary:
    """
    Run a systematic performance benchmark against diverse real-world inputs.
    Results are persisted to the database for longitudinal monitoring.
    """
    if not sentiment_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    # Expand the sample set by repeating the reference texts
    repeat = math.ceil(sample_size / len(_BENCHMARK_TEXTS))
    texts = (_BENCHMARK_TEXTS * repeat)[:sample_size]

    config = BenchmarkConfig(
        sample_size=sample_size,
        target_latency_ms=settings.latency_target_ms,
        include_charts=False,   # charts optional in API context
    )

    evaluator = SentimentEvaluator(sentiment_model, config)
    report = evaluator.run(texts)

    # Persist to DB
    run = await create_benchmark_run(
        db,
        sample_size=report.config.sample_size,
        mean_latency_ms=report.latency.mean,
        p50_latency_ms=report.latency.p50,
        p95_latency_ms=report.latency.p95,
        p99_latency_ms=report.latency.p99,
        max_latency_ms=report.latency.max,
        pct_under_target=report.latency.pct_under_target,
        target_latency_ms=report.config.target_latency_ms,
        model_version=report.model_version,
        accuracy=report.accuracy.accuracy if report.accuracy else None,
        notes=f"API-triggered benchmark, {sample_size} samples",
    )

    SentimentEvaluator.print_report(report)

    return BenchmarkSummary(
        sample_size=run.sample_size,
        mean_latency_ms=run.mean_latency_ms,
        p50_latency_ms=run.p50_latency_ms,
        p95_latency_ms=run.p95_latency_ms,
        p99_latency_ms=run.p99_latency_ms,
        max_latency_ms=run.max_latency_ms,
        accuracy=run.accuracy,
        pct_under_target=run.pct_under_target,
        target_latency_ms=run.target_latency_ms,
        timestamp=run.created_at,
    )


@router.get(
    "/benchmark/runs",
    response_model=list[BenchmarkSummary],
    summary="List recent benchmark runs",
    tags=["Benchmarking"],
)
async def get_benchmark_runs(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[BenchmarkSummary]:
    runs = await list_benchmark_runs(db, limit=limit)
    return [
        BenchmarkSummary(
            sample_size=r.sample_size,
            mean_latency_ms=r.mean_latency_ms,
            p50_latency_ms=r.p50_latency_ms,
            p95_latency_ms=r.p95_latency_ms,
            p99_latency_ms=r.p99_latency_ms,
            max_latency_ms=r.max_latency_ms,
            accuracy=r.accuracy,
            pct_under_target=r.pct_under_target,
            target_latency_ms=r.target_latency_ms,
            timestamp=r.created_at,
        )
        for r in runs
    ]


# ─── Health & observability ───────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["Ops"],
)
async def health_check() -> HealthResponse:
    db_ok = await check_connection()
    model_ok = sentiment_model.is_loaded

    status_str = (
        "healthy" if (db_ok and model_ok) else
        "degraded" if (db_ok or model_ok) else
        "unhealthy"
    )

    return HealthResponse(
        status=status_str,
        model_loaded=model_ok,
        db_connected=db_ok,
        uptime_seconds=time.time() - state.startup_time,
        model_version=sentiment_model.version,
    )
