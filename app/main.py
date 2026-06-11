"""
app/main.py
───────────
FastAPI application factory.

Lifespan (startup / shutdown):
  • Load BERT model into memory once
  • Create PostgreSQL tables
  • Warm up inference pipeline
  • Graceful shutdown frees GPU/CPU memory

This file is the entry point for uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routes import router
from app.config import settings
from app.db.database import close_db, create_tables
from app.model.sentiment import sentiment_model
from app import state


# ─── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once at startup and once at shutdown.
    Model loading and DB table creation happen here, not at import time,
    so tests can import the app without triggering heavy I/O.
    """
    logger.info("=" * 60)
    logger.info("  Starting Real-Time Sentiment Microservice")
    logger.info(f"  Environment : {settings.app_env}")
    logger.info(f"  Model       : {settings.active_model_path}")
    logger.info("=" * 60)

    # 1. Ensure DB schema exists
    try:
        await create_tables()
    except Exception as exc:
        logger.warning(f"DB table creation failed (may be expected in CI): {exc}")

    # 2. Load BERT model
    try:
        sentiment_model.load()
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        raise

    state.startup_time = time.time()
    logger.info("Service is ready to accept requests ✅")

    yield   # ← application serves requests here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    sentiment_model.unload()
    await close_db()
    logger.info("Shutdown complete.")


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Real-Time Sentiment Microservice",
        description=(
            "Production-grade sentiment analysis API built with FastAPI + BERT.\n\n"
            "**Features**\n"
            "- Fine-tuned DistilBERT achieving ~90% accuracy on SST-2\n"
            "- <200 ms p95 latency under real traffic\n"
            "- PostgreSQL persistence for audit and retraining\n"
            "- Built-in benchmarking and performance monitoring\n"
            "- Fully Dockerised with docker-compose"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        if elapsed_ms > settings.latency_target_ms:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {elapsed_ms:.1f}ms (target: {settings.latency_target_ms}ms)"
            )
        return response

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.error(f"RuntimeError: {exc}")
        return JSONResponse(
            status_code=503,
            content={"detail": "Service temporarily unavailable"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Sentiment Microservice — see /docs for the API reference"}

    return app


app = create_app()
