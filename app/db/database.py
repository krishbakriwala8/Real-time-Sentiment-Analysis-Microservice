"""
app/db/database.py
──────────────────
Async SQLAlchemy engine and session factory.
All I/O is async (asyncpg driver) to avoid blocking the FastAPI event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import settings


# ─── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # recycle stale connections automatically
    echo=settings.app_env == "development",
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # keep objects usable after commit
    autoflush=False,
    autocommit=False,
)


# ─── Base declarative class ────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── Dependency injection helper ──────────────────────────────────────────────

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context-manager version (used by CRUD helpers and scripts)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── Startup helpers ───────────────────────────────────────────────────────────

async def create_tables() -> None:
    """Create all tables defined in ORM models (idempotent)."""
    from app.db import models  # noqa: F401  — register models before create_all
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")


async def check_connection() -> bool:
    """Ping the DB — used in the /health endpoint."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning(f"DB health check failed: {exc}")
        return False


async def close_db() -> None:
    await engine.dispose()
    logger.info("Database engine disposed.")
