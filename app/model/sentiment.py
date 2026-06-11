"""
app/model/sentiment.py
──────────────────────
Model layer — entirely decoupled from HTTP and database concerns.
Responsibilities:
  • Load a (fine-tuned) DistilBERT/BERT model once at startup.
  • Expose synchronous single and batch inference with latency measurement.
  • Return structured results that the API layer converts to response schemas.

The model is loaded once and cached globally so every request reuses the same
in-memory weights — critical for the <200 ms latency target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from app.config import settings


# ─── Data contract returned by this layer ─────────────────────────────────────

@dataclass
class InferenceResult:
    label: str          # "POSITIVE" | "NEGATIVE" | "UNCERTAIN"
    score: float        # confidence in [0, 1]
    latency_ms: float   # wall-clock inference time for this item


# ─── Singleton model wrapper ───────────────────────────────────────────────────

class SentimentModel:
    """
    Wraps a HuggingFace sequence-classification pipeline.

    Loaded once via `load()` at application startup (FastAPI lifespan).
    Thread-safe for read-only inference.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._model_version: str = "not-loaded"
        self._device: str = "cpu"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download / load model weights and warm up the pipeline."""
        model_path = settings.active_model_path
        logger.info(f"Loading model from: {model_path}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Inference device: {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._pipeline = pipeline(
            task="text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if self._device == "cuda" else -1,
            truncation=True,
            max_length=settings.max_sequence_length,
            top_k=None,          # return all label scores
        )

        self._model_version = model_path.split("/")[-1]
        logger.info(f"Model loaded — version: {self._model_version}")

        # Warm-up pass eliminates first-request cold-start latency
        self._warmup()

    def _warmup(self) -> None:
        logger.info("Running model warm-up pass …")
        for _ in range(3):
            self._pipeline("warm up text")
        logger.info("Warm-up complete.")

    def unload(self) -> None:
        """Release model from memory (called at shutdown)."""
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, text: str) -> InferenceResult:
        """Single-text inference. Returns in <200 ms for typical inputs."""
        self._assert_loaded()
        t0 = time.perf_counter()

        with torch.no_grad():
            raw = self._pipeline(text)[0]  # list of {label, score} dicts

        latency_ms = (time.perf_counter() - t0) * 1_000
        return self._parse_raw(raw, latency_ms)

    def predict_batch(self, texts: list[str]) -> list[InferenceResult]:
        """
        Batch inference — processes in chunks of `inference_batch_size`.
        Latency per item is recorded individually for benchmarking granularity.
        """
        self._assert_loaded()
        results: list[InferenceResult] = []
        batch_size = settings.inference_batch_size

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()

            with torch.no_grad():
                raw_batch = self._pipeline(chunk)

            chunk_latency_ms = (time.perf_counter() - t0) * 1_000
            per_item_ms = chunk_latency_ms / len(chunk)

            for raw in raw_batch:
                results.append(self._parse_raw(raw, per_item_ms))

        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_raw(
        self, raw: list[dict] | dict, latency_ms: float
    ) -> InferenceResult:
        """
        Convert raw pipeline output to InferenceResult.
        Handles both `top_k=None` (list) and default (dict) pipeline modes.
        """
        if isinstance(raw, dict):
            # Single dict: {'label': 'POSITIVE', 'score': 0.99}
            label = raw["label"].upper()
            score = raw["score"]
        else:
            # List of dicts when top_k=None — pick the highest-scoring label
            best = max(raw, key=lambda x: x["score"])
            label = best["label"].upper()
            score = best["score"]

        # Normalise HuggingFace label names (e.g. "LABEL_1" → "POSITIVE")
        label = self._normalise_label(label, score)

        # Flag low-confidence predictions
        if score < settings.confidence_threshold:
            label = "UNCERTAIN"

        return InferenceResult(label=label, score=score, latency_ms=latency_ms)

    @staticmethod
    def _normalise_label(label: str, score: float) -> str:
        """
        Map raw HF labels to canonical POSITIVE / NEGATIVE.
        SST-2 already returns these; other models may use LABEL_0 / LABEL_1.
        """
        mapping = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "POSITIVE",
            "NEGATIVE": "NEGATIVE",
            "POSITIVE": "POSITIVE",
            "NEG": "NEGATIVE",
            "POS": "POSITIVE",
        }
        return mapping.get(label, label)

    def _assert_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

    @property
    def version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None


# ─── Global singleton ──────────────────────────────────────────────────────────
# Imported by the API layer and lifespan manager — never instantiated elsewhere.

sentiment_model = SentimentModel()
