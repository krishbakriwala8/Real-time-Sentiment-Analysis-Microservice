"""
app/benchmarking/evaluator.py
──────────────────────────────
Systematic performance monitoring across diverse real-world inputs.

This module covers the resume bullet:
  "Applied evaluation and benchmarking framework — systematic performance
   monitoring across diverse real-world inputs."

What it measures
────────────────
• Latency distribution: mean, p50, p95, p99, max
• Throughput (requests/second)
• % of predictions meeting the <200 ms SLA
• Accuracy / precision / recall / F1 when ground-truth labels are available
• Confidence score distribution
• Performance across input length buckets (short / medium / long)

Results are saved as JSON + PNG charts and optionally persisted to PostgreSQL.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings
from app.model.sentiment import InferenceResult, SentimentModel


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    sample_size: int = 500
    target_latency_ms: float = 200.0
    include_charts: bool = True
    results_dir: str = "./benchmark_results"


@dataclass
class LatencyStats:
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    min: float
    std_dev: float
    pct_under_target: float
    target_ms: float


@dataclass
class AccuracyStats:
    accuracy: float
    precision_positive: float
    recall_positive: float
    f1_positive: float
    precision_negative: float
    recall_negative: float
    f1_negative: float
    total_samples: int
    correct: int


@dataclass
class BenchmarkReport:
    model_version: str
    timestamp: str
    config: BenchmarkConfig
    latency: LatencyStats
    accuracy: Optional[AccuracyStats]
    throughput_rps: float
    label_distribution: dict[str, int]
    score_stats: dict[str, float]
    length_bucket_latency: dict[str, float]   # "short"|"medium"|"long" → mean ms
    passes_sla: bool                           # p95 < target_latency_ms


# ─── Core evaluator ───────────────────────────────────────────────────────────

class SentimentEvaluator:
    """
    Runs a structured benchmark against a loaded SentimentModel.

    Usage:
        evaluator = SentimentEvaluator(model, config)
        report = evaluator.run(texts, labels=labels)
        evaluator.save(report)
    """

    def __init__(self, model: SentimentModel, config: Optional[BenchmarkConfig] = None):
        self.model = model
        self.config = config or BenchmarkConfig(
            sample_size=settings.benchmark_sample_size,
            target_latency_ms=settings.latency_target_ms,
            results_dir=settings.benchmark_results_dir,
        )

    def run(
        self,
        texts: list[str],
        labels: Optional[list[int]] = None,
    ) -> BenchmarkReport:
        """
        Execute the benchmark.

        Parameters
        ──────────
        texts  : input strings (diverse lengths recommended)
        labels : optional integer labels (0=NEGATIVE, 1=POSITIVE) for accuracy
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        logger.info(f"Starting benchmark: {len(texts)} samples …")
        t_start = time.perf_counter()

        results: list[InferenceResult] = []
        per_text_latencies: list[float] = []

        # Run inference one-by-one for accurate per-item latency
        for text in texts:
            result = self.model.predict(text)
            results.append(result)
            per_text_latencies.append(result.latency_ms)

        total_elapsed = (time.perf_counter() - t_start)
        throughput_rps = len(texts) / total_elapsed

        logger.info(
            f"Benchmark complete. Throughput: {throughput_rps:.1f} req/s"
        )

        # ── Latency stats ──────────────────────────────────────────────────────
        latency = self._compute_latency_stats(per_text_latencies)

        # ── Accuracy stats (if labels provided) ───────────────────────────────
        accuracy_stats = None
        if labels is not None:
            if len(labels) != len(results):
                raise ValueError("labels length must match texts length")
            accuracy_stats = self._compute_accuracy_stats(results, labels)

        # ── Label distribution ────────────────────────────────────────────────
        label_dist: dict[str, int] = {}
        for r in results:
            label_dist[r.label] = label_dist.get(r.label, 0) + 1

        # ── Score distribution ────────────────────────────────────────────────
        scores = [r.score for r in results]
        score_stats = {
            "mean": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }

        # ── Latency by input-length bucket ────────────────────────────────────
        length_bucket_latency = self._bucket_latency_by_length(texts, per_text_latencies)

        return BenchmarkReport(
            model_version=self.model.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
            latency=latency,
            accuracy=accuracy_stats,
            throughput_rps=throughput_rps,
            label_distribution=label_dist,
            score_stats=score_stats,
            length_bucket_latency=length_bucket_latency,
            passes_sla=latency.p95 < self.config.target_latency_ms,
        )

    # ── Latency helpers ────────────────────────────────────────────────────────

    def _compute_latency_stats(self, latencies: list[float]) -> LatencyStats:
        sorted_lats = sorted(latencies)
        n = len(sorted_lats)

        def percentile(p: float) -> float:
            idx = int(p / 100 * n)
            return sorted_lats[min(idx, n - 1)]

        under_target = sum(1 for lat in latencies if lat < self.config.target_latency_ms)
        pct_under = (under_target / n) * 100 if n else 0.0

        return LatencyStats(
            mean=statistics.mean(latencies),
            p50=percentile(50),
            p95=percentile(95),
            p99=percentile(99),
            max=max(latencies),
            min=min(latencies),
            std_dev=statistics.stdev(latencies) if n > 1 else 0.0,
            pct_under_target=pct_under,
            target_ms=self.config.target_latency_ms,
        )

    # ── Accuracy helpers ───────────────────────────────────────────────────────

    def _compute_accuracy_stats(
        self,
        results: list[InferenceResult],
        true_labels: list[int],
    ) -> AccuracyStats:
        # Map InferenceResult labels to integers for comparison
        pred_ints = [
            1 if r.label == "POSITIVE" else 0
            for r in results
        ]

        tp = sum(1 for p, t in zip(pred_ints, true_labels) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(pred_ints, true_labels) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(pred_ints, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_ints, true_labels) if p == 0 and t == 1)

        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_pos    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_pos = (
            2 * precision_pos * recall_pos / (precision_pos + recall_pos)
            if (precision_pos + recall_pos) > 0
            else 0.0
        )

        precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_neg    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_neg = (
            2 * precision_neg * recall_neg / (precision_neg + recall_neg)
            if (precision_neg + recall_neg) > 0
            else 0.0
        )

        correct = tp + tn
        total = len(true_labels)

        return AccuracyStats(
            accuracy=correct / total if total > 0 else 0.0,
            precision_positive=precision_pos,
            recall_positive=recall_pos,
            f1_positive=f1_pos,
            precision_negative=precision_neg,
            recall_negative=recall_neg,
            f1_negative=f1_neg,
            total_samples=total,
            correct=correct,
        )

    # ── Length-bucket analysis ─────────────────────────────────────────────────

    @staticmethod
    def _bucket_latency_by_length(
        texts: list[str], latencies: list[float]
    ) -> dict[str, float]:
        buckets: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
        for text, lat in zip(texts, latencies):
            n = len(text.split())
            if n <= 20:
                buckets["short"].append(lat)
            elif n <= 100:
                buckets["medium"].append(lat)
            else:
                buckets["long"].append(lat)

        return {
            k: (statistics.mean(v) if v else 0.0)
            for k, v in buckets.items()
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, report: BenchmarkReport) -> Path:
        """Save report JSON to disk and optionally generate charts."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = results_dir / f"benchmark_{ts}.json"

        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Benchmark report saved: {report_path}")

        if self.config.include_charts:
            self._generate_charts(report, results_dir, ts)

        return report_path

    def _generate_charts(
        self, report: BenchmarkReport, results_dir: Path, ts: str
    ) -> None:
        """Generate latency and label distribution charts."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")   # headless

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Chart 1 — Latency percentile bar
            lat = report.latency
            labels = ["Mean", "P50", "P95", "P99", "Max"]
            values = [lat.mean, lat.p50, lat.p95, lat.p99, lat.max]
            colors = [
                "green" if v < self.config.target_latency_ms else "red"
                for v in values
            ]
            ax1 = axes[0]
            ax1.bar(labels, values, color=colors)
            ax1.axhline(
                self.config.target_latency_ms,
                color="orange",
                linestyle="--",
                label=f"Target {self.config.target_latency_ms}ms",
            )
            ax1.set_title("Latency Distribution (ms)")
            ax1.set_ylabel("ms")
            ax1.legend()

            # Chart 2 — Label distribution pie
            ax2 = axes[1]
            dist = report.label_distribution
            ax2.pie(
                dist.values(),
                labels=dist.keys(),
                autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336", "#9E9E9E"],
            )
            ax2.set_title("Prediction Label Distribution")

            plt.tight_layout()
            chart_path = results_dir / f"benchmark_{ts}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Charts saved: {chart_path}")
        except ImportError:
            logger.warning("matplotlib not available — skipping chart generation")
        except Exception as exc:
            logger.warning(f"Chart generation failed: {exc}")

    # ── Pretty print ──────────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: BenchmarkReport) -> None:
        """Print a concise summary table to stdout."""
        lat = report.latency
        acc = report.accuracy
        sla = "✅ PASS" if report.passes_sla else "❌ FAIL"

        print("\n" + "=" * 60)
        print(f"  BENCHMARK REPORT — {report.model_version}")
        print("=" * 60)
        print(f"  Samples        : {report.config.sample_size}")
        print(f"  Throughput     : {report.throughput_rps:.1f} req/s")
        print(f"  SLA (<{report.config.target_latency_ms:.0f}ms p95) : {sla}")
        print()
        print("  LATENCY (ms)")
        print(f"    Mean   : {lat.mean:.2f}")
        print(f"    P50    : {lat.p50:.2f}")
        print(f"    P95    : {lat.p95:.2f}")
        print(f"    P99    : {lat.p99:.2f}")
        print(f"    Max    : {lat.max:.2f}")
        print(f"    % < target : {lat.pct_under_target:.1f}%")
        if acc:
            print()
            print("  ACCURACY")
            print(f"    Overall : {acc.accuracy:.4f} ({acc.accuracy*100:.2f}%)")
            print(f"    F1 (POS): {acc.f1_positive:.4f}")
            print(f"    F1 (NEG): {acc.f1_negative:.4f}")
        print()
        print("  LABEL DISTRIBUTION")
        for label, count in report.label_distribution.items():
            pct = count / report.config.sample_size * 100
            print(f"    {label:<12}: {count:>5} ({pct:.1f}%)")
        print()
        print("  LATENCY BY INPUT LENGTH")
        for bucket, ms in report.length_bucket_latency.items():
            print(f"    {bucket:<8}: {ms:.2f} ms")
        print("=" * 60 + "\n")
