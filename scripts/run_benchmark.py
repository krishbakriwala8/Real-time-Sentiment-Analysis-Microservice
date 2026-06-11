"""
scripts/run_benchmark.py
─────────────────────────
Standalone CLI for running a systematic benchmark against the live API or
directly against the model layer.

Usage — against the live API (recommended in production):
    python scripts/run_benchmark.py --mode api --url http://localhost:8000 --samples 500

Usage — directly against the model (useful for local dev without Docker):
    python scripts/run_benchmark.py --mode local --samples 200

The script loads diverse real-world inputs across multiple text categories
(reviews, news excerpts, social posts, long-form text) to ensure the benchmark
is representative of real traffic rather than synthetic edge cases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Diverse real-world test corpus ───────────────────────────────────────────

BENCHMARK_CORPUS = [
    # --- Product reviews ---
    ("This is hands-down the best purchase I've made all year. Absolutely love it!", 1),
    ("Completely disappointed. Fell apart within a week and support ignored me.", 0),
    ("Decent quality for the price. A few minor issues but nothing dealbreaking.", 1),
    ("Five stars. Exceeded every expectation — fast shipping, great packaging.", 1),
    ("Not what was advertised. The photos are misleading and the material is cheap.", 0),
    ("Works exactly as described. Simple, effective, no complaints.", 1),
    ("Returned it immediately. The smell alone was enough to put me off.", 0),
    ("Good product overall, though the instructions could be clearer.", 1),
    # --- Restaurant reviews ---
    ("The food was incredible — every dish bursting with flavour. Will return!", 1),
    ("Waited 45 minutes for cold food. The waiter was rude. Avoid at all costs.", 0),
    ("Nice atmosphere and friendly staff. Food was okay, nothing extraordinary.", 1),
    ("Best pasta I've ever had. The chef clearly knows what they're doing.", 1),
    ("Overpriced, underwhelming, and the service was painfully slow.", 0),
    # --- App / software reviews ---
    ("The new update ruined everything. Crashes constantly. Give me the old version!", 0),
    ("Incredibly intuitive. My productivity has doubled since switching.", 1),
    ("Buggy, slow, and the UI is a mess. How did this pass QA?", 0),
    ("Clean design, fast performance, does exactly what it promises.", 1),
    ("Used to be great. Since the last update it drains my battery in minutes.", 0),
    # --- Movie / book reviews ---
    ("A masterpiece of storytelling. I laughed, I cried — emotionally unforgettable.", 1),
    ("Boring from start to finish. Two hours I'll never get back.", 0),
    ("Solid film. Not groundbreaking but very entertaining.", 1),
    ("The plot had potential but the execution was messy and the ending was awful.", 0),
    # --- Social / opinion ---
    ("I genuinely feel inspired every single day using this. Life-changing.", 1),
    ("Absolute disaster of an experience. I'm furious and want my money back.", 0),
    ("It's fine. Does what it needs to do without any fuss.", 1),
    ("I've recommended this to everyone I know. Cannot say enough good things.", 1),
    # --- Short texts ---
    ("Love it!", 1),
    ("Hate it.", 0),
    ("Meh.", 1),
    ("Perfect!", 1),
    ("Broken.", 0),
    ("Amazing!", 1),
    # --- Long-form texts ---
    (
        "After using this product for three months I can say with confidence that it "
        "represents genuine value for money. The build quality is excellent, the "
        "performance is consistent, and the customer support team resolved my one "
        "query within hours. I was initially hesitant given some negative reviews "
        "but my experience has been entirely positive. I would strongly recommend "
        "this to anyone considering a purchase in this category.",
        1,
    ),
    (
        "Where do I begin? The product arrived damaged, the replacement took three "
        "weeks, and when it finally arrived it didn't work either. The customer "
        "service team seemed entirely unconcerned, offered me a 10% discount as "
        "compensation, and then stopped responding to my emails. I have never "
        "encountered such a poor standard of service from any company. I will be "
        "disputing the charge with my bank and leaving this review everywhere I can.",
        0,
    ),
]


# ─── API benchmark mode ────────────────────────────────────────────────────────

async def run_api_benchmark(base_url: str, sample_size: int) -> None:
    try:
        import httpx
    except ImportError:
        print("httpx is required for API mode. Run: pip install httpx")
        sys.exit(1)

    texts_labels = (BENCHMARK_CORPUS * (sample_size // len(BENCHMARK_CORPUS) + 1))[:sample_size]
    texts = [t for t, _ in texts_labels]
    true_labels = [l for _, l in texts_labels]

    latencies: list[float] = []
    correct = 0
    label_dist: dict[str, int] = {}

    print(f"\nRunning API benchmark: {sample_size} requests → {base_url}\n")

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for i, (text, true_label) in enumerate(zip(texts, true_labels), 1):
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "/api/v1/predict", json={"text": text}
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  [{i}/{sample_size}] ERROR: {exc}")
                continue

            latency_ms = (time.perf_counter() - t0) * 1_000
            latencies.append(latency_ms)

            label = data["result"]["label"]
            label_dist[label] = label_dist.get(label, 0) + 1

            pred_int = 1 if label == "POSITIVE" else 0
            if pred_int == true_label:
                correct += 1

            if i % 50 == 0:
                print(f"  Progress: {i}/{sample_size} ({i/sample_size*100:.0f}%)")

    _print_results(latencies, correct, len(texts), label_dist, target_ms=200.0)


# ─── Local model benchmark mode ───────────────────────────────────────────────

def run_local_benchmark(sample_size: int) -> None:
    from app.model.sentiment import sentiment_model
    from app.benchmarking.evaluator import BenchmarkConfig, SentimentEvaluator

    print(f"\nLoading model for local benchmark ({sample_size} samples) …\n")
    sentiment_model.load()

    texts_labels = (BENCHMARK_CORPUS * (sample_size // len(BENCHMARK_CORPUS) + 1))[:sample_size]
    texts = [t for t, _ in texts_labels]
    labels = [l for _, l in texts_labels]

    config = BenchmarkConfig(
        sample_size=sample_size,
        target_latency_ms=200.0,
        include_charts=True,
        results_dir="./benchmark_results",
    )

    evaluator = SentimentEvaluator(sentiment_model, config)
    report = evaluator.run(texts, labels=labels)
    SentimentEvaluator.print_report(report)
    path = evaluator.save(report)
    print(f"Full report saved to: {path}\n")

    sentiment_model.unload()


# ─── Shared result printer ─────────────────────────────────────────────────────

def _print_results(
    latencies: list[float],
    correct: int,
    total: int,
    label_dist: dict[str, int],
    target_ms: float,
) -> None:
    if not latencies:
        print("No successful requests — cannot compute stats.")
        return

    import statistics
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    def pct(p):
        return sorted_lats[min(int(p / 100 * n), n - 1)]

    under = sum(1 for lat in latencies if lat < target_ms)
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  Requests completed : {n} / {total}")
    print(f"  Accuracy           : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("  LATENCY (ms)")
    print(f"    Mean  : {statistics.mean(latencies):.2f}")
    print(f"    P50   : {pct(50):.2f}")
    print(f"    P95   : {pct(95):.2f}   {'✅' if pct(95) < target_ms else '❌'} (target {target_ms:.0f}ms)")
    print(f"    P99   : {pct(99):.2f}")
    print(f"    Max   : {max(latencies):.2f}")
    print(f"    % < {target_ms:.0f}ms: {under/n*100:.1f}%")
    print()
    print("  LABEL DISTRIBUTION")
    for label, count in sorted(label_dist.items()):
        print(f"    {label:<12}: {count:>5} ({count/n*100:.1f}%)")
    print("=" * 55 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the sentiment microservice"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="'api' calls the running HTTP server; 'local' loads the model directly",
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL (api mode)")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "api":
        asyncio.run(run_api_benchmark(args.url, args.samples))
    else:
        run_local_benchmark(args.samples)


if __name__ == "__main__":
    main()
