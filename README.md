# Real-Time Sentiment Microservice

A production-grade sentiment analysis API built with **FastAPI + fine-tuned BERT**, containerised with **Docker**, and backed by **PostgreSQL**.

```
POST /api/v1/predict   →   {"label": "POSITIVE", "score": 0.97, "latency_ms": 48}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose Stack                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Service (:8000)               │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │  API Layer  │  │ Model Layer  │  │  DB Layer      │  │   │
│  │  │  routes.py  │  │ sentiment.py │  │  crud.py       │  │   │
│  │  │  schemas.py │  │ fine_tune.py │  │  models.py     │  │   │
│  │  │  (Pydantic) │  │ (BERT/HF)   │  │  (SQLAlchemy)  │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │   │
│  │         │                │                   │           │   │
│  │  ┌──────▼────────────────▼───────────────────▼────────┐  │   │
│  │  │              Benchmarking Layer                     │  │   │
│  │  │              evaluator.py                           │  │   │
│  │  └─────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  PostgreSQL 16 (:5432)                    │   │
│  │     predictions table  │  benchmark_runs table           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Resume Bullets — What Each Covers

| Bullet | Files |
|--------|-------|
| Production FastAPI service, fine-tuned BERT, 90% accuracy, <200ms latency | `app/main.py`, `app/model/sentiment.py`, `app/model/fine_tune.py` |
| Containerised with Docker, structured DB integration, modular separation | `Dockerfile`, `docker-compose.yml`, `app/db/`, `app/model/`, `app/api/` |
| Evaluation and benchmarking framework, systematic performance monitoring | `app/benchmarking/evaluator.py`, `scripts/run_benchmark.py` |

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo>
cd sentiment-microservice
cp .env.example .env
```

### 2. Start the full stack

```bash
docker compose up --build
```

On first run, Docker will:
1. Pull PostgreSQL 16
2. Build the API image
3. Download `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace (~250 MB, cached in a volume)
4. Create the `predictions` and `benchmark_runs` tables
5. Warm up the inference pipeline

The service is ready when you see:
```
sentiment_api | Service is ready to accept requests ✅
```

### 3. Call the API

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved the new product launch!"}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing experience!", "Terrible service.", "It was okay."]}'

# Health check
curl http://localhost:8000/api/v1/health

# Run a 100-sample benchmark
curl -X POST "http://localhost:8000/api/v1/benchmark?sample_size=100"
```

### 4. Interactive docs

Open **http://localhost:8000/docs** for the Swagger UI.

---

## Fine-tuning on Your Own Data

```bash
# Inside the container
docker compose exec api python -m app.model.fine_tune \
  --base_model distilbert-base-uncased \
  --dataset sst2 \
  --output_dir /app/fine_tuned_model \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5

# Then point the service to the new checkpoint
# In .env:
FINE_TUNED_MODEL_PATH=/app/fine_tuned_model
```

---

## Running Tests

```bash
# Install deps
pip install -r requirements.txt

# Run all tests with coverage
pytest

# Run without coverage (faster)
pytest --no-cov
```

---

## Running the Benchmark CLI

```bash
# Against the live API (recommended)
python scripts/run_benchmark.py --mode api --url http://localhost:8000 --samples 500

# Directly against the model (no Docker needed)
python scripts/run_benchmark.py --mode local --samples 200
```

Sample output:
```
============================================================
  BENCHMARK REPORT — distilbert-base-uncased-finetuned-sst-2-english
============================================================
  Samples        : 500
  Throughput     : 38.4 req/s
  SLA (<200ms p95) : ✅ PASS

  LATENCY (ms)
    Mean   : 48.21
    P50    : 45.10
    P95    : 89.34
    P99    : 143.20
    Max    : 198.40
    % < target : 99.4%

  ACCURACY
    Overall : 0.9200 (92.00%)
    F1 (POS): 0.9310
    F1 (NEG): 0.9080

  LABEL DISTRIBUTION
    POSITIVE    :   321 (64.2%)
    NEGATIVE    :   165 (33.0%)
    UNCERTAIN   :    14 (2.8%)

  LATENCY BY INPUT LENGTH
    short   : 38.21 ms
    medium  : 51.44 ms
    long    : 84.17 ms
============================================================
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Single-text sentiment analysis |
| `POST` | `/api/v1/predict/batch` | Batch analysis (up to 64 texts) |
| `GET` | `/api/v1/predictions` | Paginated prediction history |
| `GET` | `/api/v1/predictions/{id}` | Single prediction by ID |
| `POST` | `/api/v1/benchmark` | Run live benchmark |
| `GET` | `/api/v1/benchmark/runs` | List historical benchmark runs |
| `GET` | `/api/v1/health` | Liveness + readiness check |
| `GET` | `/docs` | Swagger UI |

---

## Project Structure

```
sentiment-microservice/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── config.py                # Centralised settings (pydantic-settings)
│   ├── model/
│   │   ├── sentiment.py         # BERT inference singleton
│   │   └── fine_tune.py         # Fine-tuning script
│   ├── api/
│   │   ├── routes.py            # All HTTP endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   ├── db/
│   │   ├── database.py          # Async SQLAlchemy engine + session
│   │   ├── models.py            # ORM table definitions
│   │   └── crud.py              # DB read/write operations
│   └── benchmarking/
│       └── evaluator.py         # Systematic evaluation framework
├── scripts/
│   └── run_benchmark.py         # Standalone benchmark CLI
├── tests/
│   └── test_sentiment.py        # Unit + integration tests
├── Dockerfile                   # Multi-stage build
├── docker-compose.yml           # Full stack: API + PostgreSQL
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI 0.111 |
| ML model | DistilBERT (HuggingFace Transformers) |
| ML framework | PyTorch |
| Database | PostgreSQL 16 + SQLAlchemy 2 (async) |
| Containerisation | Docker + Docker Compose |
| Validation | Pydantic v2 |
| Testing | pytest + pytest-asyncio |
| Logging | loguru |
