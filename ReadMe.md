# Real-time Sentiment Analysis Microservice

A production-ready sentiment analysis microservice that exposes a REST API for real‑time text classification. Uses a fine‑tuned BERT model (DistilBERT) to predict whether a given text expresses **positive**, **negative**, or **neutral** sentiment, along with a confidence score. Built with **FastAPI**, **PyTorch**, and **Transformers**, and containerised with **Docker**.

This project is designed as a portfolio piece for master's students in AI / Data Science, demonstrating:

- End‑to‑end ML system design  
- Modular Python development  
- REST API best practices  
- Version control with Git  
- Containerisation and deployment readiness  

---

## Features

- **Real‑time inference** – sub‑second response times.
- **Pre‑trained transformer model** – DistilBERT fine‑tuned on SST‑2 (Stanford Sentiment Treebank).
- **Clean REST API** – auto‑generated Swagger documentation at `/docs`.
- **Modular architecture** – separate modules for config, preprocessing, model inference, and API.
- **Docker support** – easy deployment anywhere.
- **Unit tests** – basic tests for API and model.
- **Environment‑aware** – configurable via environment variables.

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** – web framework
- **Uvicorn** – ASGI server
- **PyTorch** – deep learning framework
- **Transformers** (Hugging Face) – pre‑trained models and tokenizers
- **Pydantic** – data validation
- **Docker** – containerisation
- **pytest** – testing

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) Docker

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishbakriwala8/Real-time-Sentiment-Analysis-Microservice/
   cd sentiment-service
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API server**
   ```bash
   uvicorn app.api:app --reload
   ```

   The server will start at `http://127.0.0.1:8000`.  
   Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## API Usage

### Endpoints

| Method | Endpoint     | Description                     |
|--------|--------------|---------------------------------|
| GET    | `/`          | Welcome message                 |
| GET    | `/health`    | Health check                    |
| POST   | `/v1/predict`| Predict sentiment of input text |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this project!"}'
```

### Example Response

```json
{
  "label": "POSITIVE",
  "confidence": 0.9998
}
```

### Input/Output Schema

**Request Body** (`PredictionRequest`):
```json
{
  "text": "string"
}
```

**Response Body** (`PredictionResponse`):
```json
{
  "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence": 0.0 .. 1.0
}
```

> **Note:** The model used (`distilbert-base-uncased-finetuned-sst-2-english`) only outputs `POSITIVE`/`NEGATIVE`. If you need neutral sentiment, you can replace the model with a three‑class variant (e.g., `cardiffnlp/twitter-roberta-base-sentiment`).

---

## Running with Docker

1. **Build the image**
   ```bash
   docker build -t sentiment-service .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 sentiment-service
   ```

The API will be available at `http://localhost:8000`.

---

## Project Structure

```
sentiment-service/
├── app/
│   ├── __init__.py
│   ├── api.py               # FastAPI application
│   ├── config.py             # Configuration (model name, device, etc.)
│   ├── model.py              # Model loading and inference
│   ├── preprocess.py         # Text preprocessing (tokenization)
│   └── schemas.py            # Pydantic request/response models
├── tests/
│   ├── test_api.py           # API unit tests
│   └── test_model.py         # Model unit tests
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
└── API_DOCS.md               # (optional) detailed API documentation
```

---

## Configuration

You can customise the service using environment variables:

| Variable     | Default                                               | Description                          |
|--------------|-------------------------------------------------------|--------------------------------------|
| `MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english`    | Hugging Face model identifier        |
| `MAX_LENGTH` | `512`                                                 | Maximum token length                 |
| `DEVICE`     | `cpu`                                                 | `cpu` or `cuda` (if GPU available)   |

Example (Windows PowerShell):
```powershell
$env:MODEL_NAME="cardiffnlp/twitter-roberta-base-sentiment"
uvicorn app.api:app --reload
```

---

## Testing

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## Future Improvements

- Add support for batch prediction.
- Implement caching for frequent queries.
- Deploy to a cloud platform (AWS, GCP, Azure) with a load balancer.
- Add more sophisticated preprocessing (emoji handling, URL removal, etc.).
- Integrate a monitoring dashboard (e.g., Prometheus + Grafana).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Hugging Face for the [Transformers](https://github.com/huggingface/transformers) library and pre‑trained models.
- FastAPI for the amazing web framework.
- The open‑source community for the tools that made this project possible.

---

**Krish Bakriwala**  
*Master's student in Artificial Intelligence*
