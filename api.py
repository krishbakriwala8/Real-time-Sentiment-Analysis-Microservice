from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="A microservice for real-time sentiment analysis using BERT.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running. Go to /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment of input text."""
    try:
        result = predict_sentiment(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))