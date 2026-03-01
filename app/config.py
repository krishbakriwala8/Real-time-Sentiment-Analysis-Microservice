import os

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
    DEVICE = os.getenv("DEVICE", "cpu")  # or "cuda" if GPU available