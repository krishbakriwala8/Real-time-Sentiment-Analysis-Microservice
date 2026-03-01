from transformers import AutoTokenizer
from app.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def preprocess_text(text: str):
    """Clean and tokenize input text for the model."""
    # Basic cleaning (remove extra spaces, etc.)
    text = " ".join(text.split())
    # Tokenize (returns input_ids, attention_mask)
    inputs = tokenizer(
        text,
        max_length=Config.MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return inputs