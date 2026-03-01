import torch
from transformers import AutoModelForSequenceClassification
from app.config import Config
from app.preprocess import preprocess_text

# Load model globally (cached after first load)
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
model.eval()
# Move model to device (CPU/GPU)
device = torch.device(Config.DEVICE)
model.to(device)

def predict_sentiment(text: str):
    """Run inference on input text and return label and confidence."""
    inputs = preprocess_text(text)
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
    
    # Map class index to label (model-specific)
    # For distilbert sentiment: 0 = NEGATIVE, 1 = POSITIVE
    id2label = model.config.id2label  # e.g., {0: 'NEGATIVE', 1: 'POSITIVE'}
    label = id2label[predicted_class.item()]
    
    return {
        "label": label,
        "confidence": confidence.item()
    }