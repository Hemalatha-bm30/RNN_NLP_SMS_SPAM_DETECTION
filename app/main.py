from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import SMSRNN
from preprocess import clean_text, encode, pad_sequence

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sms_rnn.pt"

app = FastAPI(
    title=" SMS Spam Detection API",
    description="NLP + Regex + PyTorch RNN demonstration API",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="SMS message to analyze")

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    cleaned_text: str

model = None
checkpoint = None

def load_model():
    global model, checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = SMSRNN(len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
def root():
    return {
        "message": "SMS Spam Detection API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/clean")
def clean(request: TextRequest):
    return {
        "original_text": request.text,
        "cleaned_text": clean_text(request.text)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    cleaned = clean_text(request.text)
    ids = encode(cleaned, checkpoint["vocab"])
    x = torch.tensor(
        [pad_sequence(ids, checkpoint["max_len"])],
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_id = int(probabilities.argmax())
        confidence = float(probabilities[class_id])

    label = "spam" if class_id == 1 else "ham"
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "cleaned_text": cleaned
    }