from fastapi import FastAPI
from backend.schemas import AnxietyRequest, AnxietyResponse
from backend.model_loader import model, tokenizer, device
import torch

app = FastAPI()

# Anxiety labels
labels = {
    0: "Low Anxiety",
    1: "Moderate Anxiety",
    2: "High Anxiety"
}

@app.get("/")
def home():
    return {"message": "AI Exam Anxiety Detector API Running"}

@app.post("/predict", response_model=AnxietyResponse)
def predict(data: AnxietyRequest):

    text = data.text

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return {
        "anxiety_level": prediction,
        "anxiety_category": labels[prediction]
    }