from fastapi import FastAPI
import torch

from schemas import AnxietyRequest, AnxietyResponse
from model_loader import model, tokenizer, device

app = FastAPI(title="AI Exam Anxiety Detector API")


@app.post("/predict", response_model=AnxietyResponse)
def predict(data: AnxietyRequest):

    text = data.text

    # Tokenize the input text using BERT tokenizer
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Convert prediction to human-readable category
    labels = {
        0: "Low Anxiety",
        1: "Moderate Anxiety",
        2: "High Anxiety"
    }

    return {
        "predicted_level": prediction,
        "anxiety_category": labels[prediction]
    }