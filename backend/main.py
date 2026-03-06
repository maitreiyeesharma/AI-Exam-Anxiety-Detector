from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Exam Anxiety Detector Backend Running"}


@app.post("/predict")
def predict(data: TextInput):

    text = data.text.lower()

    if "nervous" in text or "panic" in text or "scared" in text:
        anxiety = "High Anxiety"

    elif "worried" in text or "stress" in text:
        anxiety = "Moderate Anxiety"

    else:
        anxiety = "Low Anxiety"

    return {
        "input_text": text,
        "predicted_anxiety_level": anxiety
    }