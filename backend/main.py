from fastapi import FastAPI
from schemas import AnxietyRequest, AnxietyResponse

app = FastAPI()


@app.post("/predict", response_model=AnxietyResponse)
def predict(data: AnxietyRequest):

    text = data.text.lower()

    if "nervous" in text or "panic" in text:
        level = 2
        category = "High Anxiety"

    elif "stress" in text or "worried" in text:
        level = 1
        category = "Moderate Anxiety"

    else:
        level = 0
        category = "Low Anxiety"

    return {
        "predicted_level": level,
        "anxiety_category": category
    }