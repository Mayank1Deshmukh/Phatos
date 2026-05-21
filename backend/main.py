from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

app = FastAPI()

MODEL_ID = "SlashHash/Painting_Emotion_Classifier"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

class PredictRequest(BaseModel):
    image_url: str

@app.post("/api/predict")          # <-- /api/predict, not /predict
def predict(req: PredictRequest):
    image = Image.open(BytesIO(requests.get(req.image_url).content)).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], dim=-1)
    scores = [
        {"emotion": model.config.id2label[i], "confidence": round(p.item(), 4)}
        for i, p in enumerate(probs)
    ]
    scores.sort(key=lambda x: x["confidence"], reverse=True)
    return {"predictions": scores}