"""
Run once locally to build painting_emotion_index.faiss from paintings_metadata.json.
  python build_index.py
"""
import json, numpy as np, faiss
from pathlib import Path
from transformers import AutoModelForImageClassification

MODEL_ID = "SlashHash/Painting_Emotion_Classifier"
model    = AutoModelForImageClassification.from_pretrained(MODEL_ID)

EMOTION_LABELS = list(model.config.id2label.values())
EMBED_DIM      = len(EMOTION_LABELS)

metadata = json.loads(Path("paintings_metadata.json").read_text())

# Build one one-hot vector per painting based on its dominant emotion
vectors = np.zeros((len(metadata), EMBED_DIM), dtype=np.float32)
for i, painting in enumerate(metadata):
    emotion = painting["emotion_dominant"]
    if emotion in EMOTION_LABELS:
        vectors[i, EMOTION_LABELS.index(emotion)] = 1.0

# Flat L2 index (exact search, fine for small datasets)
index = faiss.IndexFlatL2(EMBED_DIM)
index.add(vectors)
faiss.write_index(index, "painting_emotion_index.faiss")
print(f"Built index with {index.ntotal} paintings, dim={EMBED_DIM}")