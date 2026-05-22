import json
from pathlib import Path

import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
metadata = json.loads((BASE_DIR / "paintings_metadata.json").read_text(encoding="utf-8"))
model_config = json.loads((BASE_DIR / "model_config.json").read_text(encoding="utf-8"))

emotion_labels = model_config["emotions"]
embed_dim = len(emotion_labels)

vectors = np.zeros((len(metadata), embed_dim), dtype=np.float32)
for i, painting in enumerate(metadata):
    emotion = painting["emotion_dominant"]
    if emotion in emotion_labels:
        vectors[i, emotion_labels.index(emotion)] = 1.0

index = faiss.IndexFlatL2(embed_dim)
index.add(vectors)
faiss.write_index(index, str(BASE_DIR / "painting_emotion_index.faiss"))
print(f"Built index with {index.ntotal} paintings, dim={embed_dim}")