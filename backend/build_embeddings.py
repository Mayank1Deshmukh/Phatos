import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "SlashHash/Painting_Emotion_Classifier"

processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "paintings_metadata.json"

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
vectors: list[np.ndarray] = []

for painting in metadata:
    print(f"Processing: {painting['title']}")
    try:
        img_path = Path(painting["thumbnail_url"])
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy().astype(np.float32)
        vectors.append(probs)

    except Exception as e:
        print(f"  Failed ({e}), using zero vector.")
        vectors.append(np.zeros(model.config.num_classes, dtype=np.float32))

embeddings = np.stack(vectors).astype(np.float32)
np.save(BASE_DIR / "painting_embeddings.npy", embeddings)
print(f"Saved painting_embeddings.npy shape={embeddings.shape}")