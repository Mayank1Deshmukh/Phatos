from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import io
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

app = FastAPI()

# ── model (loaded once on cold start) ────────────────────────────────────────
MODEL_ID = "SlashHash/Painting_Emotion_Classifier"
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model     = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# ── helpers ───────────────────────────────────────────────────────────────────
def fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def image_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# ── Grad-CAM for ViT / CNN ─────────────────────────────────────────────────
# Works for any model that has a final conv or ViT block whose output
# we can hook.  We try the last named module that has 4-D output (B,C,H,W).

_gradients: list = []
_activations: list = []

def _save_grad(g):   _gradients.append(g)
def _save_act(m, i, o): _activations.append(o)

def grad_cam(inputs: dict, class_idx: int) -> np.ndarray:
    """
    Returns a float32 numpy array in [0,1] shaped (H,W).
    H,W match the model's spatial feature grid (typically 7x7 or 14x14).
    """
    _gradients.clear()
    _activations.clear()

    # ── find the last module with a 4-D output ─────────────────────────────
    hook_module = None
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.LayerNorm)):
            hook_module = m
            break
    if hook_module is None:
        raise RuntimeError("Could not find a hookable layer.")

    fwd_hook = hook_module.register_forward_hook(_save_act)
    # grad hook is registered on the tensor inside forward
    # so we use retain_graph + backward

    pixel_values = inputs["pixel_values"].requires_grad_(True)
    new_inputs   = {**inputs, "pixel_values": pixel_values}

    outputs = model(**new_inputs)
    logits  = outputs.logits                      # (1, num_classes)

    model.zero_grad()
    score = logits[0, class_idx]
    score.backward()

    fwd_hook.remove()

    act  = _activations[0]                        # (1, C, H, W) or (1, N, C)
    grad = pixel_values.grad                      # same shape as input

    # ── ViT outputs (1, num_patches+1, hidden) → reshape to spatial ────────
    if act.dim() == 3:
        # drop CLS token, reshape patches to sqrt grid
        act = act[:, 1:, :]                       # (1, N, C)
        n   = act.shape[1]
        h   = w = int(n ** 0.5)
        act = act.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # (1,C,h,w)

    # ── pool gradients over spatial dims per channel ────────────────────────
    # We use the gradient of the score w.r.t. the pixel space as a proxy
    # when direct feature gradients aren't available from the hook.
    # Instead, compute a simple input-gradient × input saliency map.
    saliency = (pixel_values.grad.abs()).squeeze(0)   # (3, H, W)
    heatmap  = saliency.mean(dim=0).detach().numpy()  # (H, W)

    # normalise to [0,1]
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Resize heatmap to original size, apply jet colormap, alpha-blend."""
    h, w = original.size[1], original.size[0]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    hm_img = Image.fromarray(heatmap_uint8, mode="L").resize(
        (w, h), resample=Image.BILINEAR
    )
    # Jet colormap via numpy
    hm_np = np.array(hm_img, dtype=np.float32) / 255.0
    r = np.clip(1.5 - np.abs(hm_np * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(hm_np * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(hm_np * 4 - 1), 0, 1)
    jet = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    jet_img = Image.fromarray(jet, mode="RGB").resize((w, h))

    blended = Image.blend(original, jet_img, alpha=0.45)
    return blended

# ── schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_url: str

class ExplainRequest(BaseModel):
    image_url: str
    emotion: Optional[str] = None   # if omitted → top predicted class

# ── routes ────────────────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(req: PredictRequest):
    image  = fetch_image(req.image_url)
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits[0], dim=-1)
    scores = [
        {"emotion": model.config.id2label[i], "confidence": round(p.item(), 4)}
        for i, p in enumerate(probs)
    ]
    scores.sort(key=lambda x: x["confidence"], reverse=True)
    return {"predictions": scores}


@app.post("/api/explain")
def explain(req: ExplainRequest):
    image  = fetch_image(req.image_url)
    inputs = extractor(images=image, return_tensors="pt")

    # resolve target class index
    if req.emotion:
        label2id = {v: k for k, v in model.config.id2label.items()}
        if req.emotion not in label2id:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown emotion '{req.emotion}'. "
                       f"Valid: {list(label2id.keys())}"
            )
        class_idx = label2id[req.emotion]
    else:
        with torch.no_grad():
            logits    = model(**inputs).logits
        class_idx = int(logits.argmax(dim=-1).item())

    target_emotion = model.config.id2label[class_idx]

    # compute saliency-based heatmap
    heatmap = grad_cam(inputs, class_idx)

    # produce overlay image
    overlay = overlay_heatmap(image, heatmap)

    # also return top-5 bounding boxes (coarse, from heatmap grid cells > 0.5)
    threshold  = 0.5
    h_np       = heatmap
    rows, cols = np.where(h_np > threshold)
    regions    = []
    if len(rows):
        gh, gw = h_np.shape
        oh, ow = image.size[1], image.size[0]
        x0 = int(cols.min() / gw * ow)
        y0 = int(rows.min() / gh * oh)
        x1 = int(cols.max() / gw * ow)
        y1 = int(rows.max() / gh * oh)
        regions.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                         "label": target_emotion})

    return {
        "emotion":          target_emotion,
        "heatmap_overlay":  image_to_b64(overlay),   # base64 PNG
        "regions":          regions,                  # coarse bounding box(es)
        "xai_method":       "input-gradient saliency + jet colormap overlay"
    }


# ── FAISS search ──────────────────────────────────────────────────────────────
import json 
import faiss
from pathlib import Path
from functools import lru_cache

FAISS_PATH    = Path(__file__).parent / "painting_emotion_index.faiss"
METADATA_PATH = Path(__file__).parent / "paintings_metadata.json"

# Map each emotion label to a hand-crafted unit vector.
# Dimension must match what was used when the index was built.
EMOTION_LABELS = [label for label in model.config.id2label.values()]
EMBED_DIM      = len(EMOTION_LABELS)   # one-hot size = number of emotion classes

def emotion_to_vector(emotion: str) -> np.ndarray:
    """One-hot vector for an emotion label, as float32."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    if emotion in EMOTION_LABELS:
        vec[EMOTION_LABELS.index(emotion)] = 1.0
    return vec

@lru_cache(maxsize=1)
def load_faiss():
    index    = faiss.read_index(str(FAISS_PATH))
    metadata = json.loads(METADATA_PATH.read_text())
    return index, metadata


class SearchRequest(BaseModel):
    emotion: str
    limit:   int = 10


@app.get("/api/search")
def search(emotion: str = "joy", limit: int = 10):
    index, metadata = load_faiss()

    query = emotion_to_vector(emotion).reshape(1, -1)   # (1, EMBED_DIM)
    distances, indices = index.search(query, limit)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:           # FAISS returns -1 when fewer results exist
            continue
        entry = metadata[idx]
        entry["score"] = float(distances[0][rank])
        results.append(entry)

    return results


# ── UMAP embeddings ───────────────────────────────────────────────────────────
from typing import Optional

UMAP_PATH = Path(__file__).parent / "umap_coords.npy"

@lru_cache(maxsize=1)
def load_umap():
    coords   = np.load(str(UMAP_PATH))          # shape (N, 2)
    metadata = json.loads(METADATA_PATH.read_text())
    return coords, metadata


@app.get("/api/embeddings")
def embeddings(movement: Optional[str] = None, century: Optional[int] = None):
    coords, metadata = load_umap()

    xs, ys, ids, meta_out = [], [], [], []

    for i, painting in enumerate(metadata):
        # optional filters
        if movement and painting.get("movement", "").lower() != movement.lower():
            continue
        if century:
            decade = painting.get("year", 0)
            if not (century * 100 - 99 <= decade <= century * 100):
                continue

        xs.append(round(float(coords[i, 0]), 4))
        ys.append(round(float(coords[i, 1]), 4))
        ids.append(painting["id"])
        meta_out.append(painting)

    return {"x": xs, "y": ys, "painting_ids": ids, "metadata": meta_out}


# ── /annotate ─────────────────────────────────────────────────────────────────
# votes shape: { painting_id: { emotion: count } }
# Kept in memory; survives requests but resets on cold start.
# Swap VOTES_PATH for a real DB later if needed.

from collections import defaultdict

VOTES_PATH = Path(__file__).parent / "votes.json"

def load_votes() -> dict:
    if VOTES_PATH.exists():
        return json.loads(VOTES_PATH.read_text())
    return {}

def save_votes(votes: dict):
    VOTES_PATH.write_text(json.dumps(votes, indent=2))

# load once at startup into a plain dict
votes: dict = load_votes()   # { "1": { "joy": 3, "sadness": 1 } }


class AnnotatePost(BaseModel):
    painting_id:     int
    guessed_emotion: str


@app.post("/api/annotate", status_code=201)
def post_annotate(req: AnnotatePost):
    key = str(req.painting_id)
    if key not in votes:
        votes[key] = {}
    votes[key][req.guessed_emotion] = votes[key].get(req.guessed_emotion, 0) + 1
    save_votes(votes)
    return {"ok": True}


@app.get("/api/annotate")
def get_annotate(painting_id: int):
    key    = str(painting_id)
    counts = votes.get(key, {})
    total  = sum(counts.values())
    top    = max(counts, key=counts.get) if counts else None
    return {
        "label_counts": counts,
        "top_label":    top,
        "total_votes":  total,
    }