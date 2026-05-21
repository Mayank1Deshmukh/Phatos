"""
Run once locally to produce umap_coords.npy from painting_embeddings.npy.
  pip install umap-learn
  python build_umap.py
"""
import numpy as np
from umap import UMAP

# painting_embeddings.npy: shape (N, D)
# Each row is the softmax probability vector from /predict for one painting.
embeddings = np.load("painting_embeddings.npy")

reducer = UMAP(n_components=2, random_state=42)
coords  = reducer.fit_transform(embeddings)   # shape (N, 2)

np.save("umap_coords.npy", coords)
print(f"Saved umap_coords.npy  shape={coords.shape}")