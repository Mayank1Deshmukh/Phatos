"""
Run once locally to produce umap_coords.npy from painting_embeddings.npy.
pip install umap-learn
python build_umap.py
"""
import numpy as np
from umap import UMAP

embeddings: np.ndarray = np.asarray(np.load("painting_embeddings.npy"), dtype=np.float32)

reducer = UMAP(n_components=2, random_state=42)
coords: np.ndarray = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)

np.save("umap_coords.npy", coords)
print(f"Saved umap_coords.npy shape={coords.shape}")