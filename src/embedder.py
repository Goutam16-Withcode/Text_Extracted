"""
embedder.py — Semantic Embedding Vector Generator & Visualizer.
Features:
  - Generates per-word and per-sentence embeddings using sentence-transformers
  - PCA / t-SNE dimensionality reduction for 2D visualization
  - UMAP reduction for cluster visualization (if umap-learn installed)
  - Cosine semantic similarity heatmap between sentences
  - Word cluster labeling
"""

import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple


@st.cache_resource(show_spinner="🔢 Loading embedding model (one-time)...")
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the sentence-transformer embedding model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.
    Returns ndarray of shape (N, embedding_dim).
    """
    if not texts:
        return np.array([])
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensions using PCA."""
    from sklearn.decomposition import PCA
    if len(embeddings) < n_components:
        return embeddings
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def reduce_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: int = None) -> np.ndarray:
    """Reduce embedding dimensions using t-SNE."""
    from sklearn.manifold import TSNE
    n = len(embeddings)
    if n < 4:
        return reduce_pca(embeddings, n_components)
    perp = min(perplexity or 30, n - 1)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=42,
        n_iter=500,
        learning_rate="auto",
        init="pca"
    )
    return tsne.fit_transform(embeddings)


def reduce_umap(embeddings: np.ndarray, n_components: int = 2) -> Optional[np.ndarray]:
    """Reduce embedding dimensions using UMAP (if available)."""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
    except ImportError:
        return None


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix (assumes normalized embeddings)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-8, None)
    return normalized @ normalized.T


def cluster_words(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """K-Means cluster the embeddings, return cluster labels."""
    from sklearn.cluster import KMeans
    n = len(embeddings)
    k = min(n_clusters, max(1, n // 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def build_embedding_data(
    ocr_results: List[Dict[str, Any]],
    text_unit: str = "word",  # "word" | "sentence" | "line"
    reduction: str = "PCA",    # "PCA" | "t-SNE" | "UMAP"
    n_clusters: int = 5,
) -> Dict[str, Any]:
    """
    Full embedding pipeline:
    1. Extract text units from OCR results
    2. Generate embeddings
    3. Reduce to 2D
    4. Cluster
    5. Return structured data for plotting

    Returns dict with keys:
        texts, embeddings, coords_2d, cluster_labels, similarity_matrix
    """
    if not ocr_results:
        return {}

    # ── Step 1: Extract text units ──
    if text_unit == "word":
        texts = [r["text"] for r in ocr_results if r["text"].strip()]
    elif text_unit == "sentence":
        full = " ".join(r["text"] for r in ocr_results)
        # Simple sentence split
        import re
        sentences = re.split(r"(?<=[.!?])\s+", full)
        texts = [s.strip() for s in sentences if len(s.strip()) > 3]
    else:  # line
        # Group words into lines by y-proximity
        sorted_results = sorted(ocr_results, key=lambda r: r["y_min"])
        lines = []
        current_line_y = None
        current_line_texts = []
        for r in sorted_results:
            if current_line_y is None or abs(r["y_min"] - current_line_y) < 20:
                current_line_texts.append(r["text"])
                current_line_y = r["y_min"]
            else:
                if current_line_texts:
                    lines.append(" ".join(current_line_texts))
                current_line_texts = [r["text"]]
                current_line_y = r["y_min"]
        if current_line_texts:
            lines.append(" ".join(current_line_texts))
        texts = [l for l in lines if l.strip()]

    if not texts:
        return {}

    # ── Step 2: Embed ──
    embeddings = embed_texts(texts)
    if embeddings.size == 0:
        return {}

    # ── Step 3: Reduce to 2D ──
    if reduction == "t-SNE" and len(texts) >= 4:
        coords_2d = reduce_tsne(embeddings)
    elif reduction == "UMAP":
        coords_2d = reduce_umap(embeddings)
        if coords_2d is None:
            coords_2d = reduce_pca(embeddings)
    else:
        coords_2d = reduce_pca(embeddings)

    # ── Step 4: Cluster ──
    if len(texts) >= 2:
        cluster_labels = cluster_words(embeddings, n_clusters)
    else:
        cluster_labels = np.zeros(len(texts), dtype=int)

    # ── Step 5: Similarity matrix ──
    if len(texts) <= 60:
        sim_matrix = cosine_similarity_matrix(embeddings)
    else:
        sim_matrix = None

    # ── Step 6: Top-N similar pairs ──
    similar_pairs = []
    if sim_matrix is not None and len(texts) > 1:
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similar_pairs.append({
                    "i": i,
                    "j": j,
                    "text_i": texts[i],
                    "text_j": texts[j],
                    "similarity": float(sim_matrix[i, j]),
                })
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        similar_pairs = similar_pairs[:10]

    return {
        "texts": texts,
        "embeddings": embeddings,
        "coords_2d": coords_2d,
        "cluster_labels": cluster_labels.tolist(),
        "similarity_matrix": sim_matrix,
        "similar_pairs": similar_pairs,
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        "n_texts": len(texts),
        "reduction_method": reduction,
    }
