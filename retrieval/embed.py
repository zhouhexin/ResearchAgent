"""Embedding utilities for retrieval."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Small wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install sentence-transformers to use embeddings."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts as normalized float32 vectors."""
        if not texts:
            return np.empty((0, 0), dtype="float32")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
