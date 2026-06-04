"""FAISS vector store integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    """Persisted FAISS index plus JSON metadata."""

    def __init__(self, index_dir: Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install faiss-cpu to use FAISS.") from exc

        self.faiss = faiss
        self.index_dir = Path(index_dir)
        self.index = None
        self.metadata: list[dict] = []

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    def build(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """Build an inner-product index for normalized vectors."""
        if len(chunks) == 0:
            raise ValueError("Cannot build an index with no chunks")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings must be a 2D array aligned with chunks")

        dimension = embeddings.shape[1]
        self.index = self.faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = chunks

    def save(self) -> None:
        """Save the FAISS index and metadata."""
        if self.index is None:
            raise RuntimeError("No index has been built")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS index files in {self.index_dir}. Run `index` first."
            )
        self.index = self.faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Search for nearest chunks."""
        if self.index is None:
            raise RuntimeError("Index is not loaded")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
