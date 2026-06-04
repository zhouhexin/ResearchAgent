"""Document retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

from retrieval.embed import Embedder
from retrieval.faiss_store import FaissStore


class Retriever:
    """Query interface for the local FAISS store."""

    def __init__(self, index_dir: Path, embedding_model: str) -> None:
        self.embedder = Embedder(embedding_model)
        self.store = FaissStore(index_dir)
        self.store.load()

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return top-k chunks for a query."""
        query_embedding = self.embedder.encode([query])
        return self.store.search(query_embedding, top_k=top_k)
