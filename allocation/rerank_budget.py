"""Rerank-first context allocation strategy."""

from __future__ import annotations

from allocation.baseline import allocate_baseline
from scoring.rerank import rerank_chunks


def allocate_rerank(query: str, chunks: list[dict], budget: int) -> list[dict]:
    """Rerank retrieved chunks, then fill the context budget in reranked order."""
    reranked = rerank_chunks(query, chunks)
    return allocate_baseline(reranked, budget=budget)
