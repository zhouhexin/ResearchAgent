"""Relevance scoring utilities."""

from __future__ import annotations


def relevance_score(similarity: float) -> float:
    """Convert vector similarity into a bounded relevance score."""
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def apply_relevance_scores(chunks: list[dict]) -> list[dict]:
    """Attach relevance scores to retrieved chunks."""
    scored = []
    for chunk in chunks:
        item = dict(chunk)
        item["relevance"] = relevance_score(float(item.get("score", 0.0)))
        scored.append(item)
    return scored
