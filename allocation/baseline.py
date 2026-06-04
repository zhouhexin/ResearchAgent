"""Baseline context allocation strategy."""

from __future__ import annotations

from evaluation.token_counter import count_tokens


def estimate_tokens(text: str) -> int:
    """Count tokens with the shared experiment token counter."""
    return count_tokens(text)


def allocate_baseline(chunks: list[dict], budget: int) -> list[dict]:
    """Select chunks in retrieval order until the context budget is exhausted."""
    selected: list[dict] = []
    used = 0
    for chunk in chunks:
        token_count = estimate_tokens(chunk.get("text", ""))
        if used + token_count > budget:
            continue
        item = dict(chunk)
        item["estimated_tokens"] = token_count
        selected.append(item)
        used += token_count
    return selected
