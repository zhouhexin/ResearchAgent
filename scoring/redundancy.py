"""Redundancy scoring utilities."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _token_set(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text)}


def jaccard_similarity(left: str, right: str) -> float:
    """Compute lexical overlap between two strings."""
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def remove_redundant_chunks(chunks: list[dict], threshold: float = 0.85) -> list[dict]:
    """Drop chunks that are near-duplicates of earlier chunks."""
    selected: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if all(jaccard_similarity(text, item.get("text", "")) < threshold for item in selected):
            selected.append(chunk)
    return selected


def redundancy_penalty(chunk: dict, selected: list[dict]) -> float:
    """Return max lexical overlap with already selected chunks."""
    if not selected:
        return 0.0
    text = chunk.get("text", "")
    return max(jaccard_similarity(text, item.get("text", "")) for item in selected)
