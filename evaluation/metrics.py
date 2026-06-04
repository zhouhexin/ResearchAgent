"""Evaluation metric utilities."""

from __future__ import annotations

import re


def cited_chunk_indices(answer: str) -> list[int]:
    """Extract cited chunk indices like [1] from an answer."""
    return [int(match) for match in re.findall(r"\[(\d+)\]", answer)]


def citation_count(answer: str) -> int:
    """Count bracket citations like [1]."""
    return len(cited_chunk_indices(answer))


def context_utilization(answer: str, contexts: list[dict]) -> float:
    """Estimate how many selected contexts are cited in the answer."""
    if not contexts:
        return 0.0
    cited = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    valid = {idx for idx in range(1, len(contexts) + 1)}
    return len(cited & valid) / len(valid)


def citation_validity_ratio(answer: str, contexts: list[dict]) -> float:
    """Estimate how many cited chunk references are valid for this request."""
    cited = cited_chunk_indices(answer)
    if not cited:
        return 0.0
    valid = set(range(1, len(contexts) + 1))
    valid_count = sum(1 for idx in cited if idx in valid)
    return valid_count / len(cited)


def invalid_citation_indices(answer: str, contexts: list[dict]) -> list[int]:
    """Return cited indices that were not included in the request context."""
    valid = set(range(1, len(contexts) + 1))
    return [idx for idx in cited_chunk_indices(answer) if idx not in valid]
