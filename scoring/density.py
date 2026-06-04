"""Information density scoring utilities."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"\w+", re.UNICODE)


def density_score(text: str) -> float:
    """Estimate information density with a simple lexical heuristic."""
    words = WORD_RE.findall(text)
    if not words:
        return 0.0

    unique_ratio = len(set(word.lower() for word in words)) / len(words)
    digit_bonus = min(0.2, len(re.findall(r"\d", text)) / 100.0)
    punctuation_bonus = min(0.1, sum(text.count(mark) for mark in ".:;") / 50.0)
    return max(0.0, min(1.0, unique_ratio + digit_bonus + punctuation_bonus))


def apply_density_scores(chunks: list[dict]) -> list[dict]:
    """Attach density scores to chunks."""
    scored = []
    for chunk in chunks:
        item = dict(chunk)
        item["density"] = density_score(item.get("text", ""))
        scored.append(item)
    return scored
