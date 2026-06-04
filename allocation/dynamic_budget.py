"""Dynamic context budget allocation strategy."""

from __future__ import annotations

from allocation.baseline import estimate_tokens
from scoring.density import density_score
from scoring.redundancy import redundancy_penalty
from scoring.relevance import relevance_score


def allocate_dynamic(chunks: list[dict], budget: int) -> list[dict]:
    """Select chunks by relevance, density, and redundancy."""
    candidates: list[dict] = []
    for chunk in chunks:
        item = dict(chunk)
        item["relevance"] = item.get("relevance", relevance_score(float(item.get("score", 0.0))))
        item["density"] = item.get("density", density_score(item.get("text", "")))
        item["estimated_tokens"] = estimate_tokens(item.get("text", ""))
        candidates.append(item)

    selected: list[dict] = []
    used = 0
    while candidates:
        best_index = -1
        best_score = float("-inf")
        for idx, item in enumerate(candidates):
            penalty = redundancy_penalty(item, selected)
            final_score = 0.6 * item["relevance"] + 0.3 * item["density"] - 0.1 * penalty
            if final_score > best_score:
                best_score = final_score
                best_index = idx

        best = candidates.pop(best_index)
        if used + best["estimated_tokens"] <= budget:
            best["final_score"] = best_score
            selected.append(best)
            used += best["estimated_tokens"]

    return selected
