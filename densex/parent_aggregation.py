"""Aggregate fine-grained retrieval hits back to parent chunks."""

from __future__ import annotations

import json
from pathlib import Path

from densex.corpus import deduplicate_units_by_text


def load_parent_chunks(metadata_path: Path) -> dict[str, dict]:
    """Load original chunk metadata keyed by chunk id."""
    chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list metadata in {metadata_path}")
    return {
        str(chunk.get("id")): chunk
        for chunk in chunks
        if chunk.get("id") is not None and chunk.get("text")
    }


def aggregate_fine_hits_to_parent_chunks(
    fine_hits: list[dict],
    parent_chunks: dict[str, dict],
    *,
    parent_top_k: int,
    top_child_count: int = 3,
    child_sum_weight: float = 0.1,
    fine_hit_dedup: str = "none",
) -> list[dict]:
    """Convert sentence/proposition hits into scored parent chunks.

    The v1 policy keeps one candidate per parent chunk. Relevance is estimated
    from fine-grained hits using `max(child_score) + weight * sum(top-n scores)`.
    This rewards a strong local match while giving a small boost when multiple
    fine-grained units from the same chunk match the query.
    """
    grouped: dict[str, list[dict]] = {}
    for hit in fine_hits:
        parent_id = hit.get("parent_chunk_id")
        if parent_id is None:
            continue
        parent_id = str(parent_id)
        if parent_id not in parent_chunks:
            continue
        grouped.setdefault(parent_id, []).append(hit)

    candidates: list[dict] = []
    if fine_hit_dedup not in {"none", "exact-per-parent"}:
        raise ValueError(f"Unsupported fine hit dedup mode: {fine_hit_dedup}")

    for parent_id, hits in grouped.items():
        ranked_hits = sorted(hits, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        if fine_hit_dedup == "exact-per-parent":
            ranked_hits = deduplicate_units_by_text(ranked_hits)
        top_scores = [float(hit.get("score") or 0.0) for hit in ranked_hits[:top_child_count]]
        max_score = top_scores[0] if top_scores else 0.0
        aggregate_score = max_score + child_sum_weight * sum(top_scores)

        parent = dict(parent_chunks[parent_id])
        parent["score"] = aggregate_score
        parent["fine_to_chunk"] = {
            "parent_chunk_id": parent_id,
            "matched_child_count": len(hits),
            "deduplicated_child_count": len(ranked_hits),
            "fine_hit_dedup": fine_hit_dedup,
            "max_child_score": max_score,
            "top_child_score_sum": sum(top_scores),
            "top_child_ids": [hit.get("id") for hit in ranked_hits[:top_child_count]],
            "top_child_texts": [hit.get("text", "") for hit in ranked_hits[:top_child_count]],
        }
        candidates.append(parent)

    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return candidates[:parent_top_k]
