"""Summarization-based compression utilities."""

from __future__ import annotations


def compress_text(text: str, max_chars: int) -> str:
    """Truncate text as a deterministic placeholder for summarization."""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compress_chunks(chunks: list[dict], max_chars_per_chunk: int) -> list[dict]:
    """Compress each chunk independently."""
    compressed = []
    for chunk in chunks:
        item = dict(chunk)
        item["text"] = compress_text(item.get("text", ""), max_chars_per_chunk)
        compressed.append(item)
    return compressed
