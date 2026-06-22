"""Sentence and paragraph utilities for SAGE-style semantic chunking."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？.!?])\s+|(?<=[。！？.!?])(?=[A-Z0-9\u4e00-\u9fff])"
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    """Split extracted text into paragraphs while normalizing inner whitespace."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = []
    for part in re.split(r"\n\s*\n+", normalized):
        paragraph = _normalize_space(part)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def split_sentences(text: str, *, min_chars: int = 30) -> list[str]:
    """Split text into sentence-like units.

    The `min_chars` value is intentionally conservative: it drops only obvious
    punctuation fragments. Short but meaningful answers such as "Yes." remain
    available for segmentation because SAGE needs adjacent sentence boundaries.
    """
    normalized = _normalize_space(text)
    if not normalized:
        return []
    sentences = []
    for part in _SENTENCE_BOUNDARY.split(normalized):
        sentence = part.strip()
        if not sentence:
            continue
        if len(sentence) < min_chars and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", sentence):
            continue
        sentences.append(sentence)
    return sentences


def _suffix_prefix_overlap(left: str, right: str, *, min_overlap: int) -> int:
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_ordered_chunk_texts(chunks: list[dict], *, min_overlap: int = 40) -> str:
    """Merge fixed chunks from the same source/page and remove exact overlaps."""
    merged = ""
    for chunk in chunks:
        text = _normalize_space(str(chunk.get("text", "")))
        if not text:
            continue
        if not merged:
            merged = text
            continue
        overlap = _suffix_prefix_overlap(merged, text, min_overlap=min_overlap)
        if overlap:
            merged = f"{merged}{text[overlap:]}"
        else:
            merged = f"{merged} {text}"
    return _normalize_space(merged)


def reconstruct_page_texts_from_chunks(chunks: list[dict]) -> list[dict]:
    """Reconstruct page-level text from existing overlapping metadata chunks."""
    groups: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    for chunk in chunks:
        source = str(chunk.get("source", ""))
        groups[(source, chunk.get("page"))].append(chunk)

    pages = []
    for (source, page), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        ordered = sorted(group, key=lambda chunk: (chunk.get("start") is None, chunk.get("start") or 0))
        text = merge_ordered_chunk_texts(ordered)
        if not text:
            continue
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem).strip("_") or "doc"
        page_part = f"p{page}" if page is not None else "p0"
        pages.append(
            {
                "id": f"{stem}_{page_part}",
                "source": source,
                "page": page,
                "text": text,
            }
        )
    return pages

