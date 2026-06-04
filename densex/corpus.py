"""Corpus builders for DenseX-style granularity experiments.

The project already indexes PDF chunks. DenseX-style experiments need parallel
corpora with different retrieval units: chunk, sentence, and proposition. This
module keeps those units in one JSONL schema so the same FAISS builder and
experiment runner can work across all granularities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into dictionaries."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write dictionaries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """Append one dictionary to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_chunks_from_metadata(path: Path) -> list[dict]:
    """Load existing chunk metadata from a FAISS metadata JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list metadata in {path}")
    return [chunk for chunk in data if chunk.get("text")]


def paper_title_from_source(source: str) -> str:
    """Convert a PDF source path into a readable title-like string."""
    stem = Path(source).stem
    return re.sub(r"[_-]+", " ", stem).strip() or "Unknown"


def make_chunk_units(chunks: list[dict]) -> list[dict]:
    """Convert existing chunks into the shared DenseX unit schema."""
    units = []
    for chunk in chunks:
        source = chunk.get("source", "")
        units.append(
            {
                "id": f"chunk::{chunk.get('id')}",
                "granularity": "chunk",
                "text": chunk.get("text", ""),
                "source": source,
                "page": chunk.get("page"),
                "parent_chunk_id": chunk.get("id"),
                "paper_title": paper_title_from_source(source),
            }
        )
    return units


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？.!?])\s+|(?<=[。！？.!?])(?=[A-Z0-9\u4e00-\u9fff])"
)


def split_sentences(text: str, *, min_chars: int = 30) -> list[str]:
    """Split text into reasonably sized sentence units.

    PDF extraction can fragment equations and tables. Very short fragments are
    merged into their neighbors so sentence retrieval does not fill the index
    with punctuation-only or citation-only units.
    """
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    if not parts:
        return []

    merged: list[str] = []
    buffer = ""
    for part in parts:
        if buffer:
            buffer = f"{buffer} {part}"
        else:
            buffer = part
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def make_sentence_units(chunks: list[dict], *, min_chars: int = 30) -> list[dict]:
    """Create sentence units from chunk text."""
    units = []
    for chunk in chunks:
        source = chunk.get("source", "")
        parent_id = chunk.get("id")
        for idx, sentence in enumerate(split_sentences(chunk.get("text", ""), min_chars=min_chars)):
            units.append(
                {
                    "id": f"sentence::{parent_id}::s{idx}",
                    "granularity": "sentence",
                    "text": sentence,
                    "source": source,
                    "page": chunk.get("page"),
                    "parent_chunk_id": parent_id,
                    "sentence_index": idx,
                    "paper_title": paper_title_from_source(source),
                }
            )
    return units


def normalize_proposition_output(raw: str) -> list[str]:
    """Parse propositionizer output into a clean list of proposition strings."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed = None

    values: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                value = item.get("proposition") or item.get("text") or item.get("content")
                if value:
                    values.append(str(value))
    elif isinstance(parsed, dict):
        candidates = parsed.get("propositions") or parsed.get("facts") or parsed.get("items")
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    value = item.get("proposition") or item.get("text") or item.get("content")
                    if value:
                        values.append(str(value))

    if not values:
        values = [
            re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

    cleaned = []
    seen = set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n-")
        if len(value) < 8 or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned
