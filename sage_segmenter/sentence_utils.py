"""Sentence and paragraph utilities for SAGE-style semantic chunking."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


DEFAULT_DOCUMENT_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}

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


def _read_pdf_page_texts(path: Path) -> list[dict]:
    """Read PDF page text directly while preserving extraction line breaks."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pypdf to read PDF files.") from exc

    reader = PdfReader(str(path))
    records = []
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "doc"
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        records.append(
            {
                "id": f"{stem}_p{page_index}",
                "source": str(path),
                "page": page_index,
                "text": text,
            }
        )
    return records


def _read_text_page_record(path: Path) -> dict | None:
    """Read one plain-text document as a single page-like record."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "doc"
    return {
        "id": f"{stem}_p0",
        "source": str(path),
        "page": None,
        "text": text,
    }


def load_page_records_from_docs(
    docs_dir: Path,
    *,
    extensions: set[str] | None = None,
    source_contains: str = "",
    limit: int = 0,
) -> list[dict]:
    """Load page-like records from original documents without flattening paragraphs.

    This direct document path is for SAGE semantic chunk preparation. It avoids
    reconstructing text from fixed-size metadata chunks, because that path has
    already normalized whitespace and can erase paragraph boundaries.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {docs_dir}")

    supported = extensions or DEFAULT_DOCUMENT_EXTENSIONS
    needle = source_contains.lower()
    records: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if needle and needle not in str(path).lower():
            continue
        if path.suffix.lower() == ".pdf":
            records.extend(_read_pdf_page_texts(path))
        else:
            record = _read_text_page_record(path)
            if record is not None:
                records.append(record)
        if limit > 0 and len(records) >= limit:
            return records[:limit]
    return records
