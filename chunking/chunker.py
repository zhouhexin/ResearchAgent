"""Document loading and chunking utilities."""

from __future__ import annotations

from pathlib import Path
import re


def _safe_id_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "doc"


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    source: str = "unknown",
) -> list[dict]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[dict] = []
    start = 0
    chunk_id = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_body = normalized[start:end].strip()
        if chunk_body:
            chunks.append(
                {
                    "id": f"{_safe_id_part(Path(source).stem)}_chunk_{chunk_id}",
                    "text": chunk_body,
                    "source": source,
                    "start": start,
                    "end": end,
                }
            )
            chunk_id += 1
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def chunk_text_file(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Read and chunk a UTF-8 text file."""
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=str(path))


def chunk_pdf_file(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Extract text from a PDF and chunk it page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pypdf to index PDF files.") from exc

    reader = PdfReader(str(path))
    chunks: list[dict] = []
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            source=f"{path}#page={page_index}",
        )
        for local_index, chunk in enumerate(page_chunks):
            chunk["id"] = f"{_safe_id_part(path.stem)}_p{page_index}_chunk_{local_index}"
            chunk["source"] = str(path)
            chunk["page"] = page_index
        chunks.extend(page_chunks)
    return chunks


def chunk_file(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Read and chunk a supported document file."""
    if path.suffix.lower() == ".pdf":
        return chunk_pdf_file(path, chunk_size=chunk_size, overlap=overlap)
    return chunk_text_file(path, chunk_size=chunk_size, overlap=overlap)


def load_documents(
    docs_dir: Path,
    chunk_size: int,
    overlap: int,
    extensions: set[str],
) -> list[dict]:
    """Load supported documents from a directory and split them into chunks."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {docs_dir}")

    chunks: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            chunks.extend(chunk_file(path, chunk_size=chunk_size, overlap=overlap))
    return chunks
