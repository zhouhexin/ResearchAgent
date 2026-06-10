"""Local paper title indexing and answer-to-PDF matching for the web API."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import config


PAPER_INDEX_PATH = config.BASE_DIR / "experiments" / "web_paper_index.json"
INDEX_EXTRACTOR_VERSION = 2
TITLE_STOP_MARKERS = (
    "abstract",
    "introduction",
    "keywords",
    "author",
    "authors",
    "arxiv",
    "proceedings",
)


def _paper_id(path: Path) -> str:
    resolved = path.resolve()
    try:
        key = str(resolved.relative_to(config.BASE_DIR.resolve()))
    except ValueError:
        key = str(resolved)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _normalize_for_match(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_title_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -|")


def _looks_like_title_line(line: str) -> bool:
    if len(line) < 4:
        return False
    lower = line.lower()
    if any(marker == lower or lower.startswith(f"{marker}:") for marker in TITLE_STOP_MARKERS):
        return False
    if "@" in line:
        return False
    if line.count(",") >= 2 and ":" not in line:
        return False
    if re.search(r"\b(university|institute|laboratory|department)\b", lower):
        return False
    if re.fullmatch(r"[\d\s.,:/-]+", line):
        return False
    return True


def _looks_like_person_name_line(line: str) -> bool:
    words = line.split()
    if not 2 <= len(words) <= 4:
        return False
    if any(char.isdigit() for char in line):
        return False
    return all(word[:1].isupper() for word in words if word[:1].isalpha())


def _extract_title_from_text(text: str) -> str | None:
    """Extract a likely paper title from first-page PDF text."""
    lines = [_clean_title_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None

    title_lines: list[str] = []
    for line in lines[:40]:
        lower = line.lower()
        if not title_lines and (lower.startswith("arxiv:") or "et al." in lower):
            continue
        if lower in TITLE_STOP_MARKERS or lower.startswith("abstract"):
            break
        if title_lines:
            if "@" in line or lower.startswith("http"):
                break
            if re.search(r"[†‡∗*]|\d", line) and ":" not in line:
                break
            if len(title_lines) >= 2 and _looks_like_person_name_line(line):
                break
        if not _looks_like_title_line(line):
            if title_lines:
                break
            continue
        title_lines.append(line)
        if len(title_lines) >= 3:
            break

    title = _clean_title_line(" ".join(title_lines))
    if len(title.split()) < 4:
        return None
    return title


def _extract_pdf_title(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pypdf to read paper titles.") from exc

    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages[:2]:
        text_parts.append(page.extract_text() or "")
    return _extract_title_from_text("\n".join(text_parts))


def _load_cached_entries(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {str(row.get("path")): row for row in rows if row.get("path")}


def _entry_is_current(entry: dict, path: Path) -> bool:
    stat = path.stat()
    return (
        entry.get("size") == stat.st_size
        and entry.get("mtime_ns") == stat.st_mtime_ns
        and entry.get("extractor_version") == INDEX_EXTRACTOR_VERSION
    )


def build_or_update_paper_index(
    *,
    docs_dir: Path = config.DATA_DIR,
    cache_path: Path = PAPER_INDEX_PATH,
    title_extractor: Callable[[Path], str | None] = _extract_pdf_title,
) -> list[dict]:
    """Build an incremental title index for PDFs under `docs_dir`."""
    docs_dir = Path(docs_dir)
    cache_path = Path(cache_path)
    cached = _load_cached_entries(cache_path)
    entries: list[dict] = []

    for path in sorted(docs_dir.glob("*.pdf")):
        path_key = str(path)
        cached_entry = cached.get(path_key)
        if cached_entry and _entry_is_current(cached_entry, path):
            entries.append(cached_entry)
            continue

        title = title_extractor(path)
        if not title:
            continue
        stat = path.stat()
        entries.append(
            {
                "id": _paper_id(path),
                "title": title,
                "path": path_key,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "extractor_version": INDEX_EXTRACTOR_VERSION,
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entries


def match_papers_in_answer(
    answer: str,
    *,
    docs_dir: Path = config.DATA_DIR,
    cache_path: Path = PAPER_INDEX_PATH,
) -> list[dict]:
    """Return local PDF links for paper titles mentioned in a public answer."""
    entries = build_or_update_paper_index(docs_dir=docs_dir, cache_path=cache_path)
    normalized_answer = _normalize_for_match(answer)
    matches: list[dict] = []
    seen_ids: set[str] = set()

    for entry in entries:
        title = str(entry.get("title") or "")
        paper_id = str(entry.get("id") or "")
        normalized_title = _normalize_for_match(title)
        if not normalized_title or normalized_title not in normalized_answer:
            continue
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        matches.append(
            {
                "id": paper_id,
                "title": title,
                "preview_url": f"/papers/file/{paper_id}",
                "download_url": f"/papers/file/{paper_id}?download=1",
            }
        )
    return matches


def find_paper_path(paper_id: str) -> Path | None:
    """Resolve a public paper id to a local PDF path from the cached index."""
    entries = build_or_update_paper_index()
    for entry in entries:
        if entry.get("id") != paper_id:
            continue
        path = Path(str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(config.DATA_DIR.resolve())
        except ValueError:
            return None
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
    return None
