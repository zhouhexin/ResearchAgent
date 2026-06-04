"""Lightweight query-aware reranking utilities."""

from __future__ import annotations

import math
import re
from collections import Counter

from scoring.density import density_score
from scoring.relevance import relevance_score

WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize English words, numbers, and Chinese characters for lexical scoring."""
    return [token.lower() for token in WORD_RE.findall(text)]


def keyword_overlap_score(query: str, text: str) -> float:
    """Score how many query terms appear in the chunk."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def bm25_like_score(query: str, text: str, k1: float = 1.2) -> float:
    """Small BM25-style score without requiring a corpus-wide index."""
    query_terms = tokenize(query)
    text_terms = tokenize(text)
    if not query_terms or not text_terms:
        return 0.0

    counts = Counter(text_terms)
    text_len = len(text_terms)
    raw_score = 0.0
    for term in set(query_terms):
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        raw_score += ((k1 + 1.0) * tf) / (k1 + tf)

    return raw_score / math.log(text_len + 2.0)


def rerank_score(query: str, chunk: dict) -> float:
    """Combine semantic, lexical, and density signals for reranking."""
    text = chunk.get("text", "")
    semantic = relevance_score(float(chunk.get("score", 0.0)))
    overlap = keyword_overlap_score(query, text)
    bm25 = min(1.0, bm25_like_score(query, text))
    density = density_score(text)
    return 0.45 * semantic + 0.30 * overlap + 0.20 * bm25 + 0.05 * density


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Return chunks ordered by query-aware rerank score."""
    reranked = []
    for chunk in chunks:
        item = dict(chunk)
        item["rerank_score"] = rerank_score(query, item)
        item["keyword_overlap"] = keyword_overlap_score(query, item.get("text", ""))
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)
