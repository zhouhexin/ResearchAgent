"""Token counting utilities for fixed-budget experiments."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter:
    """Count tokens with tiktoken when available, otherwise a deterministic fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self.encoding = None
        try:
            import tiktoken
        except ImportError:
            return

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoding = None

    def count(self, text: str) -> int:
        """Return a stable token count estimate for text."""
        if not text:
            return 0
        if self.encoding is not None:
            return len(self.encoding.encode(text))
        return len(_TOKEN_RE.findall(text))

    def count_chunks(self, chunks: list[dict]) -> int:
        """Count tokens for selected context chunks."""
        return sum(self.count(chunk.get("text", "")) for chunk in chunks)


DEFAULT_COUNTER = TokenCounter()


def count_tokens(text: str) -> int:
    """Count tokens using the default counter."""
    return DEFAULT_COUNTER.count(text)


def count_context_tokens(chunks: list[dict]) -> int:
    """Count tokens in selected context chunks."""
    return DEFAULT_COUNTER.count_chunks(chunks)
