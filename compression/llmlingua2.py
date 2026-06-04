"""LLMLingua-2 compression adapter.

This module intentionally keeps LLMLingua behind a small adapter instead of
letting the rest of the project depend on its Python API directly. That gives
the experiment code one stable interface even if the upstream package changes,
and it makes the run details easier to interpret because all LLMLingua-specific
parameters are collected in one place.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from evaluation.token_counter import count_tokens


# Punctuation and line breaks are forced to survive compression because this
# project builds numbered evidence blocks. Keeping separators makes compressed
# snippets easier for the answer model to read and helps avoid accidental merging
# of unrelated claims inside a chunk.
DEFAULT_FORCE_TOKENS = [
    "\n",
    ".",
    "?",
    "!",
    ",",
    ":",
    ";",
    "。",
    "？",
    "！",
    "，",
    "：",
    "；",
]


def llmlingua_version() -> str | None:
    """Return the installed llmlingua package version, if available."""
    try:
        return version("llmlingua")
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=4)
def _load_compressor(model_name: str) -> Any:
    """Load and cache the LLMLingua-2 compressor.

    Loading the model can be slow and may download Hugging Face weights on first
    use. The cache prevents repeated construction during budget-sweep runs where
    the same process compresses many prompts.
    """
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise RuntimeError(
            "LLMLingua is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`, or run without "
            "`--compression llmlingua2`."
        ) from exc

    return PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map="cpu")


def _metadata_from_result(result: dict, original_text: str, compressed_text: str) -> dict:
    """Normalize LLMLingua result fields into this project's run schema."""
    original_tokens = count_tokens(original_text)
    compressed_tokens = count_tokens(compressed_text)
    ratio = compressed_tokens / original_tokens if original_tokens else 0.0

    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": ratio,
        # Preserve upstream fields when present. These may be measured with the
        # compressor tokenizer rather than the experiment tokenizer, so they are
        # logged separately and not used for budget accounting.
        "llmlingua_origin_tokens": result.get("origin_tokens"),
        "llmlingua_compressed_tokens": result.get("compressed_tokens"),
        "llmlingua_ratio": result.get("ratio"),
        "llmlingua_rate": result.get("rate"),
        "llmlingua_saving": result.get("saving"),
    }


def compress_text_llmlingua2(
    text: str,
    *,
    model_name: str,
    rate: float,
    force_tokens: list[str] | None = None,
) -> tuple[str, dict]:
    """Compress one text block with LLMLingua-2.

    The adapter compresses individual chunks rather than the complete prompt.
    That is deliberate: chunk-level compression keeps source metadata and prompt
    citation numbering outside the compressor, so `[1]`, `[2]`, etc. remain
    deterministic and easy to evaluate.
    """
    if not text:
        return "", _metadata_from_result({}, "", "")
    if not 0 < rate <= 1:
        raise ValueError(f"LLMLingua compression rate must be in (0, 1], got {rate}")

    compressor = _load_compressor(model_name)
    result = compressor.compress_prompt(
        text,
        rate=rate,
        force_tokens=force_tokens or DEFAULT_FORCE_TOKENS,
    )
    if isinstance(result, str):
        result = {"compressed_prompt": result}
    if not isinstance(result, dict):
        raise RuntimeError(
            "Unexpected LLMLingua compress_prompt result. Expected dict or str, "
            f"got {type(result).__name__}."
        )

    compressed_text = result.get("compressed_prompt", text)
    return compressed_text, _metadata_from_result(result, text, compressed_text)


def compress_chunks_llmlingua2(
    chunks: list[dict],
    *,
    model_name: str,
    rate: float,
    stage: str = "after-allocation",
) -> tuple[list[dict], dict]:
    """Compress selected chunks and return chunk copies plus aggregate metadata."""
    compressed_chunks: list[dict] = []
    chunk_summaries: list[dict] = []

    for chunk in chunks:
        original_text = chunk.get("text", "")
        compressed_text, summary = compress_text_llmlingua2(
            original_text,
            model_name=model_name,
            rate=rate,
        )

        # Keep the original chunk untouched for reproducibility. The compressed
        # chunk is what enters the prompt, while `original_text` remains available
        # in run details for later inspection and failure analysis.
        item = dict(chunk)
        item["original_text"] = original_text
        item["text"] = compressed_text
        item["original_estimated_tokens"] = summary["original_tokens"]
        item["estimated_tokens"] = summary["compressed_tokens"]
        item["compression"] = {
            "method": "llmlingua2",
            "model_name": model_name,
            "rate": rate,
            **summary,
        }
        compressed_chunks.append(item)

        chunk_summaries.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "page": item.get("page"),
                **summary,
            }
        )

    aggregate = summarize_llmlingua2_chunks(
        compressed_chunks,
        model_name=model_name,
        rate=rate,
        stage=stage,
    )
    return compressed_chunks, aggregate


def summarize_llmlingua2_chunks(
    chunks: list[dict],
    *,
    model_name: str,
    rate: float,
    stage: str,
) -> dict:
    """Summarize LLMLingua-2 metadata for a selected chunk set.

    Before-allocation compression first compresses the whole retrieved candidate
    pool, but only a subset is eventually selected. This helper recomputes the
    aggregate statistics from the final selected chunks so run details describe
    the actual context sent to the LLM, not every compressed candidate.
    """
    chunk_summaries = []
    for chunk in chunks:
        compression = chunk.get("compression", {})
        original_tokens = int(
            compression.get(
                "original_tokens",
                chunk.get("original_estimated_tokens", count_tokens(chunk.get("original_text", ""))),
            )
            or 0
        )
        compressed_tokens = int(
            compression.get("compressed_tokens", count_tokens(chunk.get("text", ""))) or 0
        )
        chunk_summaries.append(
            {
                "id": chunk.get("id"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": (
                    compressed_tokens / original_tokens if original_tokens else 0.0
                ),
                "llmlingua_origin_tokens": compression.get("llmlingua_origin_tokens"),
                "llmlingua_compressed_tokens": compression.get("llmlingua_compressed_tokens"),
                "llmlingua_ratio": compression.get("llmlingua_ratio"),
                "llmlingua_rate": compression.get("llmlingua_rate"),
                "llmlingua_saving": compression.get("llmlingua_saving"),
            }
        )

    original_tokens = sum(item["original_tokens"] for item in chunk_summaries)
    compressed_tokens = sum(item["compressed_tokens"] for item in chunk_summaries)
    return {
        "method": "llmlingua2",
        "stage": stage,
        "model_name": model_name,
        "package_version": llmlingua_version(),
        "rate": rate,
        "original_context_tokens": original_tokens,
        "compressed_context_tokens": compressed_tokens,
        "compression_ratio": compressed_tokens / original_tokens if original_tokens else 0.0,
        "chunks": chunk_summaries,
    }
