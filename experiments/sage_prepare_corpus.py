"""Generate semantic chunk corpus with a trained SAGE-style segmenter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from densex.corpus import load_chunks_from_metadata, write_jsonl
from sage_segmenter.segmenter import SemanticSegmenter, make_semantic_chunk_units
from sage_segmenter.sentence_utils import load_page_records_from_docs, reconstruct_page_texts_from_chunks


def _device_arg(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_chunks(chunks: list[dict], source_contains: str) -> list[dict]:
    if not source_contains:
        return chunks
    needle = source_contains.lower()
    return [chunk for chunk in chunks if needle in str(chunk.get("source", "")).lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SAGE semantic chunk corpus")
    parser.add_argument("--metadata", type=Path, default=config.INDEX_DIR / "metadata.json")
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Optional document directory. When set, read original files directly and preserve paragraph breaks.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "sage_corpus" / "semantic_chunk.jsonl")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-sentence-chars", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--source-contains", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    if args.docs is not None:
        page_records = load_page_records_from_docs(
            args.docs,
            source_contains=args.source_contains,
            limit=args.limit,
        )
        input_description = f"docs={args.docs}"
    else:
        chunks = _filter_chunks(load_chunks_from_metadata(args.metadata), args.source_contains)
        if args.limit > 0:
            chunks = chunks[: args.limit]
        if not chunks:
            raise RuntimeError("No chunks matched the requested metadata/filter")
        page_records = reconstruct_page_texts_from_chunks(chunks)
        input_description = f"metadata={args.metadata}"
    if not page_records:
        raise RuntimeError("No page records matched the requested input/filter")
    print(f"Loaded {len(page_records)} page records from {input_description}", flush=True)

    segmenter = SemanticSegmenter.from_model_dir(
        args.model_dir,
        embedding_model=args.embedding_model,
        device=_device_arg(args.device),
    )
    units = make_semantic_chunk_units(
        page_records,
        scorer=segmenter.score_sentences,
        threshold=args.threshold,
        min_sentence_chars=args.min_sentence_chars,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    if not units:
        raise RuntimeError("No semantic chunks were generated")
    write_jsonl(args.output, units)
    print(f"Wrote {len(units)} semantic chunks to {args.output}")


if __name__ == "__main__":
    main()
