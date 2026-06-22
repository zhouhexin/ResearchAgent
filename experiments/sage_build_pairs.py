"""Build sentence-pair labels for SAGE-style semantic segmentation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from densex.corpus import load_chunks_from_metadata, write_jsonl
from sage_segmenter.dataset import build_sentence_pair_rows


def _filter_chunks(chunks: list[dict], source_contains: str) -> list[dict]:
    if not source_contains:
        return chunks
    needle = source_contains.lower()
    return [chunk for chunk in chunks if needle in str(chunk.get("source", "")).lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAGE sentence-pair training data")
    parser.add_argument("--metadata", type=Path, default=config.INDEX_DIR / "metadata.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "sage_pairs" / "train.jsonl")
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--source-contains", default="")
    parser.add_argument("--limit", type=int, default=0, help="Optional chunk limit for smoke tests")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-sentence-chars", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    chunks = _filter_chunks(load_chunks_from_metadata(args.metadata), args.source_contains)
    if args.limit > 0:
        chunks = chunks[: args.limit]
    if not chunks:
        raise RuntimeError("No chunks matched the requested metadata/filter")

    rows = build_sentence_pair_rows(
        chunks,
        min_sentence_chars=args.min_sentence_chars,
        max_pairs=args.max_pairs,
    )
    if not rows:
        raise RuntimeError("No sentence pairs were generated")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    if args.validation_output and args.validation_ratio > 0:
        validation_count = max(1, int(len(rows) * args.validation_ratio))
        validation_rows = rows[:validation_count]
        train_rows = rows[validation_count:]
        write_jsonl(args.output, train_rows)
        write_jsonl(args.validation_output, validation_rows)
        print(f"Wrote {len(train_rows)} train pairs to {args.output}")
        print(f"Wrote {len(validation_rows)} validation pairs to {args.validation_output}")
    else:
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
