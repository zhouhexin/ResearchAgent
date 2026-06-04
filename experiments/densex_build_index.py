"""Build FAISS indexes for DenseX granularity corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from densex.corpus import read_jsonl
from retrieval.embed import Embedder
from retrieval.faiss_store import FaissStore


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DenseX FAISS indexes")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "experiments" / "densex_corpus")
    parser.add_argument("--index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    args = parser.parse_args()

    embedder = Embedder(args.embedding_model)
    for granularity in _parse_csv(args.granularities):
        corpus_path = args.corpus_dir / f"{granularity}.jsonl"
        units = read_jsonl(corpus_path)
        if not units:
            raise RuntimeError(f"No units found in {corpus_path}")
        embeddings = embedder.encode([unit["text"] for unit in units])
        index_dir = args.index_base_dir / granularity
        store = FaissStore(index_dir)
        store.build(units, embeddings)
        store.save()
        print(f"Built {granularity} index with {len(units)} units at {index_dir}")


if __name__ == "__main__":
    main()
