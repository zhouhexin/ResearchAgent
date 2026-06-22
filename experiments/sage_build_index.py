"""Build a FAISS index for a SAGE semantic chunk corpus or comparison corpus."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAGE comparison FAISS index")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    args = parser.parse_args()

    units = read_jsonl(args.corpus)
    if not units:
        raise RuntimeError(f"No units found in {args.corpus}")

    embedder = Embedder(args.embedding_model)
    embeddings = embedder.encode([unit["text"] for unit in units])
    store = FaissStore(args.index_dir)
    store.build(units, embeddings)
    store.save()
    print(f"Built index with {len(units)} units at {args.index_dir}")


if __name__ == "__main__":
    main()
