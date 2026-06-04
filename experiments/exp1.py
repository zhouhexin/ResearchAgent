"""Run the baseline allocation experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app import answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: baseline allocation")
    parser.add_argument("--query", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answer_query(
        query=args.query,
        index_dir=config.INDEX_DIR,
        strategy="baseline",
        top_k=config.TOP_K,
        context_budget=config.CONTEXT_BUDGET,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
