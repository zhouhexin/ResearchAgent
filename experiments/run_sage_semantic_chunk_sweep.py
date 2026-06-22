"""Run QA comparison between fixed chunk and SAGE semantic chunk indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_questions(path: Path, ids: set[str] | None) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if ids is None or item.get("id") in ids:
            questions.append(item)
    if ids is not None:
        missing = ids - {item["id"] for item in questions}
        if missing:
            raise ValueError(f"Question ids not found: {sorted(missing)}")
    if not questions:
        raise ValueError(f"No questions selected from {path}")
    return questions


def _load_answer_query(embedding_model: str):
    config.EMBEDDING_MODEL = embedding_model
    from app import answer_query

    return answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE semantic chunk sweep")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument(
        "--question-ids",
        default=(
            "always_clear_depth_contributions,"
            "always_clear_depth_eval_datasets,"
            "always_clear_depth_ablation_components,"
            "always_clear_depth_sota_comparison_methods"
        ),
    )
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--chunk-index-dir", type=Path, default=PROJECT_ROOT / "storage" / "sage" / "chunk")
    parser.add_argument(
        "--semantic-index-dir",
        type=Path,
        default=PROJECT_ROOT / "storage" / "sage" / "semantic_chunk",
    )
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--budgets", default="300,500,1000,1500")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "dynamic", "rerank"])
    parser.add_argument("--run-label-prefix", default="sage_semantic_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    question_ids = None if args.all_questions else set(_parse_csv(args.question_ids))
    questions = _load_questions(args.questions, question_ids)
    budgets = _parse_csv_ints(args.budgets)
    index_dirs = {
        "chunk": args.chunk_index_dir,
        "semantic_chunk": args.semantic_index_dir,
    }
    answer_query = None if args.dry_run else _load_answer_query(args.embedding_model)

    for question in questions:
        for granularity, index_dir in index_dirs.items():
            for budget in budgets:
                label = f"{args.run_label_prefix}_{granularity}_{question['id']}"
                print(
                    "\n=== "
                    f"qid={question['id']} granularity={granularity} "
                    f"strategy={args.strategy} top_k={args.top_k} budget={budget} "
                    "==="
                )
                if answer_query is not None:
                    answer_query(
                        query=question["query"],
                        index_dir=index_dir,
                        strategy=args.strategy,
                        top_k=args.top_k,
                        context_budget=budget,
                        compression="none",
                        compression_stage="after-allocation",
                        run_label=label,
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    main()
