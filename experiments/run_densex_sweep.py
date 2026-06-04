"""Run DenseX granularity experiments over selected QA pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app import answer_query


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_questions(path: Path, ids: set[str]) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("id") in ids:
            questions.append(item)
    missing = ids - {item["id"] for item in questions}
    if missing:
        raise ValueError(f"Question ids not found: {sorted(missing)}")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DenseX granularity sweep")
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
    parser.add_argument("--index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--budgets", default="500,1000,1500")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "dynamic", "rerank"])
    parser.add_argument("--run-label-prefix", default="densex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    question_ids = set(_parse_csv(args.question_ids))
    questions = _load_questions(args.questions, question_ids)
    budgets = _parse_csv_ints(args.budgets)
    granularities = _parse_csv(args.granularities)

    for question in questions:
        for granularity in granularities:
            index_dir = args.index_base_dir / granularity
            for budget in budgets:
                label = f"{args.run_label_prefix}_{granularity}_{question['id']}"
                print(
                    "\n=== "
                    f"qid={question['id']} granularity={granularity} "
                    f"strategy={args.strategy} top_k={args.top_k} budget={budget} "
                    "==="
                )
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
