"""Run fine-grained retrieval with parent chunk context."""

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
from densex.parent_aggregation import aggregate_fine_hits_to_parent_chunks, load_parent_chunks
from retrieval.retriever import Retriever


ACDEPTH_QA_IDS = [
    "always_clear_depth_contributions",
    "always_clear_depth_eval_datasets",
    "always_clear_depth_ablation_components",
    "always_clear_depth_sota_comparison_methods",
]

DEPTHDARK_QA_IDS = [
    "depthdark_contributions",
    "depthdark_eval_datasets",
    "depthdark_training_datasets",
    "depthdark_ablation_components",
    "depthdark_sota_comparison_methods",
]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _question_ids(args: argparse.Namespace) -> set[str]:
    if args.question_ids:
        return set(_parse_csv(args.question_ids))
    if args.question_set == "acdepth":
        return set(ACDEPTH_QA_IDS)
    if args.question_set == "depthdark":
        return set(DEPTHDARK_QA_IDS)
    return set(ACDEPTH_QA_IDS + DEPTHDARK_QA_IDS)


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
    parser = argparse.ArgumentParser(description="Run DenseX fine-to-chunk sweep")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--question-set", choices=["fixed", "acdepth", "depthdark"], default="fixed")
    parser.add_argument("--question-ids", default="", help="Override question IDs as a comma-separated list")
    parser.add_argument("--fine-index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--parent-metadata", type=Path, default=config.INDEX_DIR / "metadata.json")
    parser.add_argument("--fine-granularities", default="sentence,proposition")
    parser.add_argument("--budgets", default="500,1000,1500")
    parser.add_argument("--fine-top-m", type=int, default=150)
    parser.add_argument("--parent-top-k", type=int, default=50)
    parser.add_argument("--aggregation-top-children", type=int, default=3)
    parser.add_argument("--aggregation-child-sum-weight", type=float, default=0.1)
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "dynamic", "rerank"])
    parser.add_argument("--run-label-prefix", default="qa_parent_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    question_ids = _question_ids(args)
    questions = _load_questions(args.questions, question_ids)
    budgets = _parse_csv_ints(args.budgets)
    fine_granularities = _parse_csv(args.fine_granularities)
    parent_chunks = load_parent_chunks(args.parent_metadata)

    retrievers = {
        granularity: Retriever(
            index_dir=args.fine_index_base_dir / granularity,
            embedding_model=config.EMBEDDING_MODEL,
        )
        for granularity in fine_granularities
    }

    for question in questions:
        for granularity in fine_granularities:
            fine_hits = retrievers[granularity].retrieve(question["query"], top_k=args.fine_top_m)
            parent_candidates = aggregate_fine_hits_to_parent_chunks(
                fine_hits,
                parent_chunks,
                parent_top_k=args.parent_top_k,
                top_child_count=args.aggregation_top_children,
                child_sum_weight=args.aggregation_child_sum_weight,
            )
            if not parent_candidates:
                raise RuntimeError(
                    f"No parent chunks found for question={question['id']} granularity={granularity}"
                )

            for budget in budgets:
                label = f"{args.run_label_prefix}_{granularity}-to-chunk_{question['id']}"
                print(
                    "\n=== "
                    f"qid={question['id']} granularity={granularity}-to-chunk "
                    f"strategy={args.strategy} fine_top_m={args.fine_top_m} "
                    f"parent_top_k={args.parent_top_k} budget={budget} "
                    "==="
                )
                answer_query(
                    query=question["query"],
                    index_dir=config.INDEX_DIR,
                    strategy=args.strategy,
                    top_k=args.parent_top_k,
                    context_budget=budget,
                    compression="none",
                    compression_stage="after-allocation",
                    run_label=label,
                    dry_run=args.dry_run,
                    retrieved_chunks_override=parent_candidates,
                )


if __name__ == "__main__":
    main()
