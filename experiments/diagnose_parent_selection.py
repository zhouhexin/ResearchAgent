"""Diagnose where fine-to-parent retrieval loses ACDepth evidence.

This script does not call the LLM and does not create run JSON files. It
replays the fine-grained retrieval -> parent aggregation -> context selection
pipeline and writes CSVs that show whether gold answer items and gold evidence
pages survive each stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from allocation.baseline import allocate_baseline
from allocation.dynamic_budget import allocate_dynamic
from allocation.rerank_budget import allocate_rerank
from densex.parent_aggregation import aggregate_fine_hits_to_parent_chunks, load_parent_chunks
from evaluation.list_accuracy import normalize_text
from evaluation.token_counter import count_context_tokens, count_tokens


ACDEPTH_QA_IDS = [
    "always_clear_depth_contributions",
    "always_clear_depth_eval_datasets",
    "always_clear_depth_ablation_components",
    "always_clear_depth_sota_comparison_methods",
]

CONTENT_QUERIES = {
    "always_clear_depth_contributions": (
        "main contributions proposed framework adverse weather robust monocular depth estimation "
        "multi tuple degradation multi granularity knowledge distillation ordinal guidance"
    ),
    "always_clear_depth_eval_datasets": (
        "evaluated datasets experiments benchmarks nuScenes RobotCar adverse weather test set"
    ),
    "always_clear_depth_ablation_components": (
        "ablation study components distillation learning ordinal guidance distillation "
        "feature consistency constraint"
    ),
    "always_clear_depth_sota_comparison_methods": (
        "comparison with state of the art baselines Monodepth2 PackNet SfM RNW md4all "
        "DMMDE DefeatNet ADIDS WSGD"
    ),
}

SOURCE_BASENAME_ALIASES = {
    "always_clear_depth": {
        "acd.pdf",
        "always clear depth- robust monocular depth estimation under adverse weather.pdf",
    }
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_questions(path: Path, ids: list[str]) -> list[dict]:
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        by_id[item["id"]] = item

    missing = [question_id for question_id in ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Question ids not found in {path}: {missing}")
    return [by_id[question_id] for question_id in ids]


def _resolve_fine_index_dir(
    granularity: str,
    index_base_dir: Path,
    *,
    allow_dedup_fallback: bool,
) -> Path:
    index_dir = index_base_dir / granularity
    if index_dir.exists():
        return index_dir

    dedup_dir = index_base_dir / f"{granularity}_dedup"
    if allow_dedup_fallback and dedup_dir.exists():
        return dedup_dir

    return index_dir


def _query_for(question: dict, mode: str) -> str:
    if mode == "original":
        return question["query"]
    if mode == "retrieval-query":
        return question.get("retrieval_query") or question["query"]
    if mode == "content-only":
        return question.get("retrieval_query") or CONTENT_QUERIES.get(question["id"], question["query"])
    raise ValueError(f"Unsupported query mode: {mode}")


def _contains_alias(text: str, item: dict) -> bool:
    normalized = normalize_text(text)
    aliases = [item.get("name", ""), *item.get("aliases", [])]
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if normalized_alias and f" {normalized_alias} " in f" {normalized} ":
            return True
    return False


def _chunk_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(key, "") or "")
        for key in ("id", "source", "paper_title", "text")
    )


def _matched_gold_ids(chunk: dict, question: dict) -> list[str]:
    haystack = _chunk_text(chunk)
    return [
        item["id"]
        for item in question.get("gold_items", [])
        if _contains_alias(haystack, item)
    ]


def _source_basename(value: object) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def _normalized_source_basename(value: object) -> str:
    return _source_basename(value).lower()


def _source_matches(hit_source: object, evidence: dict) -> bool:
    hit_basename = _normalized_source_basename(hit_source)
    evidence_basename = _normalized_source_basename(evidence.get("source", ""))
    if hit_basename == evidence_basename:
        return True

    paper_id = str(evidence.get("paper_id", "") or "")
    aliases = SOURCE_BASENAME_ALIASES.get(paper_id, set())
    return hit_basename in aliases and evidence_basename in aliases


def _pages_equal(left: object, right: object) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left or "") == str(right or "")


def _format_evidence_ref(evidence: dict) -> str:
    source = _source_basename(evidence.get("source", ""))
    page = evidence.get("page", "")
    evidence_for = evidence.get("evidence_for", "")
    return f"{source}:p{page}:{evidence_for}"


def _gold_evidence_refs(question: dict) -> set[str]:
    return {_format_evidence_ref(item) for item in question.get("gold_evidence", [])}


def _matched_evidence_refs(chunk: dict, question: dict) -> list[str]:
    chunk_page = chunk.get("page")
    refs = []
    for evidence in question.get("gold_evidence", []):
        if _source_matches(chunk.get("source", ""), evidence) and _pages_equal(chunk_page, evidence.get("page")):
            refs.append(_format_evidence_ref(evidence))
    return refs


def _allocate_contexts(query: str, chunks: list[dict], *, strategy: str, budget: int) -> list[dict]:
    if strategy == "baseline":
        return allocate_baseline(chunks, budget=budget)
    if strategy == "dynamic":
        return allocate_dynamic(chunks, budget=budget)
    if strategy == "rerank":
        return allocate_rerank(query, chunks, budget=budget)
    raise ValueError(f"Unknown allocation strategy: {strategy}")


def _ids(values: list[str]) -> str:
    return ",".join(sorted({value for value in values if value}))


def _refs(values: list[str]) -> str:
    return "|".join(sorted({value for value in values if value}))


def _first_rank(candidates: list[dict], key: str) -> int | str:
    for candidate in candidates:
        if candidate.get(key):
            return candidate["parent_rank"]
    return ""


def _selected_order_by_id(chunks: list[dict]) -> dict[str, int]:
    return {
        str(chunk.get("id")): order
        for order, chunk in enumerate(chunks, start=1)
        if chunk.get("id") is not None
    }


def _annotate_parent_candidates(candidates: list[dict], question: dict) -> list[dict]:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item["parent_rank"] = rank
        item["matched_gold_ids"] = _matched_gold_ids(item, question)
        item["matched_evidence_refs"] = _matched_evidence_refs(item, question)
        item["estimated_tokens"] = count_tokens(item.get("text", ""))
        rows.append(item)
    return rows


def _stage_matches(chunks: list[dict], question: dict) -> tuple[set[str], set[str]]:
    gold_ids = set()
    evidence_refs = set()
    for chunk in chunks:
        gold_ids.update(_matched_gold_ids(chunk, question))
        evidence_refs.update(_matched_evidence_refs(chunk, question))
    return gold_ids, evidence_refs


def _recall(matched: set[str], total: int) -> float:
    return len(matched) / total if total else 0.0


def _summarize(
    *,
    question: dict,
    query_mode: str,
    query_used: str,
    granularity: str,
    strategy: str,
    budget: int,
    fine_top_m: int,
    parent_top_k: int,
    annotated_candidates: list[dict],
    selected: list[dict],
) -> dict:
    gold_items = question.get("gold_items", [])
    gold_count = len(gold_items)
    evidence_refs = _gold_evidence_refs(question)
    evidence_count = len(evidence_refs)

    candidate_gold = {
        item
        for candidate in annotated_candidates
        for item in candidate.get("matched_gold_ids", [])
    }
    candidate_evidence = {
        item
        for candidate in annotated_candidates
        for item in candidate.get("matched_evidence_refs", [])
    }
    selected_gold, selected_evidence = _stage_matches(selected, question)

    selected_candidate_ids = {str(item.get("id")) for item in selected}
    selected_candidates = [
        candidate
        for candidate in annotated_candidates
        if str(candidate.get("id")) in selected_candidate_ids
    ]

    return {
        "question_id": question["id"],
        "query_mode": query_mode,
        "query_used": query_used,
        "granularity": f"{granularity}-to-chunk",
        "strategy": strategy,
        "budget": budget,
        "fine_top_m": fine_top_m,
        "parent_top_k": parent_top_k,
        "parent_candidate_count": len(annotated_candidates),
        "selected_count": len(selected),
        "context_tokens": count_context_tokens(selected),
        "parent_candidate_gold_recall": _recall(candidate_gold, gold_count),
        "parent_candidate_evidence_recall": _recall(candidate_evidence, evidence_count),
        "selected_gold_recall": _recall(selected_gold, gold_count),
        "selected_evidence_recall": _recall(selected_evidence, evidence_count),
        "first_gold_parent_rank": _first_rank(annotated_candidates, "matched_gold_ids"),
        "first_evidence_parent_rank": _first_rank(annotated_candidates, "matched_evidence_refs"),
        "selected_first_gold_parent_rank": _first_rank(selected_candidates, "matched_gold_ids"),
        "selected_first_evidence_parent_rank": _first_rank(selected_candidates, "matched_evidence_refs"),
        "parent_candidate_gold_ids": _ids(list(candidate_gold)),
        "parent_candidate_evidence_refs": _refs(list(candidate_evidence)),
        "selected_gold_ids": _ids(list(selected_gold)),
        "selected_evidence_refs": _refs(list(selected_evidence)),
        "missing_selected_gold_ids": _ids([item["id"] for item in gold_items if item["id"] not in selected_gold]),
        "missing_selected_evidence_refs": _refs(list(evidence_refs - selected_evidence)),
    }


def _detail_rows(
    *,
    question: dict,
    query_mode: str,
    query_used: str,
    granularity: str,
    budget: int,
    annotated_candidates: list[dict],
    selected: list[dict],
    text_limit: int,
) -> list[dict]:
    selected_order = _selected_order_by_id(selected)
    rows = []
    for candidate in annotated_candidates:
        fine_info = candidate.get("fine_to_chunk") or {}
        chunk_id = str(candidate.get("id", ""))
        text = " ".join(str(candidate.get("text", "") or "").split())
        rows.append(
            {
                "question_id": question["id"],
                "query_mode": query_mode,
                "query_used": query_used,
                "granularity": f"{granularity}-to-chunk",
                "budget": budget,
                "parent_rank": candidate.get("parent_rank", ""),
                "selected": chunk_id in selected_order,
                "selected_order": selected_order.get(chunk_id, ""),
                "score": candidate.get("score", ""),
                "source": candidate.get("source", ""),
                "page": candidate.get("page", ""),
                "parent_chunk_id": chunk_id,
                "estimated_tokens": candidate.get("estimated_tokens", ""),
                "matched_child_count": fine_info.get("matched_child_count", ""),
                "deduplicated_child_count": fine_info.get("deduplicated_child_count", ""),
                "max_child_score": fine_info.get("max_child_score", ""),
                "top_child_score_sum": fine_info.get("top_child_score_sum", ""),
                "top_child_ids": "|".join(str(item) for item in fine_info.get("top_child_ids", [])),
                "matched_gold_ids": ",".join(candidate.get("matched_gold_ids", [])),
                "matched_evidence_refs": "|".join(candidate.get("matched_evidence_refs", [])),
                "text": text[:text_limit],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fine-to-parent selected context failures")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--question-ids", default=",".join(ACDEPTH_QA_IDS))
    parser.add_argument("--fine-index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--parent-metadata", type=Path, default=config.INDEX_DIR / "metadata.json")
    parser.add_argument("--fine-granularities", default="sentence,proposition")
    parser.add_argument("--budgets", default="500,1000,1500")
    parser.add_argument("--fine-top-m", type=int, default=150)
    parser.add_argument("--parent-top-k", type=int, default=50)
    parser.add_argument("--aggregation-top-children", type=int, default=3)
    parser.add_argument("--aggregation-child-sum-weight", type=float, default=0.1)
    parser.add_argument(
        "--fine-hit-dedup",
        default="none",
        choices=["none", "exact-per-parent"],
    )
    parser.add_argument(
        "--allow-dedup-fallback",
        action="store_true",
        help="Use sentence_dedup/proposition_dedup if sentence/proposition indexes are missing",
    )
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "dynamic", "rerank"])
    parser.add_argument(
        "--query-mode",
        default="original",
        choices=["original", "retrieval-query", "content-only"],
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "diagnostics")
    parser.add_argument("--output-prefix", default="acdepth_parent_selection")
    parser.add_argument("--text-limit", type=int, default=500)
    args = parser.parse_args()

    from retrieval.retriever import Retriever

    questions = _load_questions(args.questions, _parse_csv(args.question_ids))
    fine_granularities = _parse_csv(args.fine_granularities)
    budgets = _parse_csv_ints(args.budgets)
    parent_chunks = load_parent_chunks(args.parent_metadata)
    retrievers = {
        granularity: Retriever(
            index_dir=_resolve_fine_index_dir(
                granularity,
                args.fine_index_base_dir,
                allow_dedup_fallback=args.allow_dedup_fallback,
            ),
            embedding_model=config.EMBEDDING_MODEL,
        )
        for granularity in fine_granularities
    }

    summary_rows = []
    detail_rows = []
    for question in questions:
        query_used = _query_for(question, args.query_mode)
        for granularity in fine_granularities:
            fine_hits = retrievers[granularity].retrieve(query_used, top_k=args.fine_top_m)
            parent_candidates = aggregate_fine_hits_to_parent_chunks(
                fine_hits,
                parent_chunks,
                parent_top_k=args.parent_top_k,
                top_child_count=args.aggregation_top_children,
                child_sum_weight=args.aggregation_child_sum_weight,
                fine_hit_dedup=args.fine_hit_dedup,
            )
            annotated = _annotate_parent_candidates(parent_candidates, question)
            for budget in budgets:
                selected = _allocate_contexts(
                    query_used,
                    annotated,
                    strategy=args.strategy,
                    budget=budget,
                )
                summary_rows.append(
                    _summarize(
                        question=question,
                        query_mode=args.query_mode,
                        query_used=query_used,
                        granularity=granularity,
                        strategy=args.strategy,
                        budget=budget,
                        fine_top_m=args.fine_top_m,
                        parent_top_k=args.parent_top_k,
                        annotated_candidates=annotated,
                        selected=selected,
                    )
                )
                detail_rows.extend(
                    _detail_rows(
                        question=question,
                        query_mode=args.query_mode,
                        query_used=query_used,
                        granularity=granularity,
                        budget=budget,
                        annotated_candidates=annotated,
                        selected=selected,
                        text_limit=args.text_limit,
                    )
                )

    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    details_path = args.output_dir / f"{args.output_prefix}_details.csv"
    _write_csv(
        summary_path,
        summary_rows,
        [
            "question_id",
            "query_mode",
            "query_used",
            "granularity",
            "strategy",
            "budget",
            "fine_top_m",
            "parent_top_k",
            "parent_candidate_count",
            "selected_count",
            "context_tokens",
            "parent_candidate_gold_recall",
            "parent_candidate_evidence_recall",
            "selected_gold_recall",
            "selected_evidence_recall",
            "first_gold_parent_rank",
            "first_evidence_parent_rank",
            "selected_first_gold_parent_rank",
            "selected_first_evidence_parent_rank",
            "parent_candidate_gold_ids",
            "parent_candidate_evidence_refs",
            "selected_gold_ids",
            "selected_evidence_refs",
            "missing_selected_gold_ids",
            "missing_selected_evidence_refs",
        ],
    )
    _write_csv(
        details_path,
        detail_rows,
        [
            "question_id",
            "query_mode",
            "query_used",
            "granularity",
            "budget",
            "parent_rank",
            "selected",
            "selected_order",
            "score",
            "source",
            "page",
            "parent_chunk_id",
            "estimated_tokens",
            "matched_child_count",
            "deduplicated_child_count",
            "max_child_score",
            "top_child_score_sum",
            "top_child_ids",
            "matched_gold_ids",
            "matched_evidence_refs",
            "text",
        ],
    )
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")
    print(f"Wrote {len(detail_rows)} detail rows to {details_path}")


if __name__ == "__main__":
    main()
