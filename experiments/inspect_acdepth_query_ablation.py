"""Compare ACDepth retrieval with original/title/content/source-filtered queries.

The goal is to test whether ACDepth relevance failures are caused by the paper
title prefix in questions, e.g. "Always Clear Depth ...", dominating embedding
similarity and pulling title-like propositions to the top.

The script does not call the LLM and does not modify experiment runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.list_accuracy import normalize_text
from experiments.inspect_acdepth_relevance import (
    ACDEPTH_QA_IDS,
    _format_evidence_ref,
    _matched_gold_ids,
    _parse_csv,
    _resolve_index_dir,
)


ACDEPTH_TITLE = "Always Clear Depth Robust Monocular Depth Estimation Under Adverse Weather"
ACDEPTH_SOURCE = "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf"

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


def _build_query_variants(question: dict, title: str) -> dict[str, str]:
    content_query = question.get("retrieval_query") or CONTENT_QUERIES.get(question["id"]) or question["query"]
    return {
        "original": question["query"],
        "title_only": title,
        "content_only": content_query,
        "source_filtered": content_query,
    }


def _source_basename(value: str) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def _source_matches(hit: dict, target_source: str) -> bool:
    return _source_basename(str(hit.get("source", ""))) == _source_basename(target_source)


def _pages_equal(left: object, right: object) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left or "") == str(right or "")


def _matched_evidence_refs(hit: dict, question: dict) -> list[str]:
    refs = []
    hit_source = _source_basename(str(hit.get("source", "")))
    for evidence in question.get("gold_evidence", []):
        evidence_source = _source_basename(str(evidence.get("source", "")))
        if hit_source == evidence_source and _pages_equal(hit.get("page"), evidence.get("page")):
            refs.append(_format_evidence_ref(evidence))
    return refs


def _is_title_junk(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    title_patterns = [
        "the title of the paper is",
        "the paper has a section called page",
        "the paper has a section called",
    ]
    if any(pattern in normalized for pattern in title_patterns):
        return True

    metric_terms = ["absrel", "rmse", "sqrel", "delta", "rmse log"]
    metric_hits = sum(term in normalized for term in metric_terms)
    token_count = len(normalized.split())
    if metric_hits >= 3 and token_count <= 30:
        return True

    if token_count <= 4 and not re.search(r"[a-zA-Z\u4e00-\u9fff]{8,}", normalized):
        return True
    return False


def _load_retriever_class():
    try:
        from retrieval.retriever import Retriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing retrieval dependency. Run this script in the same Python environment "
            "used for indexing/experiments, for example the environment with numpy, "
            "faiss-cpu, and sentence-transformers installed."
        ) from exc
    return Retriever


def _clean_text(text: str, limit: int) -> str:
    value = " ".join(str(text or "").replace("\x00", " ").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _make_hit_row(
    question: dict,
    granularity: str,
    variant: str,
    query_used: str,
    rank: int,
    raw_rank: int,
    hit: dict,
    *,
    text_limit: int,
) -> dict:
    text = str(hit.get("text", "") or "")
    matched_gold_ids = _matched_gold_ids(hit, question)
    matched_evidence_refs = _matched_evidence_refs(hit, question)
    return {
        "question_id": question["id"],
        "granularity": granularity,
        "variant": variant,
        "query_used": query_used,
        "rank": rank,
        "raw_rank": raw_rank,
        "score": hit.get("score", ""),
        "source": hit.get("source", ""),
        "page": hit.get("page", ""),
        "parent_chunk_id": hit.get("parent_chunk_id", hit.get("id", "")),
        "unit_id": hit.get("id", ""),
        "matched_gold_ids": ",".join(matched_gold_ids),
        "matched_evidence_refs": "|".join(matched_evidence_refs),
        "is_gold_alias_hit": bool(matched_gold_ids),
        "is_evidence_page_hit": bool(matched_evidence_refs),
        "is_title_junk": _is_title_junk(text),
        "text": _clean_text(text, text_limit),
    }


def _overlap_rate(rows: list[dict], title_only_rows: list[dict]) -> float:
    if not rows:
        return 0.0
    title_ids = {row.get("unit_id") for row in title_only_rows if row.get("unit_id")}
    if not title_ids:
        return 0.0
    overlap = sum(1 for row in rows if row.get("unit_id") in title_ids)
    return overlap / len(rows)


def _summarize_variant_hits(
    question_id: str,
    granularity: str,
    variant: str,
    rows: list[dict],
    *,
    title_only_rows: list[dict],
) -> dict:
    gold_rows = [row for row in rows if row.get("matched_gold_ids")]
    evidence_rows = [row for row in rows if row.get("matched_evidence_refs")]
    title_junk_count = sum(1 for row in rows if row.get("is_title_junk"))
    matched_gold_ids = sorted(
        {
            item
            for row in gold_rows
            for item in str(row.get("matched_gold_ids", "")).split(",")
            if item
        }
    )
    matched_evidence_refs = sorted(
        {
            item
            for row in evidence_rows
            for item in str(row.get("matched_evidence_refs", "")).split("|")
            if item
        }
    )
    return {
        "question_id": question_id,
        "granularity": granularity,
        "variant": variant,
        "query_used": rows[0].get("query_used", "") if rows else "",
        "hit_count": len(rows),
        "top_score": rows[0].get("score", "") if rows else "",
        "first_gold_alias_rank": gold_rows[0]["rank"] if gold_rows else "",
        "first_evidence_page_rank": evidence_rows[0]["rank"] if evidence_rows else "",
        "gold_alias_hit_count": len(gold_rows),
        "evidence_page_hit_count": len(evidence_rows),
        "title_junk_count": title_junk_count,
        "title_junk_rate": title_junk_count / len(rows) if rows else 0.0,
        "overlap_with_title_only": 1.0 if variant == "title_only" else _overlap_rate(rows, title_only_rows),
        "matched_gold_ids_top_m": ",".join(matched_gold_ids),
        "matched_evidence_refs_top_m": "|".join(matched_evidence_refs),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACDepth query ablation diagnostics")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--question-ids", default=",".join(ACDEPTH_QA_IDS))
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--top-m", type=int, default=100)
    parser.add_argument(
        "--source-filter-pool",
        type=int,
        default=1000,
        help="How many hits to retrieve before filtering source_filtered to the ACDepth paper",
    )
    parser.add_argument("--index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--chunk-index-dir", type=Path, default=PROJECT_ROOT / "storage")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "diagnostics")
    parser.add_argument("--output-prefix", default="acdepth_query_ablation")
    parser.add_argument("--title", default=ACDEPTH_TITLE)
    parser.add_argument("--target-source", default=ACDEPTH_SOURCE)
    parser.add_argument("--allow-dedup-fallback", action="store_true")
    parser.add_argument("--text-limit", type=int, default=500)
    args = parser.parse_args()

    question_ids = _parse_csv(args.question_ids)
    granularities = _parse_csv(args.granularities)
    questions = _load_questions(args.questions, question_ids)
    try:
        Retriever = _load_retriever_class()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    hit_rows: list[dict] = []
    rows_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for granularity in granularities:
        index_dir = _resolve_index_dir(
            granularity,
            args.index_base_dir,
            args.chunk_index_dir,
            allow_dedup_fallback=args.allow_dedup_fallback,
        )
        output_granularity = index_dir.name if granularity != "chunk" else "chunk"
        retriever = Retriever(index_dir=index_dir, embedding_model=config.EMBEDDING_MODEL)
        print(f"Inspecting granularity={output_granularity} index={index_dir}")

        for question in questions:
            variants = _build_query_variants(question, args.title)
            for variant, query_used in variants.items():
                retrieve_count = args.source_filter_pool if variant == "source_filtered" else args.top_m
                raw_hits = retriever.retrieve(query_used, top_k=retrieve_count)
                ranked_hits = list(enumerate(raw_hits, start=1))
                if variant == "source_filtered":
                    ranked_hits = [
                        (raw_rank, hit)
                        for raw_rank, hit in ranked_hits
                        if _source_matches(hit, args.target_source)
                    ][: args.top_m]
                else:
                    ranked_hits = ranked_hits[: args.top_m]

                rows = [
                    _make_hit_row(
                        question,
                        output_granularity,
                        variant,
                        query_used,
                        rank,
                        raw_rank,
                        hit,
                        text_limit=args.text_limit,
                    )
                    for rank, (raw_rank, hit) in enumerate(ranked_hits, start=1)
                ]
                rows_by_key[(question["id"], output_granularity, variant)] = rows
                hit_rows.extend(rows)

    summary_rows = []
    for question in questions:
        for granularity in granularities:
            index_dir = _resolve_index_dir(
                granularity,
                args.index_base_dir,
                args.chunk_index_dir,
                allow_dedup_fallback=args.allow_dedup_fallback,
            )
            output_granularity = index_dir.name if granularity != "chunk" else "chunk"
            title_only_rows = rows_by_key.get((question["id"], output_granularity, "title_only"), [])
            for variant in ("original", "title_only", "content_only", "source_filtered"):
                rows = rows_by_key.get((question["id"], output_granularity, variant), [])
                summary_rows.append(
                    _summarize_variant_hits(
                        question["id"],
                        output_granularity,
                        variant,
                        rows,
                        title_only_rows=title_only_rows,
                    )
                )

    hits_path = args.output_dir / f"{args.output_prefix}_hits.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    _write_csv(
        hits_path,
        hit_rows,
        [
            "question_id",
            "granularity",
            "variant",
            "query_used",
            "rank",
            "raw_rank",
            "score",
            "source",
            "page",
            "parent_chunk_id",
            "unit_id",
            "matched_gold_ids",
            "matched_evidence_refs",
            "is_gold_alias_hit",
            "is_evidence_page_hit",
            "is_title_junk",
            "text",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "question_id",
            "granularity",
            "variant",
            "query_used",
            "hit_count",
            "top_score",
            "first_gold_alias_rank",
            "first_evidence_page_rank",
            "gold_alias_hit_count",
            "evidence_page_hit_count",
            "title_junk_count",
            "title_junk_rate",
            "overlap_with_title_only",
            "matched_gold_ids_top_m",
            "matched_evidence_refs_top_m",
        ],
    )
    print(f"Wrote {len(hit_rows)} hit rows to {hits_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
