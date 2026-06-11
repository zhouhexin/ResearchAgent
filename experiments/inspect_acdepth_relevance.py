"""Inspect ACDepth retrieval relevance scores across retrieval granularities.

This script is diagnostic only: it does not call the LLM and does not write run
JSON files. It retrieves top-M units for ACDepth questions and annotates each
hit with two lightweight signals:

- matched_gold_ids: gold answer items whose aliases appear in the hit text.
- matched_evidence_refs: gold evidence source/page references matched by the hit.

Use these fields to decide whether ACDepth failures come from retrieval,
fine-to-chunk aggregation, context selection, or answer generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.list_accuracy import normalize_text


ACDEPTH_QA_IDS = [
    "always_clear_depth_contributions",
    "always_clear_depth_eval_datasets",
    "always_clear_depth_ablation_components",
    "always_clear_depth_sota_comparison_methods",
]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _contains_alias(text: str, item: dict) -> bool:
    normalized = normalize_text(text)
    aliases = [item.get("name", ""), *item.get("aliases", [])]
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if normalized_alias and f" {normalized_alias} " in f" {normalized} ":
            return True
    return False


def _hit_text(hit: dict) -> str:
    return " ".join(str(hit.get(key, "") or "") for key in ("id", "source", "paper_title", "text"))


def _matched_gold_ids(hit: dict, question: dict) -> list[str]:
    text = _hit_text(hit)
    return [
        item["id"]
        for item in question.get("gold_items", [])
        if _contains_alias(text, item)
    ]


def _source_basename(value: str) -> str:
    return Path(value or "").name


def _pages_equal(left: object, right: object) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left or "") == str(right or "")


def _format_evidence_ref(evidence: dict) -> str:
    source = _source_basename(str(evidence.get("source", "")))
    page = evidence.get("page", "")
    evidence_for = evidence.get("evidence_for", "")
    return f"{source}:p{page}:{evidence_for}"


def _matched_evidence_refs(hit: dict, question: dict) -> list[str]:
    hit_source = _source_basename(str(hit.get("source", "")))
    hit_page = hit.get("page")
    refs = []
    for evidence in question.get("gold_evidence", []):
        evidence_source = _source_basename(str(evidence.get("source", "")))
        if hit_source == evidence_source and _pages_equal(hit_page, evidence.get("page")):
            refs.append(_format_evidence_ref(evidence))
    return refs


def _resolve_index_dir(
    granularity: str,
    index_base_dir: Path,
    chunk_index_dir: Path,
    *,
    allow_dedup_fallback: bool,
    exists: Callable[[Path], bool] | None = None,
) -> Path:
    path_exists = exists or Path.exists
    if granularity == "chunk":
        return chunk_index_dir

    index_dir = index_base_dir / granularity
    if path_exists(index_dir):
        return index_dir

    dedup_dir = index_base_dir / f"{granularity}_dedup"
    if allow_dedup_fallback and path_exists(dedup_dir):
        return dedup_dir

    return index_dir


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


def _truncate_text(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _make_hit_row(question: dict, granularity: str, rank: int, hit: dict, *, text_limit: int) -> dict:
    matched_gold_ids = _matched_gold_ids(hit, question)
    matched_evidence_refs = _matched_evidence_refs(hit, question)
    return {
        "question_id": question["id"],
        "query": question.get("query", ""),
        "granularity": granularity,
        "rank": rank,
        "score": hit.get("score", ""),
        "source": hit.get("source", ""),
        "page": hit.get("page", ""),
        "parent_chunk_id": hit.get("parent_chunk_id", hit.get("id", "")),
        "unit_id": hit.get("id", ""),
        "matched_gold_ids": ",".join(matched_gold_ids),
        "matched_evidence_refs": "|".join(matched_evidence_refs),
        "is_gold_alias_hit": bool(matched_gold_ids),
        "is_evidence_page_hit": bool(matched_evidence_refs),
        "text": _truncate_text(str(hit.get("text", "")), text_limit),
    }


def _summarize_hits(question_id: str, granularity: str, hit_rows: list[dict]) -> dict:
    gold_rows = [row for row in hit_rows if row.get("matched_gold_ids")]
    evidence_rows = [row for row in hit_rows if row.get("matched_evidence_refs")]
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
        "hit_count": len(hit_rows),
        "top_score": hit_rows[0].get("score", "") if hit_rows else "",
        "first_gold_alias_rank": gold_rows[0]["rank"] if gold_rows else "",
        "first_evidence_page_rank": evidence_rows[0]["rank"] if evidence_rows else "",
        "gold_alias_hit_count": len(gold_rows),
        "evidence_page_hit_count": len(evidence_rows),
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
    parser = argparse.ArgumentParser(description="Inspect ACDepth retrieval scores")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--question-ids", default=",".join(ACDEPTH_QA_IDS))
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--top-m", type=int, default=300)
    parser.add_argument("--index-base-dir", type=Path, default=PROJECT_ROOT / "storage" / "densex")
    parser.add_argument("--chunk-index-dir", type=Path, default=PROJECT_ROOT / "storage")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "diagnostics")
    parser.add_argument("--output-prefix", default="acdepth_relevance")
    parser.add_argument(
        "--allow-dedup-fallback",
        action="store_true",
        help="Use sentence_dedup/proposition_dedup if sentence/proposition indexes are missing",
    )
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
    summary_rows: list[dict] = []
    for granularity in granularities:
        index_dir = _resolve_index_dir(
            granularity,
            args.index_base_dir,
            args.chunk_index_dir,
            allow_dedup_fallback=args.allow_dedup_fallback,
        )
        retriever = Retriever(index_dir=index_dir, embedding_model=config.EMBEDDING_MODEL)

        output_granularity = index_dir.name if granularity != "chunk" else "chunk"
        print(f"Inspecting granularity={output_granularity} index={index_dir}")
        for question in questions:
            hits = retriever.retrieve(question["query"], top_k=args.top_m)
            rows = [
                _make_hit_row(question, output_granularity, rank, hit, text_limit=args.text_limit)
                for rank, hit in enumerate(hits, start=1)
            ]
            hit_rows.extend(rows)
            summary_rows.append(_summarize_hits(question["id"], output_granularity, rows))

    hits_path = args.output_dir / f"{args.output_prefix}_hits.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    _write_csv(
        hits_path,
        hit_rows,
        [
            "question_id",
            "query",
            "granularity",
            "rank",
            "score",
            "source",
            "page",
            "parent_chunk_id",
            "unit_id",
            "matched_gold_ids",
            "matched_evidence_refs",
            "is_gold_alias_hit",
            "is_evidence_page_hit",
            "text",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "question_id",
            "granularity",
            "hit_count",
            "top_score",
            "first_gold_alias_rank",
            "first_evidence_page_rank",
            "gold_alias_hit_count",
            "evidence_page_hit_count",
            "matched_gold_ids_top_m",
            "matched_evidence_refs_top_m",
        ],
    )
    print(f"Wrote {len(hit_rows)} hit rows to {hits_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
