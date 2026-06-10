"""Evaluate DenseX runs for retrieval relevance and answer accuracy."""

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

from evaluation.list_accuracy import list_accuracy, normalize_text


def _load_questions(path: Path) -> dict[str, dict]:
    questions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        questions[item["id"]] = item
    return questions


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


def _matched_gold_ids(chunks: list[dict], gold_items: list[dict]) -> set[str]:
    matched = set()
    for chunk in chunks:
        haystack = _chunk_text(chunk)
        for item in gold_items:
            if _contains_alias(haystack, item):
                matched.add(item["id"])
    return matched


def _relevance_precision(chunks: list[dict], gold_items: list[dict]) -> float:
    if not chunks:
        return 0.0
    relevant = 0
    for chunk in chunks:
        haystack = _chunk_text(chunk)
        if any(_contains_alias(haystack, item) for item in gold_items):
            relevant += 1
    return relevant / len(chunks)


def _parse_run_label(
    label: str,
    prefix: str,
    question_ids: set[str] | None = None,
) -> tuple[str, str] | tuple[None, None]:
    remainder = (label or "").removeprefix(prefix + "_")
    if remainder == label:
        return None, None

    if question_ids:
        for question_id in sorted(question_ids, key=len, reverse=True):
            suffix = "_" + question_id
            if remainder.endswith(suffix):
                granularity = remainder[: -len(suffix)]
                if granularity:
                    return granularity, question_id
        return None, None

    match = re.match(r"([^_]+)_(.+)", remainder)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DenseX experiment runs")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "experiments" / "runs")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--run-label-prefix", default="densex")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "densex_results.csv")
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    rows = []
    for path in sorted(args.runs_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        label = run.get("run_label", "")
        if not label.startswith(args.run_label_prefix + "_"):
            continue
        granularity, question_id = _parse_run_label(
            label,
            args.run_label_prefix,
            set(questions),
        )
        if not granularity or question_id not in questions:
            continue

        question = questions[question_id]
        gold_items = question.get("gold_items", [])
        candidate_items = question.get("candidate_items", gold_items)
        answer_metrics = list_accuracy(
            run.get("answer", ""),
            gold_items=gold_items,
            candidate_items=candidate_items,
        )
        retrieved_chunks = run.get("retrieved_chunks", [])
        selected_chunks = run.get("selected_chunks", [])
        retrieved_gold = _matched_gold_ids(retrieved_chunks, gold_items)
        selected_gold = _matched_gold_ids(selected_chunks, gold_items)
        gold_count = len(gold_items) or 1
        context_tokens = int(run.get("context_tokens") or 0)
        answer_f1 = float(answer_metrics["f1"])
        rows.append(
            {
                "run_id": run.get("run_id", ""),
                "run_label": label,
                "question_id": question_id,
                "granularity": granularity,
                "strategy": run.get("strategy", ""),
                "budget": run.get("budget", ""),
                "top_k": run.get("top_k", ""),
                "context_tokens": context_tokens,
                "retrieved_chunk_count": len(retrieved_chunks),
                "selected_chunk_count": len(selected_chunks),
                "answer_precision": answer_metrics["precision"],
                "answer_recall": answer_metrics["recall"],
                "answer_f1": answer_f1,
                "retrieved_gold_recall": len(retrieved_gold) / gold_count,
                "selected_gold_recall": len(selected_gold) / gold_count,
                "retrieved_relevance_precision": _relevance_precision(retrieved_chunks, gold_items),
                "selected_relevance_precision": _relevance_precision(selected_chunks, gold_items),
                "token_efficiency": answer_f1 / context_tokens * 1000 if context_tokens else 0.0,
                "matched_gold_ids": ",".join(sorted(answer_metrics["matched_gold_ids"])),
                "retrieved_gold_ids": ",".join(sorted(retrieved_gold)),
                "selected_gold_ids": ",".join(sorted(selected_gold)),
                "details_path": str(path),
            }
        )

    fieldnames = [
        "run_id",
        "run_label",
        "question_id",
        "granularity",
        "strategy",
        "budget",
        "top_k",
        "context_tokens",
        "retrieved_chunk_count",
        "selected_chunk_count",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "retrieved_gold_recall",
        "selected_gold_recall",
        "retrieved_relevance_precision",
        "selected_relevance_precision",
        "token_efficiency",
        "matched_gold_ids",
        "retrieved_gold_ids",
        "selected_gold_ids",
        "details_path",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
