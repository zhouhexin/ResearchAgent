"""Inspect whether frontend QA runs retrieve stable FAISS top-k chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"


@dataclass(frozen=True)
class RunTopkSummary:
    """Compact top-k information for one saved web QA run."""

    run_id: str
    query: str
    top_k: int | None
    retrieved_ids: list[str]
    selected_ids: list[str]
    retrieved_rows: list[dict]
    overlap_with_first: int
    same_order_as_first: bool
    selected_overlap_with_first: int
    selected_same_order_as_first: bool


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _chunk_id(chunk: dict, index: int) -> str:
    """Return a stable chunk identifier from compact run details."""
    chunk_id = chunk.get("id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id

    source = chunk.get("source", "unknown_source")
    page = chunk.get("page", "unknown_page")
    return f"{source}#page={page}#rank={index + 1}"


def _chunk_rows(chunks: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, chunk in enumerate(chunks):
        rows.append(
            {
                "rank": index + 1,
                "id": _chunk_id(chunk, index),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", ""),
                "score": chunk.get("score", ""),
            }
        )
    return rows


def _same_order(left: list[str], right: list[str]) -> bool:
    return left == right[: len(left)] and len(left) == len(right)


def summarize_runs(
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    query: str | None = None,
    limit: int = 10,
    prefix: str = "web_",
) -> list[RunTopkSummary]:
    """Load recent web run JSON files and summarize retrieved top-k stability."""
    paths = sorted(
        runs_dir.glob(f"{prefix}*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    summaries: list[RunTopkSummary] = []

    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        run_query = str(payload.get("query", ""))
        if query is not None and run_query != query:
            continue

        retrieved_chunks = payload.get("retrieved_chunks") or []
        selected_chunks = payload.get("selected_chunks") or []
        retrieved_rows = _chunk_rows(retrieved_chunks)
        retrieved_ids = [row["id"] for row in retrieved_rows]
        selected_ids = [_chunk_id(chunk, index) for index, chunk in enumerate(selected_chunks)]

        summaries.append(
            RunTopkSummary(
                run_id=str(payload.get("run_id") or path.stem),
                query=run_query,
                top_k=payload.get("top_k"),
                retrieved_ids=retrieved_ids,
                selected_ids=selected_ids,
                retrieved_rows=retrieved_rows,
                overlap_with_first=0,
                same_order_as_first=False,
                selected_overlap_with_first=0,
                selected_same_order_as_first=False,
            )
        )
        if len(summaries) >= limit:
            break

    if not summaries:
        return []

    first_ids = summaries[0].retrieved_ids
    first_id_set = set(first_ids)
    first_selected_ids = summaries[0].selected_ids
    first_selected_id_set = set(first_selected_ids)
    return [
        RunTopkSummary(
            run_id=item.run_id,
            query=item.query,
            top_k=item.top_k,
            retrieved_ids=item.retrieved_ids,
            selected_ids=item.selected_ids,
            retrieved_rows=item.retrieved_rows,
            overlap_with_first=len(set(item.retrieved_ids) & first_id_set),
            same_order_as_first=_same_order(item.retrieved_ids, first_ids),
            selected_overlap_with_first=len(set(item.selected_ids) & first_selected_id_set),
            selected_same_order_as_first=_same_order(item.selected_ids, first_selected_ids),
        )
        for item in summaries
    ]


def format_report(summaries: list[RunTopkSummary], *, show_chunks: bool) -> str:
    """Format summaries as a plain text report for terminal inspection."""
    if not summaries:
        return "No matching web run files found."

    lines = [
        f"Matched runs: {len(summaries)}",
        f"Reference run: {summaries[0].run_id}",
        "",
        (
            "run_id\ttop_k\tretrieved_count\tselected_count\t"
            "overlap_with_first\tsame_order_as_first\t"
            "selected_overlap_with_first\tselected_same_order_as_first"
        ),
    ]
    for item in summaries:
        lines.append(
            "\t".join(
                [
                    item.run_id,
                    str(item.top_k),
                    str(len(item.retrieved_ids)),
                    str(len(item.selected_ids)),
                    str(item.overlap_with_first),
                    "yes" if item.same_order_as_first else "no",
                    str(item.selected_overlap_with_first),
                    "yes" if item.selected_same_order_as_first else "no",
                ]
            )
        )

    if show_chunks:
        for item in summaries:
            lines.extend(["", f"[{item.run_id}] retrieved_chunks"])
            lines.append("rank\tscore\tpage\tsource\tid")
            for row in item.retrieved_rows:
                lines.append(
                    "\t".join(
                        [
                            str(row["rank"]),
                            str(row["score"]),
                            str(row["page"]),
                            str(row["source"]),
                            str(row["id"]),
                        ]
                    )
                )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect saved frontend QA runs to compare retrieved FAISS top-k stability.",
    )
    parser.add_argument("--query", help="Only inspect runs whose query exactly matches this text.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of recent runs.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing saved run JSON files.",
    )
    parser.add_argument("--prefix", default="web_", help="Run filename prefix to inspect.")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved chunk rank, score, source, page, and id for each run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = summarize_runs(
        runs_dir=args.runs_dir,
        query=args.query,
        limit=args.limit,
        prefix=args.prefix,
    )
    print(format_report(summaries, show_chunks=args.show_chunks))


if __name__ == "__main__":
    main()
