"""Persist user feedback for public QA answers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config


FEEDBACK_PATH = config.BASE_DIR / "experiments" / "web_feedback.jsonl"


def record_feedback(
    *,
    run_id: str | None,
    query: str,
    answer: str,
    rating: str,
    feedback_path: Path = FEEDBACK_PATH,
) -> dict:
    """Append one frontend feedback event to a standalone JSONL file."""
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "query": query,
        "answer": answer,
        "rating": rating,
    }
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
