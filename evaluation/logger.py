"""Experiment logging utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

FIELDNAMES = [
    "run_id",
    "run_label",
    "query",
    "strategy",
    "compression",
    "compression_stage",
    "top_k",
    "budget",
    "retrieved_chunk_count",
    "selected_chunk_count",
    "original_context_tokens",
    "context_tokens",
    "compression_ratio",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model",
    "details_path",
]


def log_result(path: Path, row: dict) -> None:
    """Append one experiment row to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
        if existing_fieldnames != FIELDNAMES:
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key, "") for key in FIELDNAMES})

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def create_run_id(prefix: str = "run") -> str:
    """Create a short unique run id."""
    return f"{prefix}_{uuid4().hex[:12]}"


def save_run_details(path: Path, details: dict) -> None:
    """Persist full run details for reproducible analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
