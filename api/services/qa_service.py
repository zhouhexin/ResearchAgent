"""Service wrapper around the existing ResearchAgent QA pipeline."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import config
from api import settings


def _extract_final_answer(raw_answer: str) -> str:
    """Return a display-friendly answer string from the model response."""
    text = raw_answer.strip()
    if not text:
        return ""

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return text
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return text

    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        return parsed["answer"].strip()
    return text


def _find_run_id(run_label: str) -> str | None:
    """Find the run id saved by `answer_query` for this API request."""
    candidates = sorted(
        config.RUNS_DIR.glob(f"{run_label}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        details = json.loads(candidates[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return candidates[0].stem
    return details.get("run_id") or candidates[0].stem


def ask_public_question(query: str) -> dict:
    """Ask a question with stable public defaults and return only final answer."""
    from app import answer_query

    run_label = f"{settings.API_RUN_LABEL_PREFIX}_{uuid.uuid4().hex[:12]}"
    raw_answer = answer_query(
        query=query,
        index_dir=Path(config.INDEX_DIR),
        strategy=settings.API_STRATEGY,
        top_k=settings.API_TOP_K,
        context_budget=settings.API_CONTEXT_BUDGET,
        compression=settings.API_COMPRESSION,
        compression_stage=settings.API_COMPRESSION_STAGE,
        run_label=run_label,
        dry_run=False,
    )
    return {
        "answer": _extract_final_answer(raw_answer),
        "run_id": _find_run_id(run_label),
        "error": None,
    }
