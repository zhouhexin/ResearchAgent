"""List-answer accuracy metrics for reproducible QA evaluation.

The current project studies questions such as "which papers use diffusion?".
Those answers are naturally evaluated as sets: a run is good when it mentions
the expected items and does not invent many extra items. This module therefore
implements deterministic precision / recall / F1 over manually curated gold
items instead of using an LLM judge.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize text for robust title/item matching.

    Matching is intentionally simple and auditable: lower-case the text, replace
    punctuation with spaces, and collapse whitespace. This avoids hidden model
    behavior in the accuracy metric while still tolerating small formatting
    differences such as hyphenation or line breaks.
    """
    lowered = text.lower()
    without_punct = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", without_punct).strip()


def _contains_phrase(normalized_answer: str, phrase: str) -> bool:
    """Return whether a normalized phrase appears as a loose token phrase."""
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_answer} "


def find_matched_items(answer: str, gold_items: list[dict]) -> list[str]:
    """Find gold item ids whose name or aliases appear in the answer."""
    normalized_answer = normalize_text(answer)
    matched: list[str] = []
    for item in gold_items:
        aliases = [item.get("name", ""), *item.get("aliases", [])]
        if any(_contains_phrase(normalized_answer, alias) for alias in aliases):
            matched.append(item["id"])
    return matched


def find_predicted_items(answer: str, candidate_items: list[dict]) -> list[str]:
    """Find all known candidate item ids mentioned by an answer.

    Candidate items represent the closed-world label space for a dataset, such
    as the set of papers in `data/`. Any mentioned candidate that is not in the
    gold set becomes a false positive. If no candidates are provided, precision
    falls back to the gold matches only and false positives cannot be counted.
    """
    return find_matched_items(answer, candidate_items)


def list_accuracy(
    answer: str,
    *,
    gold_items: list[dict],
    candidate_items: list[dict] | None = None,
) -> dict:
    """Compute deterministic list-answer precision, recall, and F1.

    `gold_items` and `candidate_items` use dictionaries with at least:

    - id: stable item id
    - name: canonical display name
    - aliases: optional alternative strings to match

    When `candidate_items` is omitted, the evaluator cannot detect extra items
    outside the gold list, so precision is optimistic. For paper-list questions,
    provide all known paper titles as candidates whenever possible.
    """
    gold_ids = {item["id"] for item in gold_items}
    candidates = candidate_items if candidate_items is not None else gold_items
    predicted_ids = set(find_predicted_items(answer, candidates))
    matched_gold_ids = predicted_ids & gold_ids

    true_positive = len(matched_gold_ids)
    false_positive = len(predicted_ids - gold_ids)
    false_negative = len(gold_ids - predicted_ids)

    precision = true_positive / (true_positive + false_positive) if predicted_ids else 0.0
    recall = true_positive / len(gold_ids) if gold_ids else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "matched_gold_ids": sorted(matched_gold_ids),
        "predicted_ids": sorted(predicted_ids),
        "missing_gold_ids": sorted(gold_ids - predicted_ids),
        "extra_predicted_ids": sorted(predicted_ids - gold_ids),
    }
