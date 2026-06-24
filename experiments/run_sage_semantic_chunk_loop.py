"""Run SAGE semantic chunk QA experiments with retry and resume support."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from densex.corpus import append_jsonl


@dataclass(frozen=True)
class Job:
    """One deterministic QA experiment job."""

    question: dict
    granularity: str
    index_dir: Path
    budget: int
    run_label: str


@dataclass(frozen=True)
class JobResult:
    """Execution result for one job."""

    ok: bool
    attempts: int
    error: str = ""


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_questions(path: Path, ids: set[str] | None) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if ids is None or item.get("id") in ids:
            questions.append(item)
    if ids is not None:
        missing = ids - {item["id"] for item in questions}
        if missing:
            raise ValueError(f"Question ids not found: {sorted(missing)}")
    if not questions:
        raise ValueError(f"No questions selected from {path}")
    return questions


def _load_answer_query(embedding_model: str):
    config.EMBEDDING_MODEL = embedding_model
    from app import answer_query

    return answer_query


def completed_run_labels(runs_dir: Path, *, require_answer: bool = True) -> set[str]:
    """Return run labels that already have successful run detail JSON files."""
    labels = set()
    if not runs_dir.exists():
        return labels
    for path in runs_dir.glob("*.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label = str(run.get("run_label") or "")
        if not label:
            continue
        if require_answer and not str(run.get("answer") or "").strip():
            continue
        labels.add(label)
    return labels


def is_retryable_exception(exc: BaseException) -> bool:
    """Identify transient API/network failures worth retrying."""
    message = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}".lower()
    retryable_markers = [
        "apiconnectionerror",
        "connecterror",
        "connection error",
        "timeout",
        "timed out",
        "unexpected_eof_while_reading",
        "ssl:",
        "rate limit",
        "ratelimit",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in retryable_markers)


def build_jobs(
    *,
    questions: list[dict],
    budgets: list[int],
    index_dirs: dict[str, Path],
    run_label_prefix: str,
) -> list[Job]:
    """Build deterministic jobs in the same order as the non-resumable SAGE runner."""
    jobs = []
    for question in questions:
        for granularity, index_dir in index_dirs.items():
            for budget in budgets:
                jobs.append(
                    Job(
                        question=question,
                        granularity=granularity,
                        index_dir=index_dir,
                        budget=budget,
                        run_label=f"{run_label_prefix}_{granularity}_{question['id']}",
                    )
                )
    return jobs


def run_job_with_retries(
    job: Job,
    *,
    answer_query: Callable[..., str],
    strategy: str,
    top_k: int,
    max_retries: int,
    retry_wait_seconds: float,
    retry_backoff: float,
    dry_run: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> JobResult:
    """Run one job, retrying transient API/network failures."""
    attempts = 0
    wait_seconds = retry_wait_seconds
    while attempts < max(1, max_retries):
        attempts += 1
        try:
            answer_query(
                query=job.question["query"],
                index_dir=job.index_dir,
                strategy=strategy,
                top_k=top_k,
                context_budget=job.budget,
                compression="none",
                compression_stage="after-allocation",
                run_label=job.run_label,
                dry_run=dry_run,
            )
            return JobResult(ok=True, attempts=attempts)
        except Exception as exc:  # noqa: BLE001 - batch runner must isolate failed jobs.
            error = f"{exc.__class__.__name__}: {exc}"
            if attempts >= max(1, max_retries) or not is_retryable_exception(exc):
                return JobResult(ok=False, attempts=attempts, error=error)
            sleep(wait_seconds)
            wait_seconds *= retry_backoff
    return JobResult(ok=False, attempts=attempts, error="retry loop exited unexpectedly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE semantic chunk sweep with retry/resume")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument(
        "--question-ids",
        default=(
            "always_clear_depth_contributions,"
            "always_clear_depth_eval_datasets,"
            "always_clear_depth_ablation_components,"
            "always_clear_depth_sota_comparison_methods"
        ),
    )
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--chunk-index-dir", type=Path, default=PROJECT_ROOT / "storage" / "sage" / "chunk")
    parser.add_argument(
        "--semantic-index-dir",
        type=Path,
        default=PROJECT_ROOT / "storage" / "sage" / "semantic_chunk",
    )
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--budgets", default="300,500,1000,1500")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "dynamic", "rerank"])
    parser.add_argument("--run-label-prefix", default="sage_semantic_v1")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "sage_failed_jobs.jsonl",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=20.0)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip run labels that already have answers")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even when some jobs fail")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    question_ids = None if args.all_questions else set(_parse_csv(args.question_ids))
    questions = _load_questions(args.questions, question_ids)
    budgets = _parse_csv_ints(args.budgets)
    index_dirs = {
        "chunk": args.chunk_index_dir,
        "semantic_chunk": args.semantic_index_dir,
    }
    jobs = build_jobs(
        questions=questions,
        budgets=budgets,
        index_dirs=index_dirs,
        run_label_prefix=args.run_label_prefix,
    )
    completed = set() if args.no_resume else completed_run_labels(args.runs_dir, require_answer=not args.dry_run)
    answer_query = _load_answer_query(args.embedding_model)

    skipped = 0
    succeeded = 0
    failed = 0
    for job_index, job in enumerate(jobs, start=1):
        if job.run_label in completed:
            skipped += 1
            print(f"[{job_index}/{len(jobs)}] SKIP existing {job.run_label}", flush=True)
            continue

        print(
            f"[{job_index}/{len(jobs)}] RUN qid={job.question['id']} "
            f"granularity={job.granularity} budget={job.budget}",
            flush=True,
        )
        result = run_job_with_retries(
            job,
            answer_query=answer_query,
            strategy=args.strategy,
            top_k=args.top_k,
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
            retry_backoff=args.retry_backoff,
            dry_run=args.dry_run,
        )
        if result.ok:
            succeeded += 1
            print(f"[{job_index}/{len(jobs)}] OK attempts={result.attempts}", flush=True)
            continue

        failed += 1
        failure = {
            "run_label": job.run_label,
            "question_id": job.question.get("id"),
            "granularity": job.granularity,
            "budget": job.budget,
            "attempts": result.attempts,
            "error": result.error,
        }
        append_jsonl(args.failure_log, failure)
        print(f"[{job_index}/{len(jobs)}] FAIL attempts={result.attempts}: {result.error}", flush=True)

    print(
        f"Finished jobs={len(jobs)} succeeded={succeeded} skipped={skipped} failed={failed} "
        f"failure_log={args.failure_log}",
        flush=True,
    )
    if failed and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
