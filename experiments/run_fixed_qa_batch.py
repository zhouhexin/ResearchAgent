"""Run and summarize fixed QA DenseX comparison batches."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ACDEPTH_QA_IDS = [
    "always_clear_depth_contributions",
    "always_clear_depth_eval_datasets",
    "always_clear_depth_ablation_components",
    "always_clear_depth_sota_comparison_methods",
]

DEPTHDARK_QA_IDS = [
    "depthdark_contributions",
    "depthdark_eval_datasets",
    "depthdark_training_datasets",
    "depthdark_ablation_components",
    "depthdark_sota_comparison_methods",
]


def _csv(items: list[str]) -> str:
    return ",".join(items)


def _question_ids(args: argparse.Namespace) -> str:
    if args.question_ids:
        return args.question_ids
    if args.question_set == "acdepth":
        return _csv(ACDEPTH_QA_IDS)
    if args.question_set == "depthdark":
        return _csv(DEPTHDARK_QA_IDS)
    return _csv(ACDEPTH_QA_IDS + DEPTHDARK_QA_IDS)


def _run(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_summary(results_path: Path, summary_path: Path) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with results_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            groups[(row.get("granularity", ""), row.get("budget", ""))].append(row)

    fieldnames = [
        "granularity",
        "budget",
        "run_count",
        "avg_context_tokens",
        "avg_answer_f1",
        "avg_answer_recall",
        "avg_selected_gold_recall",
        "avg_selected_relevance_precision",
        "avg_token_efficiency",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for (granularity, budget), rows in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1] or 0))):
            count = len(rows) or 1
            writer.writerow(
                {
                    "granularity": granularity,
                    "budget": budget,
                    "run_count": len(rows),
                    "avg_context_tokens": sum(_to_float(row.get("context_tokens", "")) for row in rows) / count,
                    "avg_answer_f1": sum(_to_float(row.get("answer_f1", "")) for row in rows) / count,
                    "avg_answer_recall": sum(_to_float(row.get("answer_recall", "")) for row in rows) / count,
                    "avg_selected_gold_recall": sum(_to_float(row.get("selected_gold_recall", "")) for row in rows) / count,
                    "avg_selected_relevance_precision": sum(
                        _to_float(row.get("selected_relevance_precision", "")) for row in rows
                    )
                    / count,
                    "avg_token_efficiency": sum(_to_float(row.get("token_efficiency", "")) for row in rows)
                    / count,
                }
            )
    print(f"Wrote summary to {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed QA DenseX comparison batch")
    parser.add_argument("--question-set", choices=["fixed", "acdepth", "depthdark"], default="fixed")
    parser.add_argument("--question-ids", default="", help="Override question IDs as a comma-separated list")
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--budgets", default="500,1000,1500")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", choices=["baseline", "dynamic", "rerank"], default="baseline")
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--prepare-corpora", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--proposition-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--results-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--skip-run", action="store_true", help="Only evaluate existing runs with the prefix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    run_prefix = args.run_prefix or f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    question_ids = _question_ids(args)
    results_output = args.results_output or PROJECT_ROOT / "experiments" / f"{run_prefix}_densex_results.csv"
    summary_output = args.summary_output or PROJECT_ROOT / "experiments" / f"{run_prefix}_densex_summary.csv"

    if args.prepare_corpora:
        prepare_command = [
            sys.executable,
            "experiments/densex_prepare_corpus.py",
            "--granularities",
            args.granularities,
            "--metadata",
            "storage/metadata.json",
            "--device",
            args.proposition_device,
            "--resume",
        ]
        _run(prepare_command, dry_run=args.dry_run)

    if args.build_index:
        build_command = [
            sys.executable,
            "experiments/densex_build_index.py",
            "--granularities",
            args.granularities,
        ]
        _run(build_command, dry_run=args.dry_run)

    if not args.skip_run:
        run_command = [
            sys.executable,
            "experiments/run_densex_sweep.py",
            "--question-ids",
            question_ids,
            "--granularities",
            args.granularities,
            "--budgets",
            args.budgets,
            "--top-k",
            str(args.top_k),
            "--strategy",
            args.strategy,
            "--run-label-prefix",
            run_prefix,
        ]
        _run(run_command, dry_run=args.dry_run)

    evaluate_command = [
        sys.executable,
        "experiments/evaluate_densex_runs.py",
        "--run-label-prefix",
        run_prefix,
        "--output",
        str(results_output),
    ]
    _run(evaluate_command, dry_run=args.dry_run)

    if not args.dry_run:
        _write_summary(results_output, summary_output)
    print(f"Run prefix: {run_prefix}", flush=True)


if __name__ == "__main__":
    main()
