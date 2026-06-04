"""Run P0 budget sweep experiments for strict strategy comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app import answer_query


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 experiment: budget sweep")
    parser.add_argument("--query", required=True)
    parser.add_argument("--strategies", default="baseline,dynamic,rerank")
    parser.add_argument("--compressions", default="none")
    parser.add_argument("--compression-stages", default="after-allocation")
    parser.add_argument("--budgets", default="500,1000,2000,4000")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--llmlingua-rate", type=float, default=config.LLMLINGUA2_RATE)
    parser.add_argument("--llmlingua-model", default=config.LLMLINGUA2_MODEL)
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional experiment label saved to every generated run",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategies = _parse_csv_strings(args.strategies)
    compressions = _parse_csv_strings(args.compressions)
    compression_stages = _parse_csv_strings(args.compression_stages)
    budgets = _parse_csv_ints(args.budgets)

    for strategy in strategies:
        if strategy not in {"baseline", "dynamic", "rerank"}:
            raise ValueError(f"Unsupported strategy: {strategy}")
        for compression in compressions:
            if compression not in {"none", "truncate", "llmlingua2"}:
                raise ValueError(f"Unsupported compression: {compression}")
            for compression_stage in compression_stages:
                if compression_stage not in {"after-allocation", "before-allocation"}:
                    raise ValueError(f"Unsupported compression stage: {compression_stage}")
                if compression_stage == "before-allocation" and compression != "llmlingua2":
                    continue
                for budget in budgets:
                    print(
                        "\n=== "
                        f"strategy={strategy} compression={compression} "
                        f"stage={compression_stage} "
                        f"top_k={args.top_k} budget={budget} "
                        "==="
                    )
                    answer_query(
                        query=args.query,
                        index_dir=args.index_dir,
                        strategy=strategy,
                        top_k=args.top_k,
                        context_budget=budget,
                        compression=compression,
                        compression_stage=compression_stage,
                        llmlingua_rate=args.llmlingua_rate,
                        llmlingua_model=args.llmlingua_model,
                        run_label=args.run_label,
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    main()
