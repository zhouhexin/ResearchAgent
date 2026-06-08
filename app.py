"""Command line entry point for ResearchAgent."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import config
from allocation.baseline import allocate_baseline
from allocation.dynamic_budget import allocate_dynamic
from allocation.rerank_budget import allocate_rerank
from chunking.chunker import load_documents
from compression.llmlingua2 import compress_chunks_llmlingua2, summarize_llmlingua2_chunks
from compression.summarize import compress_chunks
from evaluation.logger import create_run_id, log_result, save_run_details
from evaluation.token_counter import count_context_tokens, count_tokens
from llm.minimax_client import MiniMaxClient
from prompt.builder import build_prompt
from retrieval.embed import Embedder
from retrieval.faiss_store import FaissStore
from retrieval.retriever import Retriever


def _safe_label(value: str) -> str:
    """Convert a user-provided run label into a filesystem-friendly id segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_")[:48]


def _allocate_contexts(
    query: str,
    chunks: list[dict],
    *,
    strategy: str,
    context_budget: int,
) -> list[dict]:
    """Run one allocation strategy over the provided candidate chunks."""
    if strategy == "baseline":
        return allocate_baseline(chunks, budget=context_budget)
    if strategy == "dynamic":
        return allocate_dynamic(chunks, budget=context_budget)
    if strategy == "rerank":
        return allocate_rerank(query, chunks, budget=context_budget)
    raise ValueError(f"Unknown allocation strategy: {strategy}")


def _original_context_tokens(contexts: list[dict]) -> int:
    """Count original tokens for selected chunks.

    Compressed chunks store `original_estimated_tokens`; uncompressed historical
    chunks do not. This helper keeps token accounting consistent across no
    compression, after-allocation compression, and before-allocation compression.
    """
    total = 0
    for chunk in contexts:
        if "original_estimated_tokens" in chunk:
            total += int(chunk.get("original_estimated_tokens") or 0)
        else:
            total += count_tokens(chunk.get("text", ""))
    return total


def _default_compression_info(
    *,
    method: str,
    stage: str,
    original_context_tokens: int,
    compressed_context_tokens: int,
    extra: dict | None = None,
) -> dict:
    """Build the common compression metadata block saved in every run detail."""
    info = {
        "method": method,
        "stage": stage,
        "original_context_tokens": original_context_tokens,
        "compressed_context_tokens": compressed_context_tokens,
        "compression_ratio": (
            compressed_context_tokens / original_context_tokens
            if original_context_tokens
            else 0.0
        ),
    }
    if extra:
        info.update(extra)
    return info


def _compact_chunk(chunk: dict) -> dict:
    """Keep only fields needed for answer/evidence evaluation.

    Raw chunks can contain offsets, original compressed text, per-chunk
    LLMLingua metadata, and scoring internals. Those are useful while debugging
    one failure, but they make every experiment JSON unnecessarily large. The
    compact form preserves source identity and the exact text that can be used
    as evidence.
    """
    compact = {
        "id": chunk.get("id"),
        "source": chunk.get("source"),
        "page": chunk.get("page"),
        "score": chunk.get("score"),
        "text": chunk.get("text", ""),
    }
    for key in ("estimated_tokens", "original_estimated_tokens"):
        if key in chunk:
            compact[key] = chunk.get(key)
    compression = chunk.get("compression")
    if isinstance(compression, dict) and compression:
        compact["compression_ratio"] = compression.get("compression_ratio")
    fine_to_chunk = chunk.get("fine_to_chunk")
    if isinstance(fine_to_chunk, dict) and fine_to_chunk:
        compact["fine_to_chunk"] = fine_to_chunk
    return {key: value for key, value in compact.items() if value is not None}


def _compact_chunks(chunks: list[dict]) -> list[dict]:
    """Compact a list of retrieved or selected chunks for run details."""
    return [_compact_chunk(chunk) for chunk in chunks]


def _compact_compression_info(info: dict) -> dict:
    """Drop verbose per-chunk compression internals from run details."""
    keys = [
        "method",
        "stage",
        "model_name",
        "rate",
        "original_context_tokens",
        "compressed_context_tokens",
        "compression_ratio",
        "candidate_pool",
    ]
    return {key: info.get(key) for key in keys if key in info and info.get(key) is not None}


def _run_details(
    *,
    run_id: str,
    run_label: str,
    dry_run: bool,
    query: str,
    strategy: str,
    compression: str,
    compression_stage: str,
    top_k: int,
    budget: int,
    retrieved_chunks: list[dict],
    selected_chunks: list[dict],
    compression_info: dict,
    original_context_tokens: int,
    context_tokens: int,
    prompt_token_estimate: int,
    answer: str | None = None,
    usage: dict | None = None,
) -> dict:
    """Build the compact run-details JSON saved for each experiment."""
    details = {
        "run_id": run_id,
        "dry_run": dry_run,
        "query": query,
        "strategy": strategy,
        "compression": compression,
        "compression_stage": compression_stage,
        "top_k": top_k,
        "budget": budget,
        "retrieved_chunks": _compact_chunks(retrieved_chunks),
        "selected_chunks": _compact_chunks(selected_chunks),
        "compression_info": _compact_compression_info(compression_info),
        "original_context_tokens": original_context_tokens,
        "context_tokens": context_tokens,
        "prompt_token_estimate": prompt_token_estimate,
    }
    if run_label:
        details["run_label"] = run_label
    if answer is not None:
        details["answer"] = answer
    if usage:
        details["usage"] = usage
    return details


def build_index(docs_dir: Path, index_dir: Path) -> None:
    """Build and persist a FAISS index from local text documents."""
    chunks = load_documents(
        docs_dir=docs_dir,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        extensions=config.SUPPORTED_EXTENSIONS,
    )
    if not chunks:
        raise RuntimeError(f"No supported documents found in {docs_dir}")

    embedder = Embedder(config.EMBEDDING_MODEL)
    print(f"Embedding Model: {config.EMBEDDING_MODEL}")
    embeddings = embedder.encode([chunk["text"] for chunk in chunks])
    store = FaissStore(index_dir)
    store.build(chunks, embeddings)
    store.save()
    print(f"Indexed {len(chunks)} chunks from {docs_dir}")


def answer_query(
    query: str,
    index_dir: Path,
    strategy: str,
    top_k: int,
    context_budget: int,
    compression: str = "none",
    compression_stage: str = "after-allocation",
    llmlingua_rate: float = config.LLMLINGUA2_RATE,
    llmlingua_model: str = config.LLMLINGUA2_MODEL,
    run_label: str = "",
    dry_run: bool = False,
    retrieved_chunks_override: list[dict] | None = None,
) -> str:
    """Retrieve context and answer a user query."""
    if retrieved_chunks_override is None:
        retriever = Retriever(index_dir=index_dir, embedding_model=config.EMBEDDING_MODEL)
        chunks = retriever.retrieve(query, top_k=top_k)
    else:
        chunks = retrieved_chunks_override[:top_k]

    if compression not in {"none", "truncate", "llmlingua2"}:
        raise ValueError(f"Unknown compression method: {compression}")
    if compression_stage not in {"after-allocation", "before-allocation"}:
        raise ValueError(f"Unknown compression stage: {compression_stage}")
    if compression_stage == "before-allocation" and compression != "llmlingua2":
        raise ValueError("before-allocation compression currently supports only llmlingua2")

    # The default and historical path: allocation sees original chunks, then the
    # selected chunks may be compressed. This measures token savings under the
    # same selected evidence set, but it cannot increase selected_chunk_count.
    if compression_stage == "after-allocation" or compression == "none":
        contexts = _allocate_contexts(
            query,
            chunks,
            strategy=strategy,
            context_budget=context_budget,
        )
        original_context_tokens = _original_context_tokens(contexts)
        compression_info = _default_compression_info(
            method="none",
            stage="none",
            original_context_tokens=original_context_tokens,
            compressed_context_tokens=original_context_tokens,
        )

        if compression == "truncate":
            contexts = compress_chunks(contexts, max_chars_per_chunk=1000)
            context_tokens_after_compression = count_context_tokens(contexts)
            compression_info = _default_compression_info(
                method="truncate",
                stage="after-allocation",
                original_context_tokens=original_context_tokens,
                compressed_context_tokens=context_tokens_after_compression,
                extra={"max_chars_per_chunk": 1000},
            )
        elif compression == "llmlingua2":
            contexts, compression_info = compress_chunks_llmlingua2(
                contexts,
                model_name=llmlingua_model,
                rate=llmlingua_rate,
                stage="after-allocation",
            )

    # The experimental path requested for LLMLingua-2: compress the retrieved
    # candidate pool first, then allocation uses compressed token counts. This is
    # the version that can fit more chunks into the same final context budget.
    else:
        compressed_chunks, candidate_compression_info = compress_chunks_llmlingua2(
            chunks,
            model_name=llmlingua_model,
            rate=llmlingua_rate,
            stage="before-allocation-candidates",
        )
        contexts = _allocate_contexts(
            query,
            compressed_chunks,
            strategy=strategy,
            context_budget=context_budget,
        )
        compression_info = summarize_llmlingua2_chunks(
            contexts,
            model_name=llmlingua_model,
            rate=llmlingua_rate,
            stage="before-allocation",
        )
        compression_info["candidate_pool"] = {
            "retrieved_chunk_count": len(chunks),
            "compressed_candidate_count": len(compressed_chunks),
            "original_candidate_tokens": candidate_compression_info.get(
                "original_context_tokens"
            ),
            "compressed_candidate_tokens": candidate_compression_info.get(
                "compressed_context_tokens"
            ),
            "candidate_compression_ratio": candidate_compression_info.get(
                "compression_ratio"
            ),
        }

    original_context_tokens = compression_info.get(
        "original_context_tokens",
        _original_context_tokens(contexts),
    )

    prompt = build_prompt(query, contexts)
    context_tokens = count_context_tokens(contexts)
    prompt_token_estimate = count_tokens(prompt)
    run_stage = compression_info.get("stage", compression_stage)
    run_prefix = f"ask_{strategy}_{compression}"
    if compression != "none":
        run_prefix = f"{run_prefix}_{run_stage}"
    safe_run_label = _safe_label(run_label)
    if safe_run_label:
        run_prefix = f"{safe_run_label}_{run_prefix}"
    run_id = create_run_id(run_prefix)
    details_path = config.RUNS_DIR / f"{run_id}.json"

    if dry_run:
        print(prompt)
        save_run_details(
            details_path,
            _run_details(
                run_id=run_id,
                run_label=run_label,
                dry_run=True,
                query=query,
                strategy=strategy,
                compression=compression,
                compression_stage=compression_info.get("stage", compression_stage),
                top_k=top_k,
                budget=context_budget,
                retrieved_chunks=chunks,
                selected_chunks=contexts,
                compression_info=compression_info,
                original_context_tokens=original_context_tokens,
                context_tokens=context_tokens,
                prompt_token_estimate=prompt_token_estimate,
            ),
        )
        print(f"\nRun details saved to {details_path}")
        return prompt

    result = MiniMaxClient().chat_with_usage(prompt)
    answer = result.content
    print(answer)

    log_result(
        config.RESULTS_PATH,
        {
            "run_id": run_id,
            "run_label": run_label,
            "query": query,
            "strategy": strategy,
            "compression": compression,
            "compression_stage": compression_info.get("stage", compression_stage),
            "top_k": top_k,
            "budget": context_budget,
            "retrieved_chunk_count": len(chunks),
            "selected_chunk_count": len(contexts),
            "original_context_tokens": original_context_tokens,
            "context_tokens": context_tokens,
            "compression_ratio": compression_info.get("compression_ratio"),
            "prompt_tokens": result.prompt_tokens or prompt_token_estimate,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "model": result.model,
            "details_path": str(details_path),
        },
    )
    save_run_details(
        details_path,
        _run_details(
            run_id=run_id,
            run_label=run_label,
            dry_run=False,
            query=query,
            strategy=strategy,
            compression=compression,
            compression_stage=compression_info.get("stage", compression_stage),
            top_k=top_k,
            budget=context_budget,
            retrieved_chunks=chunks,
            selected_chunks=contexts,
            compression_info=compression_info,
            original_context_tokens=original_context_tokens,
            context_tokens=context_tokens,
            prompt_token_estimate=prompt_token_estimate,
            answer=answer,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "model": result.model,
            },
        ),
    )
    print(f"\nRun details saved to {details_path}")
    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResearchAgent MVP")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build a local document index")
    index_parser.add_argument("--docs", type=Path, default=config.DATA_DIR)
    index_parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index")
    ask_parser.add_argument("--query",default="有哪些使用了diffusion的论文")
    ask_parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    ask_parser.add_argument(
        "--strategy",
        choices=["baseline", "dynamic", "rerank"],
        default="dynamic",
    )
    ask_parser.add_argument(
        "--compression",
        choices=["none", "truncate", "llmlingua2"],
        default="none",
        help="Compress selected contexts after allocation before building the prompt",
    )
    ask_parser.add_argument(
        "--compression-stage",
        choices=["after-allocation", "before-allocation"],
        default="after-allocation",
        help="Apply compression after selected chunks or before allocation over candidates",
    )
    ask_parser.add_argument(
        "--llmlingua-rate",
        type=float,
        default=config.LLMLINGUA2_RATE,
        help="LLMLingua-2 target compression rate when --compression llmlingua2",
    )
    ask_parser.add_argument(
        "--llmlingua-model",
        default=config.LLMLINGUA2_MODEL,
        help="Hugging Face model name for LLMLingua-2",
    )
    ask_parser.add_argument("--top-k", type=int, default=config.TOP_K)
    ask_parser.add_argument("--context-budget", type=int, default=config.CONTEXT_BUDGET)
    ask_parser.add_argument(
        "--run-label",
        default="",
        help="Optional experiment label saved to CSV/run details and prefixed to run_id",
    )
    ask_parser.add_argument("--dry-run", action="store_true", help="Print prompt without LLM call")
    args = parser.parse_args()
    if args.command is None:
        args.command = "ask"
        args.query = "有哪些使用了diffusion的论文"
        args.index_dir = config.INDEX_DIR
        args.strategy = "dynamic"
        args.compression = "none"
        args.compression_stage = "after-allocation"
        args.llmlingua_rate = config.LLMLINGUA2_RATE
        args.llmlingua_model = config.LLMLINGUA2_MODEL
        args.run_label = ""
        args.top_k = config.TOP_K
        args.context_budget = config.CONTEXT_BUDGET
        args.dry_run = False
    return args


def main() -> None:
    args = parse_args()
    if args.command == "index":
        build_index(args.docs, args.index_dir)
    elif args.command == "ask":
        answer_query(
            query=args.query,
            index_dir=args.index_dir,
            strategy=args.strategy,
            top_k=args.top_k,
            context_budget=args.context_budget,
            compression=args.compression,
            compression_stage=args.compression_stage,
            llmlingua_rate=args.llmlingua_rate,
            llmlingua_model=args.llmlingua_model,
            run_label=args.run_label,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
