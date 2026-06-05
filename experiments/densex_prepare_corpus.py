"""Prepare chunk, sentence, and proposition corpora for DenseX experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from densex.corpus import (
    append_jsonl,
    load_chunks_from_metadata,
    make_chunk_units,
    make_sentence_units,
    normalize_proposition_output,
    paper_title_from_source,
    read_jsonl,
    write_jsonl,
)


DEFAULT_MODEL = "chentong00/propositionizer-wiki-flan-t5-large"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _filter_chunks(chunks: list[dict], source_contains: str) -> list[dict]:
    if not source_contains:
        return chunks
    needle = source_contains.lower()
    return [chunk for chunk in chunks if needle in chunk.get("source", "").lower()]


def _device_arg(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_propositionizer(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install propositionizer dependencies first: "
            "`pip install transformers sentencepiece accelerate torch`."
        ) from exc

    # The propositionizer is based on Flan-T5/SentencePiece. Some Transformers
    # versions try to convert SentencePiece tokenizers into fast tokenizers and
    # require protobuf for that conversion; when protobuf is missing they may
    # incorrectly fall back to TikToken and fail to parse `spiece.model`. The
    # slow tokenizer is stable here and avoids that conversion path.
    print(f"Loading propositionizer tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    kwargs = {"low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    print(f"Loading propositionizer model on device={device}: {model_name}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    print("Propositionizer model loaded.", flush=True)
    return tokenizer, model, torch


def _propositionize(
    *,
    chunks: list[dict],
    output_path: Path,
    model_name: str,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    resume: bool,
) -> None:
    existing_ids = {row.get("parent_chunk_id") for row in read_jsonl(output_path)} if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()

    print(
        f"Preparing propositions for {len(chunks)} chunks "
        f"(resume={resume}, existing_parent_chunks={len(existing_ids)}).",
        flush=True,
    )
    tokenizer, model, torch = _load_propositionizer(model_name, device)
    for chunk_index, chunk in enumerate(chunks, start=1):
        parent_id = chunk.get("id")
        if parent_id in existing_ids:
            continue

        title = paper_title_from_source(chunk.get("source", ""))
        section = f"page {chunk.get('page')}" if chunk.get("page") else "unknown"
        prompt = f"Title: {title}. Section: {section}. Content: {chunk.get('text', '')}"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        print(f"[{chunk_index}/{len(chunks)}] Generating propositions for {parent_id}", flush=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        propositions = normalize_proposition_output(raw)

        for prop_index, proposition in enumerate(propositions):
            append_jsonl(
                output_path,
                {
                    "id": f"proposition::{parent_id}::p{prop_index}",
                    "granularity": "proposition",
                    "text": proposition,
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "parent_chunk_id": parent_id,
                    "proposition_index": prop_index,
                    "paper_title": title,
                    "model_name": model_name,
                },
            )
        print(
            f"[{chunk_index}/{len(chunks)}] {parent_id}: "
            f"{len(propositions)} propositions",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DenseX corpora")
    parser.add_argument("--metadata", type=Path, default=config.INDEX_DIR / "metadata.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "densex_corpus")
    parser.add_argument("--granularities", default="chunk,sentence,proposition")
    parser.add_argument("--source-contains", default="", help="Optional source path filter")
    parser.add_argument("--limit", type=int, default=0, help="Optional chunk limit for smoke tests")
    parser.add_argument("--sentence-min-chars", type=int, default=30)
    parser.add_argument("--proposition-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    chunks = _filter_chunks(load_chunks_from_metadata(args.metadata), args.source_contains)
    if not chunks:
        raise RuntimeError("No chunks matched the requested metadata/filter")
    if args.limit > 0:
        chunks = chunks[: args.limit]
    print(
        f"Loaded {len(chunks)} chunks from {args.metadata} "
        f"(source_contains={args.source_contains!r}, limit={args.limit}).",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    granularities = set(_parse_csv(args.granularities))

    if "chunk" in granularities:
        path = args.output_dir / "chunk.jsonl"
        write_jsonl(path, make_chunk_units(chunks))
        print(f"Wrote {path}")

    if "sentence" in granularities:
        path = args.output_dir / "sentence.jsonl"
        units = make_sentence_units(chunks, min_chars=args.sentence_min_chars)
        write_jsonl(path, units)
        print(f"Wrote {path}")

    if "proposition" in granularities:
        path = args.output_dir / "proposition.jsonl"
        _propositionize(
            chunks=chunks,
            output_path=path,
            model_name=args.proposition_model,
            device=_device_arg(args.device),
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
