# DenseX Server Run Guide

This guide prepares all-paper chunk, sentence, and proposition indexes, then
runs the Always Clear Depth QA comparison.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install transformers sentencepiece accelerate torch
```

For CPU-only servers this still works, but proposition generation can be slow.
Use `--device cuda` on a GPU server.

## 2. Prepare Corpora

Generate chunk and sentence corpora locally or on the server:

```bash
python experiments/densex_prepare_corpus.py \
  --granularities chunk,sentence \
  --metadata storage/metadata.json
```

Generate propositions on the server:

```bash
python experiments/densex_prepare_corpus.py \
  --granularities proposition \
  --metadata storage/metadata.json \
  --device auto \
  --resume
```

Outputs:

```text
experiments/densex_corpus/chunk.jsonl
experiments/densex_corpus/sentence.jsonl
experiments/densex_corpus/proposition.jsonl
```

## 3. Build Indexes

```bash
python experiments/densex_build_index.py \
  --granularities chunk,sentence,proposition
```

Outputs:

```text
storage/densex/chunk
storage/densex/sentence
storage/densex/proposition
```

## 4. Run DenseX Sweep

```bash
python experiments/run_densex_sweep.py \
  --granularities chunk,sentence,proposition \
  --budgets 500,1000,1500 \
  --top-k 50 \
  --strategy baseline
```

This runs only the four Always Clear Depth QA pairs.

## 5. Evaluate

```bash
python experiments/evaluate_densex_runs.py
```

Output:

```text
experiments/densex_results.csv
```

Key fields:

- `answer_f1`: final answer accuracy over gold items.
- `retrieved_gold_recall`: whether gold items appeared in retrieved units.
- `selected_gold_recall`: whether gold items entered final context.
- `retrieved_relevance_precision`: fraction of retrieved units that mention any gold item.
- `selected_relevance_precision`: fraction of selected units that mention any gold item.
- `token_efficiency`: `answer_f1 / context_tokens * 1000`.
