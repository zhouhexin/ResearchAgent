# ResearchAgent

Minimal local RAG research assistant with document chunking, FAISS retrieval,
context allocation, prompt building, and MiniMax generation.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `MINIMAX_API_KEY`.

## Server handoff

For a fresh machine, clone the repository and install dependencies:

```bash
git clone <repo-url>
cd ResearchAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set the API key and model settings. The repository keeps
the PDF corpus in `data/` and the current FAISS index in `storage/`, so the
baseline can run immediately after dependencies are installed. If the PDF corpus
changes, rebuild the index with `python app.py index --docs ./data`.

Generated run-detail JSON files under `experiments/runs/`, DenseX generated
corpora under `experiments/densex_corpus/`, and DenseX indexes under
`storage/densex/` are intentionally ignored by git.

Fixed QA and DenseX experiment steps are documented in
`docs/experiment_test_steps.md`.

Windows PowerShell experiment steps are documented in
`docs/windows_experiment_steps.md`.

Frontend implementation order is documented in
`docs/frontend_implementation_plan.md`.

## Usage

Put `.txt`, `.md`, `.markdown`, or `.pdf` files in `data/`, then build the index:

```bash
python app.py index --docs ./data
```

Ask a question:

```bash
python app.py ask --query "这批资料的核心结论是什么？"
```

Print the generated prompt without calling the LLM:

```bash
python app.py ask --query "这批资料的核心结论是什么？" --dry-run
```

Compare allocation strategies:

```bash
python app.py ask --query "问题" --strategy baseline
python app.py ask --query "问题" --strategy dynamic
python app.py ask --query "问题" --strategy rerank
```

Run the first compression baseline with LLMLingua-2:

```bash
python app.py ask \
  --query "问题" \
  --strategy rerank \
  --top-k 20 \
  --context-budget 2000 \
  --compression llmlingua2 \
  --llmlingua-rate 0.5
```

This baseline compresses selected chunks after allocation. That keeps the
allocation strategy comparable with existing runs, while recording how many
context tokens LLMLingua-2 removes before prompt construction.

Run before-allocation LLMLingua-2 to let allocation use compressed token counts:

```bash
python app.py ask \
  --query "问题" \
  --strategy baseline \
  --top-k 50 \
  --context-budget 2000 \
  --compression llmlingua2 \
  --compression-stage before-allocation \
  --llmlingua-rate 0.5
```

Before-allocation compression first compresses retrieved candidates, then fills
the budget with compressed chunks. This is the mode that can select more chunks
under the same final context budget.

Run a P0 budget sweep for strict comparison:

```bash
python experiments/run_budget_sweep.py \
  --query "问题" \
  --strategies baseline,dynamic,rerank \
  --budgets 500,1000,2000,4000 \
  --top-k 20
```

Compare no compression against LLMLingua-2 in the same sweep:

```bash
python experiments/run_budget_sweep.py \
  --query "问题" \
  --strategies baseline,dynamic,rerank \
  --compressions none,llmlingua2 \
  --compression-stages after-allocation,before-allocation \
  --budgets 1000,2000 \
  --top-k 20 \
  --dry-run
```

Evaluate DenseX saved answers against list-style gold labels:

```bash
python experiments/evaluate_densex_runs.py \
  --run-label-prefix densex_ \
  --output experiments/densex_results.csv
```

Gold questions live in `evaluation/questions.jsonl`. The first evaluator is a
deterministic list metric: it matches known item names and aliases in each
answer, then reports precision, recall, and F1 in
the selected output CSV.

The same CSV also reports evidence recall at two pipeline stages:

```text
检索标准答案召回率：retrieved_gold_recall = gold items found in retrieved chunks / gold items
筛选标准答案召回率：selected_gold_recall = gold items found in selected prompt chunks / gold items
```

These fields help separate retrieval failures from allocation, compression, and
generation failures.

Each non-dry-run answer appends a row to `experiments/results.csv`, including
MiniMax token usage when the API response provides it:

```text
prompt_tokens, completion_tokens, total_tokens, model
context_tokens, original_context_tokens, compression, compression_ratio
selected_chunk_count(以筛选分块数), retrieved_chunk_count（以检索分块数）
citation_validity_ratio（引用有效率）, invalid_citation_count（引用无效率）
top_k（候选数量）, budget, strategy, run_id, details_path
```

Compact run details are saved as JSON files in `experiments/runs/`. Each JSON
file contains the retrieved chunks, selected chunks, final answer, token counts,
and usage metadata for reproducible analysis.

For LLMLingua-2 runs, details also include compact `compression_info`, so
compression behavior can be inspected after the run without storing the full
prompt text.
