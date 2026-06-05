# Experiment Test Steps

This document records the fixed steps for running QA-based retrieval and
granularity experiments.

## 1. Prepare Environment

Run from the project root:

```bash
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `MINIMAX_API_KEY`. If Hugging Face downloads are slow, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Verify package imports:

```bash
python - <<'PY'
import compression
import compression.llmlingua2 as llmlingua2
print(compression.__file__)
print(llmlingua2.__file__)
PY
```

## 2. Build Or Refresh The Baseline Index

Use this when setting up a fresh machine or after changing files in `data/`:

```bash
python app.py index --docs ./data
```

The first run downloads the embedding model configured by `EMBEDDING_MODEL`.
The default is `sentence-transformers/all-MiniLM-L6-v2`.

## 3. Fixed QA Sets

Use these question IDs for the current manually checked QA set:

```bash
ACDEPTH_QA_IDS="always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods"
DEPTHDARK_QA_IDS="depthdark_contributions,depthdark_eval_datasets,depthdark_training_datasets,depthdark_ablation_components,depthdark_sota_comparison_methods"
FIXED_QA_IDS="${ACDEPTH_QA_IDS},${DEPTHDARK_QA_IDS}"
```

The executable QA file is:

```text
evaluation/questions.jsonl
```

The human review file is:

```text
evaluation/questions_review.md
```

## 4. Prepare DenseX Corpora

Generate chunk and sentence corpora:

```bash
python experiments/densex_prepare_corpus.py \
  --granularities chunk,sentence \
  --metadata storage/metadata.json \
  --resume
```

Generate proposition corpus:

```bash
python experiments/densex_prepare_corpus.py \
  --granularities proposition \
  --metadata storage/metadata.json \
  --device auto \
  --resume
```

On CPU-only machines, proposition generation can be slow. `--resume` lets the
process continue from previously written units.

## 5. Build DenseX Indexes

```bash
python experiments/densex_build_index.py \
  --granularities chunk,sentence,proposition
```

Expected outputs:

```text
storage/densex/chunk
storage/densex/sentence
storage/densex/proposition
```

## 6. Run Fixed QA Granularity Test

Recommended one-command batch runner:

```bash
python experiments/run_fixed_qa_batch.py \
  --question-set fixed \
  --granularities chunk,sentence,proposition \
  --budgets 500,1000,1500 \
  --top-k 50 \
  --strategy baseline \
  --run-prefix qa_v1
```

This command runs the fixed QA sweep, evaluates the generated runs, and writes:

```text
experiments/qa_v1_densex_results.csv
experiments/qa_v1_densex_summary.csv
```

Use `--question-set depthdark` or `--question-set acdepth` to run only one
paper's QA set.

If corpora or indexes need to be rebuilt, add:

```bash
python experiments/run_fixed_qa_batch.py \
  --question-set fixed \
  --granularities chunk,sentence,proposition \
  --budgets 500,1000,1500 \
  --top-k 50 \
  --strategy baseline \
  --run-prefix qa_v1 \
  --prepare-corpora \
  --build-index
```

For a quick command check without running experiments:

```bash
python experiments/run_fixed_qa_batch.py \
  --question-set depthdark \
  --granularities chunk,sentence \
  --budgets 500 \
  --top-k 20 \
  --run-prefix smoke \
  --dry-run
```

The manual command sequence is below.

Use a unique prefix for each experiment batch:

```bash
RUN_PREFIX="qa_v1"
```

Run chunk, sentence, and proposition under the same budgets:

```bash
python experiments/run_densex_sweep.py \
  --question-ids "$FIXED_QA_IDS" \
  --granularities chunk,sentence,proposition \
  --budgets 500,1000,1500 \
  --top-k 50 \
  --strategy baseline \
  --run-label-prefix "$RUN_PREFIX"
```

For a quick smoke test, use one question and one budget first:

```bash
python experiments/run_densex_sweep.py \
  --question-ids "depthdark_training_datasets" \
  --granularities chunk,sentence \
  --budgets 500 \
  --top-k 20 \
  --strategy baseline \
  --run-label-prefix "smoke"
```

## 7. Evaluate Runs

Evaluate only the batch with the matching prefix:

```bash
python experiments/evaluate_densex_runs.py \
  --run-label-prefix "$RUN_PREFIX" \
  --output "experiments/${RUN_PREFIX}_densex_results.csv"
```

For the smoke test:

```bash
python experiments/evaluate_densex_runs.py \
  --run-label-prefix smoke \
  --output experiments/smoke_densex_results.csv
```

## 8. Check Key Metrics

Focus on these fields first:

```text
answer_f1
answer_recall
context_tokens
selected_gold_recall
selected_relevance_precision
token_efficiency
```

Recommended interpretation:

- `answer_f1`: whether the final answer includes the expected gold items.
- `answer_recall`: how many gold items were recovered in the final answer.
- `context_tokens`: actual context size used by the model.
- `selected_gold_recall`: whether selected context mentions the gold items.
- `selected_relevance_precision`: how many selected units are relevant by alias matching.
- `token_efficiency`: answer F1 normalized by context tokens.

## 9. Commit Reproducible Changes

Commit code, QA, and documentation changes:

```bash
git status --short
git add evaluation/questions.jsonl evaluation/questions_review.md docs/experiment_test_steps.md experiments/evaluate_densex_runs.py README.md
git commit -m "Add fixed QA experiment steps"
git push origin main
```

Generated run details under `experiments/runs/`, DenseX generated corpora under
`experiments/densex_corpus/`, and DenseX indexes under `storage/densex/` are not
committed.
