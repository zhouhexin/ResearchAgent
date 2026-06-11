# Windows Experiment Steps

This guide runs the current fixed QA experiments on a Windows machine with
PowerShell.

## 1. Sync Code

```powershell
cd E:\PycharmProject\ResearchAgent
git pull origin main
```

If this is a fresh machine:

```powershell
git clone https://github.com/zhouhexin/ResearchAgent.git
cd ResearchAgent
```

## 2. Prepare Python Environment

Use your existing conda/venv if it already has the project dependencies.
Otherwise create a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

If Hugging Face downloads are slow:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

Create local config if needed:

```powershell
copy .env.example .env
```

Edit `.env` and set `MINIMAX_API_KEY`.

## 3. Build Baseline Chunk Index

Run this after a fresh clone or after changing files in `data\`:

```powershell
python app.py index --docs .\data
```

This builds:

```text
storage\index.faiss
storage\metadata.json
```

## 4. Prepare DenseX Corpora

Generate chunk and sentence corpora:

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities chunk,sentence `
  --metadata storage\metadata.json `
  --resume
  
python experiments/densex_prepare_corpus.py  --granularities chunk,sentence   --metadata storage/metadata.json  --device auto    --resume --proposition-model=./models/propositionizer-wiki-flan-t5-large

```

Generate propositions with the contextual 25-40 word prompt:

```powershell
python experiments\densex_prepare_corpus.py --granularities proposition --metadata storage\metadata.json --device auto --resume  --proposition-model=./models/propositionizer-wiki-flan-t5-large
```

For a quick proposition smoke test:

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities proposition `
  --source-contains "Always Clear Depth" `
  --device auto `
  --resume `
  --limit 1 `
  --max-new-tokens 512
```

## 5. Check Proposition Length

After proposition generation, check whether the new prompt actually produces
longer propositions:

```powershell
@'
import json
from pathlib import Path

path = Path("experiments/densex_corpus/proposition.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
lens = [len(row.get("text", "").split()) for row in rows]
print("count:", len(lens))
print("avg_words:", sum(lens) / len(lens) if lens else 0)
print("min_words:", min(lens) if lens else 0)
print("max_words:", max(lens) if lens else 0)
'@ | python
```

## 6. Build DenseX FAISS Indexes

```powershell
python experiments\densex_build_index.py --granularities chunk,sentence,proposition
```

Expected outputs:

```text
storage\densex\chunk
storage\densex\sentence
storage\densex\proposition
```

## 7. Run Direct Granularity Baseline

This compares chunk, sentence, and proposition as direct context units:

```powershell
python experiments\run_fixed_qa_batch.py --question-set fixed --granularities chunk,sentence,proposition --budgets 500,1000,1500  --top-k 50 --strategy baseline  --run-prefix qa_v2
```

Outputs:

```text
experiments\qa_v2_densex_results.csv
experiments\qa_v2_densex_summary.csv
```

## 8. Run Fine-To-Chunk V1

This uses sentence/proposition retrieval to locate evidence, then maps results
back to parent chunks for context construction:

```powershell
python experiments\run_densex_parent_sweep.py --question-set fixed --fine-granularities sentence,proposition  --budgets 500,1000,1500 --fine-top-m 150 --parent-top-k 50 --strategy baseline --run-label-prefix qa_parent_v1
```

Evaluate:

```powershell
python experiments\evaluate_densex_runs.py  --run-label-prefix qa_parent_v1  --output experiments\qa_parent_v1_densex_results.csv
```

## 9. Compare Results

Open these CSV files:

```text
experiments\qa_v2_densex_summary.csv
experiments\qa_parent_v1_densex_results.csv
```

Focus on:

```text
answer_f1
answer_recall
context_tokens
selected_gold_recall
selected_relevance_precision
token_efficiency
```

Interpretation:

- If `sentence-to-chunk` beats `sentence`, sentence retrieval is useful as an evidence locator.
- If `proposition-to-chunk` beats `proposition`, propositions need parent context for generation.
- If fine-to-chunk approaches chunk performance with fewer tokens, fine-grained retrieval is useful.
- If fine-to-chunk is still weak, inspect proposition quality and retrieval misses before training a new model.

## 10. Run All Questions

After `evaluation\questions.jsonl` has been expanded, run all QA pairs with a
separate prefix. This keeps the full-question results separate from earlier
ACDepth/DepthDark-only runs.

Direct chunk/sentence/proposition comparison:

```powershell
python experiments\run_fixed_qa_batch.py `
  --question-set all `
  --granularities chunk,sentence,proposition `
  --budgets 500,1000,1500 `
  --top-k 50 `
  --strategy baseline `
  --run-prefix qa_all_v1
```

Outputs:

```text
experiments\qa_all_v1_densex_results.csv
experiments\qa_all_v1_densex_summary.csv
```

Fine-to-chunk comparison:

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set all `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix qa_all_parent_v1
```

Evaluate:

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix qa_all_parent_v1 `
  --output experiments\qa_all_parent_v1_densex_results.csv
```

Quick command check without running the LLM:

```powershell
python experiments\run_fixed_qa_batch.py `
  --question-set all `
  --granularities chunk `
  --budgets 500 `
  --run-prefix qa_all_check `
  --dry-run

python experiments\run_densex_parent_sweep.py `
  --question-set all `
  --fine-granularities sentence `
  --budgets 500 `
  --run-label-prefix qa_all_parent_check `
  --dry-run
```

If the current `evaluation\questions.jsonl` contains N questions, the direct
comparison with 3 granularities and 3 budgets will produce:

```text
N questions × 3 granularities × 3 budgets = N × 9 LLM runs
```

The fine-to-chunk comparison with 2 granularities and 3 budgets will produce:

```text
N questions × 2 granularities × 3 budgets = N × 6 LLM runs
```

## 11. Optional: Run Only DepthDark

```powershell
python experiments\run_fixed_qa_batch.py `
  --question-set depthdark `
  --granularities chunk,sentence,proposition `
  --budgets 500,1000,1500 `
  --top-k 50 `
  --strategy baseline `
  --run-prefix depthdark_v2
```

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set depthdark `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix depthdark_parent_v1
```

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix depthdark_parent_v1 `
  --output experiments\depthdark_parent_v1_densex_results.csv
```
