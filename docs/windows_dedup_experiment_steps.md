# Windows Dedup Experiment Steps

本文件用于在 Windows 服务器上执行 sentence/proposition 精确去重后的对比实验。
它不会覆盖已有的 `qa_v2` 或 `qa_parent_v1` 结果，新增结果统一使用 `qa_dedup_v1`
前缀。

## 1. 更新代码

```powershell
cd E:\PycharmProject\ResearchAgent
git pull origin main
```

激活项目 Python 环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果当前机器还没有安装依赖：

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

如果 Hugging Face 下载慢，可以临时设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

建议在运行 FAISS + sentence-transformers 实验前限制线程，避免部分机器上的
OpenMP/FAISS 运行时冲突：

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
```

## 2. 确认基础索引

如果是新机器，或 `data\` 中 PDF 有变化，先重建基础 chunk 索引：

```powershell
python app.py index --docs .\data
```

确认存在：

```text
storage\metadata.json
storage\index.faiss
```

## 3. 准备原始 DenseX Corpus

如果已经生成过 `sentence.jsonl` 和 `proposition.jsonl`，可以跳过本节。

生成 chunk 和 sentence corpus：

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities chunk,sentence `
  --metadata storage\metadata.json `
  --resume
```

生成 proposition corpus：

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities proposition `
  --metadata storage\metadata.json `
  --device auto `
  --resume `
  --max-new-tokens 512
```

## 4. 生成 Dedup Corpus

生成 `sentence_dedup.jsonl` 和 `proposition_dedup.jsonl`：

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities sentence_dedup,proposition_dedup `
  --metadata storage\metadata.json
```

说明：

- `sentence_dedup` 会从原始 chunk 重新切 sentence，然后做全局规范化精确去重。
- `proposition_dedup` 会优先复用已有 `experiments\densex_corpus\proposition.jsonl`，
  然后做全局规范化精确去重。
- 去重方式不是 embedding 语义去重，只删除规范化后完全相同的文本。

可以检查去重前后的数量：

```powershell
@'
from pathlib import Path
from densex.corpus import read_jsonl

base = Path("experiments/densex_corpus")
for name in ["sentence", "sentence_dedup", "proposition", "proposition_dedup"]:
    path = base / f"{name}.jsonl"
    rows = read_jsonl(path)
    print(f"{name}: {len(rows)}")
'@ | python
```

## 5. 构建 Dedup FAISS 索引

如果只跑 dedup 实验：

```powershell
python experiments\densex_build_index.py `
  --granularities sentence_dedup,proposition_dedup
```

如果还需要重新构建原始 fine-grained 索引用于对比：

```powershell
python experiments\densex_build_index.py `
  --granularities sentence,proposition,sentence_dedup,proposition_dedup
```

确认存在：

```text
storage\densex\sentence_dedup
storage\densex\proposition_dedup
```

## 6. 运行 Dedup Fine-To-Chunk 实验

运行 sentence/proposition 去重后的 parent chunk 回填实验：

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set fixed `
  --fine-granularities sentence_dedup,proposition_dedup `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix qa_dedup_v1
```

生成的 run JSON 会保存在：

```text
experiments\runs\
```

run label 中会包含：

```text
qa_dedup_v1_sentence_dedup-to-chunk_...
qa_dedup_v1_proposition_dedup-to-chunk_...
```

## 7. 评估 Dedup 结果

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix qa_dedup_v1 `
  --output experiments\qa_dedup_v1_densex_results.csv
```

生成简单 summary：

```powershell
@'
import csv
from collections import defaultdict
from pathlib import Path

input_path = Path("experiments/qa_dedup_v1_densex_results.csv")
output_path = Path("experiments/qa_dedup_v1_densex_summary.csv")
rows = list(csv.DictReader(input_path.open(encoding="utf-8", newline="")))
groups = defaultdict(list)
for row in rows:
    groups[(row["granularity"], row["budget"])].append(row)

fields = [
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

def f(row, key):
    return float(row.get(key) or 0)

with output_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for (granularity, budget), items in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        count = len(items)
        writer.writerow({
            "granularity": granularity,
            "budget": budget,
            "run_count": count,
            "avg_context_tokens": sum(f(row, "context_tokens") for row in items) / count,
            "avg_answer_f1": sum(f(row, "answer_f1") for row in items) / count,
            "avg_answer_recall": sum(f(row, "answer_recall") for row in items) / count,
            "avg_selected_gold_recall": sum(f(row, "selected_gold_recall") for row in items) / count,
            "avg_selected_relevance_precision": sum(f(row, "selected_relevance_precision") for row in items) / count,
            "avg_token_efficiency": sum(f(row, "token_efficiency") for row in items) / count,
        })

print(f"Wrote {output_path}")
'@ | python
```

## 8. 对比结果

重点对比这几个文件：

```text
experiments\qa_parent_v1_densex_results.csv
experiments\qa_dedup_v1_densex_results.csv
experiments\qa_dedup_v1_densex_summary.csv
```

主要看：

```text
answer_f1
answer_recall
context_tokens
selected_gold_recall
selected_relevance_precision
token_efficiency
```

解释方式：

- 如果 `sentence_dedup-to-chunk` 优于 `sentence-to-chunk`，说明 sentence 重复会消耗
  检索名额或 context token。
- 如果 `proposition_dedup-to-chunk` 优于 `proposition-to-chunk`，说明 proposition 重复
  会影响 parent chunk 排序。
- 如果 F1 没提升但 `context_tokens` 降低，说明去重可能提升 token efficiency。
- 如果 F1 和 recall 都下降，说明精确去重误删了有用证据，后续要改成按论文或按
  parent chunk 局部去重。

## 9. 可选：只跑 DepthDark

如果先做小范围验证：

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set depthdark `
  --fine-granularities sentence_dedup,proposition_dedup `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix depthdark_dedup_v1
```

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix depthdark_dedup_v1 `
  --output experiments\depthdark_dedup_v1_densex_results.csv
```
