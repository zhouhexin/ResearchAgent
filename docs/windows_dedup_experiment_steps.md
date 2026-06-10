# Windows Chunk-Aware Dedup Experiment Steps

本文件用于在 Windows 服务器上执行 fine-to-chunk 的 chunk-aware 精确去重实验。
主实验不需要构建 `sentence_dedup` 或 `proposition_dedup` 索引，而是使用原始
`sentence` / `proposition` 索引，在检索后、聚合回 parent chunk 时去重。

核心流程：

```text
query
-> 检索更多 sentence/proposition
-> 按 parent_chunk_id 分组
-> 每个 parent chunk 内部做规范化精确去重
-> 聚合 parent chunk 分数
-> 选 parent chunk 进入 context
```

这样可以避免全局去重导致某些 chunk 永远失去召回机会。

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

## 4. 构建原始 Fine-Grained 索引

本实验使用原始 `sentence` 和 `proposition` 索引：

```powershell
python experiments\densex_build_index.py `
  --granularities sentence,proposition
```

确认存在：

```text
storage\densex\sentence
storage\densex\proposition
```

## 5. 运行原始 Parent Baseline

如果你已经有 `qa_parent_v1_densex_results.csv`，可以跳过本节。

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set fixed `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix qa_parent_v1
```

评估：

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix qa_parent_v1 `
  --output experiments\qa_parent_v1_densex_results.csv
```

## 6. 运行 Chunk-Aware Dedup 实验

建议比 baseline 检索更多 fine hits，例如把 `fine-top-m` 从 150 提到 300 或 500。
去重发生在 parent chunk 内部，所以不会因为全局去重损失 chunk 覆盖。

推荐先跑 300：

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set fixed `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 300 `
  --parent-top-k 50 `
  --fine-hit-dedup exact-per-parent `
  --strategy baseline `
  --run-label-prefix qa_chunkaware_dedup_v1
```

如果 300 有提升，再跑 500：

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set fixed `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 500 `
  --parent-top-k 50 `
  --fine-hit-dedup exact-per-parent `
  --strategy baseline `
  --run-label-prefix qa_chunkaware_dedup_m500_v1
```

run JSON 会保存在：

```text
experiments\runs\
```

## 7. 评估 Dedup 结果

评估 top-M=300：

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix qa_chunkaware_dedup_v1 `
  --output experiments\qa_chunkaware_dedup_v1_densex_results.csv
```

评估 top-M=500：

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix qa_chunkaware_dedup_m500_v1 `
  --output experiments\qa_chunkaware_dedup_m500_v1_densex_results.csv
```

生成 summary。把 `$inputPath` 和 `$outputPath` 改成你要汇总的结果文件：

```powershell
$inputPath="experiments/qa_chunkaware_dedup_v1_densex_results.csv"
$outputPath="experiments/qa_chunkaware_dedup_v1_densex_summary.csv"

@'
import csv
import os
from collections import defaultdict
from pathlib import Path

input_path = Path(os.environ["INPUT_PATH"])
output_path = Path(os.environ["OUTPUT_PATH"])
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
'@ | set-content tmp_summary.py -Encoding utf8

$env:INPUT_PATH=$inputPath
$env:OUTPUT_PATH=$outputPath
python tmp_summary.py
remove-item tmp_summary.py
```

## 8. 对比结果

重点对比：

```text
experiments\qa_parent_v1_densex_results.csv
experiments\qa_chunkaware_dedup_v1_densex_results.csv
experiments\qa_chunkaware_dedup_m500_v1_densex_results.csv
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

- 如果 `qa_chunkaware_dedup_v1` 优于 `qa_parent_v1`，说明 parent chunk 内重复
  fine hits 会干扰 chunk 排序。
- 如果 `fine-top-m=500` 优于 `fine-top-m=300`，说明检索更多 fine hits 后再去重
  可以扩大有效候选覆盖。
- 如果 F1 没提升但 `selected_gold_recall` 提升，说明去重改善了证据进入 context，
  但答案生成或 gold alias 仍需要检查。
- 如果 F1 和 recall 都下降，说明去重或更大的 `fine-top-m` 引入了更多噪声，需要降低
  `fine-top-m` 或调整聚合权重。

## 9. 可选：Corpus-Level Dedup 对照

如果你仍想单独测试“去重索引”这个变量，可以额外执行：

```powershell
python experiments\densex_prepare_corpus.py `
  --granularities sentence_dedup,proposition_dedup `
  --metadata storage\metadata.json
```

```powershell
python experiments\densex_build_index.py `
  --granularities sentence_dedup,proposition_dedup
```

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set fixed `
  --fine-granularities sentence_dedup,proposition_dedup `
  --budgets 500,1000,1500 `
  --fine-top-m 150 `
  --parent-top-k 50 `
  --strategy baseline `
  --run-label-prefix qa_corpus_dedup_v1
```

这个对照可能损失 chunk 覆盖，因此不作为主实验。

## 10. 可选：只跑 DepthDark

小范围验证 top-M=300：

```powershell
python experiments\run_densex_parent_sweep.py `
  --question-set depthdark `
  --fine-granularities sentence,proposition `
  --budgets 500,1000,1500 `
  --fine-top-m 300 `
  --parent-top-k 50 `
  --fine-hit-dedup exact-per-parent `
  --strategy baseline `
  --run-label-prefix depthdark_chunkaware_dedup_v1
```

```powershell
python experiments\evaluate_densex_runs.py `
  --run-label-prefix depthdark_chunkaware_dedup_v1 `
  --output experiments\depthdark_chunkaware_dedup_v1_densex_results.csv
```
