# ACDepth Parent Selection 诊断实验步骤

本文档用于在服务器上定位 ACDepth 问题的断点：

```text
fine retrieval
-> parent aggregation
-> selected context
```

诊断脚本不会调用 LLM，不会生成 run JSON，也不会影响正式实验结果。

## 1. 同步代码

Linux / macOS:

```bash
cd /path/to/ResearchAgent
git pull origin main
```

Windows PowerShell:

```powershell
cd E:\PycharmProject\ResearchAgent
git pull origin main
```

如果是新服务器：

```bash
git clone https://github.com/zhouhexin/ResearchAgent.git
cd ResearchAgent
```

## 2. 准备 Python 环境

使用已有 conda/venv 环境即可。确认依赖已安装：

```bash
pip install -r requirements.txt
```

该诊断脚本需要加载 embedding 模型和 FAISS 索引，因此至少需要：

```text
numpy
faiss-cpu
sentence-transformers
```

如果服务器不能访问 Hugging Face，需要提前把 embedding model 缓存在服务器本地，或配置本地模型路径。当前本地常用模型是：

```text
BAAI/bge-small-en-v1.5
```

## 3. 确认基础索引存在

必须先有基础 chunk 索引：

```text
storage/index.faiss
storage/metadata.json
```

如果不存在，先构建：

```bash
python app.py index --docs ./data
```

然后确认 DenseX sentence / proposition 索引存在。至少需要下面之一：

```text
storage/densex/sentence
storage/densex/proposition
```

或去重版本：

```text
storage/densex/sentence_dedup
storage/densex/proposition_dedup
```

如果 DenseX 索引不存在，需要先准备 corpus 并构建索引：

```bash
python experiments/densex_prepare_corpus.py \
  --granularities sentence,proposition \
  --metadata storage/metadata.json \
  --device auto \
  --resume

python experiments/densex_build_index.py \
  --granularities sentence,proposition
```

如果服务器上已经只有 `sentence_dedup` / `proposition_dedup`，后续命令加上 `--allow-dedup-fallback` 即可。

## 4. 运行 original query 诊断

这组使用正式问题原文，例如：

```text
Always Clear Depth 的消融实验验证了哪些组件？
```

运行：

```bash
python experiments/diagnose_parent_selection.py \
  --question-ids always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods \
  --fine-granularities sentence,proposition \
  --budgets 500,1000,1500 \
  --fine-top-m 150 \
  --parent-top-k 50 \
  --strategy baseline \
  --allow-dedup-fallback \
  --output-prefix acdepth_parent_selection
```

输出：

```text
experiments/diagnostics/acdepth_parent_selection_summary.csv
experiments/diagnostics/acdepth_parent_selection_details.csv
```

## 5. 运行 content-only query 诊断

这组使用去掉论文名前缀后的内容 query，用来判断前缀是否影响 parent aggregation 和 selected context。

运行：

```bash
python experiments/diagnose_parent_selection.py \
  --question-ids always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods \
  --fine-granularities sentence,proposition \
  --budgets 500,1000,1500 \
  --fine-top-m 150 \
  --parent-top-k 50 \
  --strategy baseline \
  --query-mode content-only \
  --allow-dedup-fallback \
  --output-prefix acdepth_parent_selection_content_only
```

输出：

```text
experiments/diagnostics/acdepth_parent_selection_content_only_summary.csv
experiments/diagnostics/acdepth_parent_selection_content_only_details.csv
```
## 6. 重点查看的字段

优先看 `summary.csv`：

| 字段 | 含义 |
|---|---|
| `parent_candidate_gold_recall` | parent candidates 中是否包含正确答案项 |
| `parent_candidate_evidence_recall` | parent candidates 中是否包含正确 evidence page |
| `selected_gold_recall` | 最终 selected context 中是否包含正确答案项 |
| `selected_evidence_recall` | 最终 selected context 中是否包含正确 evidence page |
| `first_gold_parent_rank` | 第一个包含答案项的 parent chunk 排名 |
| `first_evidence_parent_rank` | 第一个命中 evidence page 的 parent chunk 排名 |
| `selected_first_gold_parent_rank` | 被选入 context 的第一个答案 parent chunk 原始排名 |
| `selected_first_evidence_parent_rank` | 被选入 context 的第一个 evidence parent chunk 原始排名 |

`details.csv` 用于人工检查具体 chunk：

| 字段 | 含义 |
|---|---|
| `parent_rank` | parent aggregation 后的排序 |
| `selected` | 是否进入最终 context |
| `selected_order` | 进入 context 后的顺序 |
| `source` / `page` | 来源论文和页码 |
| `matched_gold_ids` | 当前 parent chunk 命中的 gold items |
| `matched_evidence_refs` | 当前 parent chunk 命中的 evidence refs |
| `top_child_ids` | 触发该 parent chunk 的 sentence/proposition id |
| `text` | parent chunk 文本片段 |

## 7. 判断规则

按下面顺序判断问题断点：

```text
parent_candidate_evidence_recall 高，但 selected_evidence_recall 低
=> 正确证据进入了 parent candidates，但 selected 阶段被 budget 或排序挤掉。
=> 下一步优化 selected context 选择。
```

```text
parent_candidate_evidence_recall 低
=> 正确证据没有进入 parent candidates。
=> 下一步优化 fine retrieval、query、fine_top_m 或 parent_score。
```

```text
selected_evidence_recall 高，但最终 answer_recall 低
=> evidence 已进入 prompt，问题更可能在 prompt、LLM 回答格式或 gold alias 匹配。
```

## 8. 后续优化方向

如果问题在 parent aggregation：

```text
1. 提高 fine_top_m。
2. 使用 content-only query。
3. 对目标 source 做 boost。
4. 调整 parent_score，例如增加 source_match_bonus 或 query_term_overlap_bonus。
```

如果问题在 selected context：

```text
1. 不只按 parent_score 顺序塞满 token budget。
2. 优先保留目标 source、低重复、包含 evidence page 的 parent chunk。
3. 限制同一 source/page 的重复 chunk 数量。
4. 对标题页、参考文献、纯表格残片等低质量 chunk 降权。
```

## 9. 建议保存的结果

跑完后保留这四个文件，后续用于对比分析：

```text
experiments/diagnostics/acdepth_parent_selection_summary.csv
experiments/diagnostics/acdepth_parent_selection_details.csv
experiments/diagnostics/acdepth_parent_selection_content_only_summary.csv
experiments/diagnostics/acdepth_parent_selection_content_only_details.csv
```
