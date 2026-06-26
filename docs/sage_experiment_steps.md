# SAGE 语义分割实验执行步骤

本文档用于在本地或服务器上执行 SAGE-style semantic chunk 实验。当前实验只复现 SAGE 的语义分割模块：`AnglE embedding + MLP + threshold split`，不改动现有 DenseX、LLMLingua、前端和 QA 生成流程。

## 1. 实验目标

对比固定长度 chunk 与 SAGE semantic chunk 在 QA 检索中的效果，重点观察：

- evidence 是否进入候选集；
- evidence 在 parent candidate 中的排名是否更靠前；
- evidence 是否进入最终 selected context；
- 相同 token budget 下最终回答准确率是否提升。

本实验新增的输出目录：

```text
experiments/sage_pairs/
experiments/sage_corpus/
models/sage_segmenter_angle/
storage/sage/
```

## 2. 前置条件

确认已经存在基础索引 metadata：

```bash
python app.py index --docs ./data
```

确认环境中已经安装依赖：

```bash
pip install -r requirements.txt
```

如果服务器不能连接 Hugging Face，需要提前把 embedding model 下载到本地，并把下面命令中的：

```text
WhereIsAI/UAE-Large-V1
```

替换为本地模型目录。

## 3. 生成固定 chunk 语料

SAGE 对比实验需要在同一个 embedding model 下分别构建 fixed chunk index 和 semantic chunk index。先生成固定 chunk JSONL：

```bash
python experiments/densex_prepare_corpus.py  --granularities chunk
```

输出：

```text
experiments/densex_corpus/chunk.jsonl
```

## 4. 构造 SAGE 训练样本

使用现有 PDF chunk metadata 构造相邻句子对：

```bash
python experiments/sage_build_pairs.py --metadata storage/metadata.json --output experiments/sage_pairs/pairs.jsonl  --validation-output experiments/sage_pairs/validation.jsonl --max-pairs 5000
```

说明：

- 同一段落内的相邻句子标记为 `label=1`；
- 不同段落之间的相邻句子标记为 `label=0`；
- `--max-pairs 5000` 是第一版推荐值，便于快速验证流程。

如果只想先在 ACDepth 上做 smoke test：

```bash
python experiments/sage_build_pairs.py \
  --metadata storage/metadata.json \
  --output experiments/sage_pairs/acd_pairs.jsonl \
  --validation-output experiments/sage_pairs/acd_validation.jsonl \
  --source-contains "ACD" \
  --max-pairs 1000
```

## 5. 训练轻量级 MLP

第一版按照论文思路默认使用 MSE：

```bash
python experiments/sage_train_segmenter.py --pairs experiments/sage_pairs/pairs.jsonl  --validation-pairs experiments/sage_pairs/validation.jsonl --embedding-model WhereIsAI/UAE-Large-V1 --output-dir models/sage_segmenter_angle --epochs 3  --batch-size 16 --loss mse --device auto
```

输出：

```text
models/sage_segmenter_angle/config.json
models/sage_segmenter_angle/mlp.pt
models/sage_segmenter_angle/metrics.json
```

如果显存或内存不足，可以先降低 batch size：

```bash
--batch-size 4
```

## 6. 生成 semantic chunk 语料

使用训练好的 MLP 对相邻句子打分，并根据阈值切分 semantic chunk。当前实现会先把文本切成段落，**段落边界作为强切分边界**；MLP 只在同一段落内部判断相邻句子是否继续切分。过短段落第一版不做合并，先保留为独立 semantic chunk，便于观察段落强边界对检索结果的影响。

推荐使用 `--docs data` 从原始文档直接读取 page text，避免从固定长度 metadata chunks 反推页面文本时丢失段落换行。

```bash
python experiments/sage_prepare_corpus.py --docs data --model-dir models/sage_segmenter_angle --embedding-model WhereIsAI/UAE-Large-V1 --output experiments/sage_corpus/semantic_chunk.jsonl --threshold 0.55
```

输出：

```text
experiments/sage_corpus/semantic_chunk.jsonl
```

如果要与旧版 semantic chunk 结果区分，建议输出为新文件：

```bash
python experiments/sage_prepare_corpus.py \
  --docs data \
  --model-dir models/sage_segmenter_angle \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --output experiments/sage_corpus/semantic_chunk_paragraph_v1.jsonl \
  --threshold 0.55
```

如果需要明确区分“直接从原始文档提取段落”的版本，建议使用 v3 命名：

```bash
python experiments/sage_prepare_corpus.py \
  --docs data \
  --model-dir models/sage_segmenter_angle \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --output experiments/sage_corpus/semantic_chunk_paragraph_v3.jsonl \
  --threshold 0.55
```

后续可以扫描不同阈值：

```bash
--threshold 0.45
--threshold 0.50
--threshold 0.55
--threshold 0.60
--threshold 0.65
```

不同阈值实验建议输出到不同文件，例如：

```text
experiments/sage_corpus/semantic_chunk_t055.jsonl
experiments/sage_corpus/semantic_chunk_t060.jsonl
```

## 7. 构建 FAISS index

固定 chunk 和 semantic chunk 必须使用同一个 embedding model 构建索引，否则对比不公平。

构建 fixed chunk index：

```bash
python experiments/sage_build_index.py --corpus experiments/densex_corpus/chunk.jsonl --index-dir storage/sage/chunk --embedding-model WhereIsAI/UAE-Large-V1
```

构建 semantic chunk index：

```bash
python experiments/sage_build_index.py --corpus experiments/sage_corpus/semantic_chunk.jsonl --index-dir storage/sage/semantic_chunk --embedding-model WhereIsAI/UAE-Large-V1
```

如果使用段落强边界的新语料，则构建独立 index：

```bash
python experiments/sage_build_index.py \
  --corpus experiments/sage_corpus/semantic_chunk_paragraph_v3.jsonl \
  --index-dir storage/sage/semantic_chunk_paragraph_v3 \
  --embedding-model WhereIsAI/UAE-Large-V1
```

## 8. 运行 QA 对比实验

先只跑 ACDepth 四个问题：

```bash
python experiments/run_sage_semantic_chunk_sweep.py --questions evaluation/questions.jsonl --question-ids always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods --embedding-model WhereIsAI/UAE-Large-V1 --budgets 300,500,1000,1500 --top-k 50 --run-label-prefix sage_semantic_v1
python experiments/run_sage_semantic_chunk_sweep.py \
  --questions evaluation/questions.jsonl \
  --question-ids always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --semantic-index-dir storage/sage/semantic_chunk_paragraph_v3 \
  --budgets 300,500,1000,1500 \
  --top-k 50 \
  --run-label-prefix sage_semantic_paragraph_v3
```

如果 ACDepth 流程跑通，再跑全部问题：

```bash
python experiments/run_sage_semantic_chunk_sweep.py --questions evaluation/questions.jsonl --all-questions --embedding-model WhereIsAI/UAE-Large-V1 --budgets 300,500,1000,1500 --top-k 50 --run-label-prefix sage_semantic_all_v1
```

如果全量实验中网络不稳定，建议使用可恢复的循环脚本。该脚本会：

- 对 APIConnectionError、SSL EOF、timeout、429、5xx 等临时错误自动重试；
- 某条任务失败后继续执行后续任务；
- 将失败任务写入 `experiments/sage_failed_jobs.jsonl`；
- 默认跳过已经有成功 answer 的 `run_label`，可以断点续跑。

Windows PowerShell 推荐命令：

```powershell
python experiments/run_sage_semantic_chunk_loop.py `
  --questions evaluation/questions.jsonl `
  --all-questions `
  --embedding-model WhereIsAI/UAE-Large-V1 `
  --budgets 300,500,1000,1500 `
  --top-k 50 `
  --run-label-prefix sage_semantic_all_v1 `
  --max-retries 5 `
  --retry-wait-seconds 20 `
  --retry-backoff 1.5 `
  --allow-failures
```

如果脚本中途断开，直接再次执行同一条命令即可。它会扫描 `experiments/runs/`，跳过已经完成的 run label。

## 9. 结果分析

QA 实验结束后，继续使用现有 DenseX evaluation 脚本统计准确率：

```bash
python experiments/evaluate_densex_runs.py \
  --questions evaluation/questions.jsonl \
  --run-label-prefix sage_semantic_paragraph_v3 \
  --output experiments/accuracy_results_sage_semantic_paragraph_v3.csv
```

重点比较：

- `chunk` vs `semantic_chunk`;
- 相同 budget 下的 recall / precision / f1；
- 达到相同 recall 时需要的 context tokens；
- ACDepth 的 evidence 是否更容易进入 selected context。

如果 semantic chunk 的候选 evidence 命中率提高，但 selected evidence recall 没有提高，说明下一步重点仍然是优化 selected context 选择，而不是继续调整切分。

## 10. 常见问题

### 10.1 模型下载卡住

如果卡在 `WhereIsAI/UAE-Large-V1` 下载，先在能联网的机器下载模型，再复制到服务器，并把命令中的 `--embedding-model` 改成本地目录。

### 10.2 内存不足

优先降低：

```bash
--batch-size 4
--max-pairs 1000
```

也可以先用当前项目已有的轻量模型做流程验证：

```bash
--embedding-model BAAI/bge-small-en-v1.5
```

但这只能叫 `semantic_chunk_bge_small`，不能和 AnglE 版本混为同一个实验。

### 10.3 对比结果没有提升

这不一定说明 SAGE segmentation 无效，需要拆开看：

- 如果 `first_evidence_parent_rank` 没有下降，说明 semantic chunk 没改善 retrieval ranking；
- 如果 evidence 已经进入候选但 selected 没命中，说明 selected context 选择策略仍然是瓶颈；
- 如果 selected 命中但 answer recall 低，说明问题可能在 LLM 回答或 gold item 匹配。
