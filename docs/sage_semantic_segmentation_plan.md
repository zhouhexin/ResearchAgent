# SAGE 语义分割模块复现方案

**日期:** 2026-06-17

## 1. 目标

在当前 ResearchAgent 项目中复现 SAGE: A Framework of Precise Retrieval for RAG 的语义分割模块，用语义完整的 chunk 替代固定长度 chunk，并评估它是否能提高 evidence 进入候选集、进入 selected context，以及最终 QA 的准确率。

本阶段只复现 SAGE 的 semantic segmentation，不复现 gradient-based chunk selection 和 LLM self-feedback。这样可以单独判断“语义切分”是否解决当前项目中 ACDepth 等问题的 evidence 排名和 selected 命中问题。

## 2. 论文依据

SAGE 认为固定长度切分容易破坏语义完整性，导致 chunk embedding 不能准确表示问题所需证据。论文的语义分割模块使用一个轻量级模型判断相邻句子是否应该放在同一个 chunk。

论文中的模型结构是：

```text
sentence_1, sentence_2
  -> AnglE / Angle-optimized embedding model
  -> x1, x2
  -> feature augmentation: x1, x2, x1 - x2, x1 * x2
  -> MLP
  -> same-chunk score
  -> score < threshold 时切分
```

训练标签来自已有段落结构：

- 同一段落内连续句子：label = 1，表示应该合并；
- 不同段落之间的句子：label = 0，表示应该切开。

## 3. 当前项目中的定位

当前项目已经有三类实验粒度：

- `chunk`: 固定长度 chunk；
- `sentence`: 句子级 retrieval unit；
- `proposition`: 命题级 retrieval unit。

SAGE 语义分割应该作为第四类语料粒度加入：

- `semantic_chunk`: 由轻量级语义分割模型生成的 chunk。

它不替代现有 chunk/sentence/proposition，也不改动现有 DenseX 实验逻辑。所有输出放到单独目录，便于和已有实验结果区分。

建议目录：

```text
experiments/sage_corpus/
  semantic_chunk.jsonl

storage/sage/
  semantic_chunk/

models/sage_segmenter/
  config.json
  mlp.pt
  metrics.json
```

## 4. 实现范围

第一版只实现可复现实验闭环：

1. 从现有 PDF chunk metadata 中恢复每篇论文的文本。
2. 按段落和句子构造训练样本。
3. 使用 AnglE embedding 生成相邻句子向量。
4. 训练一个小型 MLP 二分类器。
5. 使用 MLP 对全文相邻句子打分。
6. 根据阈值生成 semantic chunks。
7. 构建 FAISS index。
8. 使用现有 QA 和诊断脚本比较固定 chunk 与 semantic chunk。

第一版不做：

- 不训练 propositionizer；
- 不改动前端；
- 不改动正式 QA 文件；
- 不改动现有 `experiments/densex_corpus/`；
- 不把 SAGE 切分直接接入默认 `app.py index` 流程。

## 5. 模型选择

### 5.1 推荐默认模型

论文提到 Angle-optimized text embeddings，因此默认使用：

```text
WhereIsAI/UAE-Large-V1
```

原因：

- 它是 AnglE 系列公开 embedding 模型；
- Hugging Face model card 支持 `sentence-transformers` 加载；
- 和论文提到的 Angle-optimized embedding 思路最接近。

### 5.2 本地快速测试模型

如果机器资源不足或无法下载 UAE-Large-V1，可先使用当前项目已使用过的：

```text
BAAI/bge-small-en-v1.5
```

但实验记录中必须区分：

```text
semantic_chunk_angle
semantic_chunk_bge_small
```

因为这两者不能混为同一个 SAGE 复现实验。

## 6. 文件设计

### 6.1 新增模块

```text
sage_segmenter/
  __init__.py
  sentence_utils.py
  dataset.py
  model.py
  segmenter.py
```

职责：

- `sentence_utils.py`: 段落切分、句子切分、文本清洗。
- `dataset.py`: 从段落构造相邻句子训练样本。
- `model.py`: MLP 模型和 feature augmentation。
- `segmenter.py`: 加载 embedding model + MLP，对文本执行语义切分。

### 6.2 新增实验脚本

```text
experiments/sage_build_pairs.py
experiments/sage_train_segmenter.py
experiments/sage_prepare_corpus.py
experiments/sage_build_index.py
experiments/run_sage_semantic_chunk_sweep.py
```

职责：

- `sage_build_pairs.py`: 构造训练集、验证集。
- `sage_train_segmenter.py`: 训练 MLP。
- `sage_prepare_corpus.py`: 对论文生成 semantic chunk 语料。
- `sage_build_index.py`: 对 semantic chunk 建 FAISS index。
- `run_sage_semantic_chunk_sweep.py`: 和固定 chunk 做 QA 对比。

### 6.3 新增测试

```text
tests/test_sage_sentence_utils.py
tests/test_sage_dataset.py
tests/test_sage_model.py
tests/test_sage_segmenter.py
```

重点测试：

- 句子切分不会产生空句子；
- 同段相邻句子生成正样本；
- 跨段句子生成负样本；
- feature augmentation 维度为 `embedding_dim * 4`；
- 阈值低于 `ss` 时会切分；
- semantic chunk 输出 schema 与现有 DenseX corpus schema 兼容。

## 7. 数据格式

### 7.1 训练样本 JSONL

```json
{"id":"pair_000001","source":"data/ACD.pdf","page":3,"s1":"...","s2":"...","label":1}
```

字段含义：

- `id`: 样本 ID；
- `source`: PDF 来源；
- `page`: 页码，可为空；
- `s1`: 前一句；
- `s2`: 后一句；
- `label`: 是否应属于同一语义 chunk。

### 7.2 semantic chunk JSONL

```json
{"id":"semantic_chunk::ACD_p3_sc12","granularity":"semantic_chunk","text":"...","source":"data/ACD.pdf","page":3,"parent_chunk_id":"ACD_p3_semantic_12","paper_title":"ACD","segment_score_min":0.62,"sentence_count":4}
```

字段尽量兼容现有 DenseX corpus：

- `id`
- `granularity`
- `text`
- `source`
- `page`
- `parent_chunk_id`
- `paper_title`

额外保留：

- `segment_score_min`: chunk 内相邻句子的最低 same-chunk score；
- `sentence_count`: chunk 包含的句子数。

## 8. 训练方案

第一版训练 MLP，不微调 embedding model。

原因：

- 成本低；
- 对 CPU/普通服务器更友好；
- 可以先判断语义分割是否对 retrieval 有收益；
- 避免引入过多变量。

模型结构：

```text
input_dim = embedding_dim * 4
hidden_dim = 256
dropout = 0.1
output_dim = 1
activation = sigmoid
loss = BCEWithLogitsLoss 或 MSELoss
```

第一版默认使用 MSE，以便更贴近 SAGE 论文中将 same-chunk score 作为连续分数拟合的做法。BCE 先保留为后续对比实验选项：

```text
--loss mse
--loss bce
```

默认建议：

```text
--loss mse
```

## 9. 切分规则

输入一篇论文或一页文本后：

1. 先按段落切分；
2. 段落内按句子切分；
3. 对相邻句子 `(s_i, s_{i+1})` 计算 same-chunk score；
4. 如果 score 小于阈值 `ss`，则在两句之间切开；
5. 如果当前 semantic chunk 超过最大 token 限制，也强制切开；
6. 如果当前 semantic chunk 低于最小 token 限制，优先和后一段合并，但不能跨越明显标题或章节边界。

默认参数：

```text
--threshold 0.55
--min-chars 120
--max-chars 1200
```

后续实验可扫描：

```text
--threshold 0.45,0.50,0.55,0.60,0.65
```

## 10. 实验对比

第一轮只比较：

```text
chunk
semantic_chunk
```

不和 sentence/proposition 混在一起，避免指标解释过于复杂。

推荐实验：

```text
questions: evaluation/questions.jsonl
granularities: chunk,semantic_chunk
budgets: 300,500,1000,1500
top_k: 50
strategy: baseline
```

重点看以下指标：

- `parent_candidate_evidence_recall`: evidence 是否进入候选集合；
- `first_evidence_parent_rank`: 第一个 evidence chunk 的排名；
- `selected_evidence_recall`: evidence 是否进入最终 selected context；
- `answer_recall`: 最终答案是否覆盖 gold items；
- `context_tokens`: 实际传给 LLM 的 token 数量。

对于当前 ACDepth 问题，最关键的是：

```text
semantic_chunk 是否能降低 first_evidence_parent_rank，并提高 selected_evidence_recall。
```

如果 evidence 已经能进入候选集但 selected 仍然丢失，说明问题主要不在 segmentation，而在 parent aggregation 后排序和 selected context 选择。

## 11. 推荐执行顺序

### 阶段 1：只写基础模块和测试

```bash
python -m unittest tests/test_sage_sentence_utils.py
python -m unittest tests/test_sage_dataset.py
python -m unittest tests/test_sage_model.py
python -m unittest tests/test_sage_segmenter.py
```

目标：保证语义分割模块本身可靠。

### 阶段 2：构造训练样本

```bash
python experiments/sage_build_pairs.py \
  --metadata storage/metadata.json \
  --output experiments/sage_pairs/pairs.jsonl \
  --source-contains "ACD" \
  --max-pairs 2000
```

目标：先用 ACDepth 做小规模 smoke test。

### 阶段 3：训练轻量 MLP

```bash
python experiments/sage_train_segmenter.py \
  --pairs experiments/sage_pairs/pairs.jsonl \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --output-dir models/sage_segmenter_angle \
  --epochs 3 \
  --batch-size 16 \
  --loss mse \
  --device auto
```

目标：得到可加载的 `mlp.pt` 和训练指标。

### 阶段 4：生成 semantic chunk corpus

```bash
python experiments/sage_prepare_corpus.py \
  --metadata storage/metadata.json \
  --model-dir models/sage_segmenter_angle \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --output experiments/sage_corpus/semantic_chunk.jsonl \
  --threshold 0.55
```

目标：生成新的语义 chunk 语料，不覆盖原有 corpus。

### 阶段 5：构建对比 index

```bash
python experiments/sage_build_index.py \
  --corpus experiments/densex_corpus/chunk.jsonl \
  --index-dir storage/sage/chunk \
  --embedding-model WhereIsAI/UAE-Large-V1

python experiments/sage_build_index.py \
  --corpus experiments/sage_corpus/semantic_chunk.jsonl \
  --index-dir storage/sage/semantic_chunk \
  --embedding-model WhereIsAI/UAE-Large-V1
```

目标：得到同一个 embedding model 下的固定 chunk index 和 semantic chunk index。注意不要直接拿旧的 `storage/densex/chunk` 与 `storage/sage/semantic_chunk` 对比，除非两者使用的是同一个 embedding model。

### 阶段 6：跑对比实验

```bash
python experiments/run_sage_semantic_chunk_sweep.py \
  --questions evaluation/questions.jsonl \
  --question-ids always_clear_depth_contributions,always_clear_depth_eval_datasets,always_clear_depth_ablation_components,always_clear_depth_sota_comparison_methods \
  --embedding-model WhereIsAI/UAE-Large-V1 \
  --budgets 300,500,1000,1500 \
  --top-k 50 \
  --run-label-prefix sage_semantic_v1
```

目标：判断 semantic chunk 是否优于固定 chunk。

## 12. 成功标准

第一版不要求最终 answer recall 一定提升。更重要的是定位 semantic segmentation 是否改善 retrieval。

优先级从高到低：

1. `first_evidence_parent_rank` 下降；
2. `selected_evidence_recall` 上升；
3. 相同 `context_tokens` 下 `answer_recall` 上升；
4. 达到相同 `answer_recall` 所需的 `context_tokens` 下降。

如果只看到 `parent_candidate_evidence_recall` 上升，但 `selected_evidence_recall` 不上升，则下一步应该优化 selected context 选择，而不是继续改分割模型。

## 13. 风险和注意事项

1. SAGE 官方没有公开可直接使用的完整 segmentation model 权重，因此需要自己训练 MLP。
2. `WhereIsAI/UAE-Large-V1` 约 0.3B 参数，比 `bge-small-en-v1.5` 更重，CPU 可运行但速度会慢。
3. 如果训练数据只来自 PDF 抽取文本，段落边界可能有噪声，需要保留人工检查样本。
4. 如果 semantic chunk 过短，可能类似 sentence 检索，证据不完整。
5. 如果 semantic chunk 过长，可能退化成固定 chunk，token efficiency 变差。
6. 所有实验必须记录 embedding model、threshold、max chunk 长度，否则结果无法比较。

## 14. 和现有研究问题的关系

当前 ACDepth 诊断已经说明：部分 evidence 可以进入 parent candidates，但进入 selected context 不稳定。SAGE 语义分割主要优化的是 corpus segmentation，也就是索引前的 chunk 构造。

因此它可能解决的问题是：

- 固定 chunk 把关键证据切断；
- chunk 语义不完整导致 embedding 排名低；
- evidence 分散在多个固定 chunk 中，导致 selected context 不容易同时覆盖。

它不直接解决的问题是：

- parent aggregation 后 evidence chunk 排名靠后；
- selected context 在预算内选错 chunk；
- query 前缀导致相似度偏移；
- LLM 对已经进入 context 的 evidence 理解错误。

所以复现 SAGE segmentation 后，仍然需要用现有诊断指标判断问题到底发生在：

```text
segmentation -> retrieval ranking -> parent aggregation -> selected context -> LLM answer
```

哪一个阶段。

## 15. 参考资料

- SAGE: A Framework of Precise Retrieval for RAG, arXiv:2503.01713
- AnglE-optimized Text Embeddings, arXiv:2309.12871
- WhereIsAI/UAE-Large-V1 Hugging Face model card
