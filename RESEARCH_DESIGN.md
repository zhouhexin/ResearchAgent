# ResearchAgent 固定 Token Budget 信息选择研究设计

## 研究目标

本项目的核心研究问题是：

```text
在固定长度 token budget 下，如何从候选文档中选择、压缩和组织上下文，使 LLM 获得最准确的信息并生成最准确的答案？
```

因此，系统不只是一个普通 RAG 助手，而是一个用于比较不同 context selection / allocation / compression 策略的实验框架。

## 当前已有能力

### 1. 文档加载与切分

位置：`chunking/chunker.py`

当前支持：

- `.txt`
- `.md`
- `.markdown`
- `.pdf`

PDF 会按页抽取文本，并保留页码元数据：

```json
{
  "id": "paper_p1_chunk_0",
  "text": "...",
  "source": "data/paper.pdf",
  "page": 1,
  "start": 0,
  "end": 500
}
```

当前配置：

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
```

这些参数会影响召回质量、上下文完整性和 token 利用效率。

### 2. 向量检索

位置：

- `retrieval/embed.py`
- `retrieval/faiss_store.py`
- `retrieval/retriever.py`

当前流程：

```text
query -> embedding
chunk -> embedding
FAISS inner product search
return top_k chunks
```

默认配置：

```python
TOP_K = 8
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

向量库持久化在：

```text
storage/index.faiss
storage/metadata.json
```

注意：当前 top-k 只表示向量相似度靠前，不保证每个 chunk 都直接回答问题。

### 3. 上下文分配策略

位置：

- `allocation/baseline.py`
- `allocation/dynamic_budget.py`
- `allocation/rerank_budget.py`
- `scoring/rerank.py`

当前已有三种策略。

#### Baseline

按检索顺序依次放入上下文，直到达到 token budget。

```text
FAISS top-k order
-> add chunk if budget allows
-> stop when budget filled
```

优点：

- 简单
- 可作为所有策略的对照组

缺点：

- 不考虑冗余
- 不考虑信息密度
- 不做 query-aware compression

#### Dynamic Budget

使用 relevance、density 和 redundancy 进行重排选择：

```python
final_score = 0.6 * relevance + 0.3 * density - 0.1 * redundancy
```

其中：

- relevance：来自向量相似度
- density：当前使用词汇多样性、数字、标点等简单启发式
- redundancy：当前使用 Jaccard lexical overlap

优点：

- 开始考虑 token budget 下的选择效率
- 能减少部分重复 chunk

缺点：

- density 是粗糙启发式
- relevance 依赖 embedding，不一定等价于 answer relevance
- 没有 reranker
- 没有 token-level sentence selection

#### Rerank Budget

先对 FAISS 召回的 top-k chunks 做 query-aware rerank，再按 rerank 后顺序填充 token budget。

当前 rerank score 组合：

```python
rerank_score = (
    0.45 * semantic_relevance
    + 0.30 * keyword_overlap
    + 0.20 * bm25_like_score
    + 0.05 * density
)
```

其中：

- semantic_relevance：来自 FAISS similarity
- keyword_overlap：query terms 在 chunk 中的覆盖率
- bm25_like_score：轻量词频匹配分数
- density：信息密度启发式

优点：

- 无额外模型依赖
- 比纯 embedding 更关注 query 中的显式关键词
- 对“有哪些论文使用 diffusion”这类关键词明显的问题更稳

缺点：

- 不是 cross-encoder reranker
- 对同义改写、隐含相关性判断能力有限
- 中文分词目前是字符级近似

### 4. Prompt 构造

位置：`prompt/builder.py`

当前 prompt 会把选中的上下文编号：

```text
[1] source: ...
chunk text

[2] source: ...
chunk text
```

并要求模型：

- 只基于给定资料回答
- 使用资料编号引用
- 资料不足时明确说明
- 不编造事实

### 5. LLM 调用与 Token Usage 记录

位置：`llm/minimax_client.py`

当前使用 MiniMax OpenAI-compatible API。

每次非 dry-run 调用会记录：

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `model`

如果 API 没有返回 usage，则对应字段为空。

### 6. 当前评估指标

位置：

- `evaluation/metrics.py`
- `evaluation/logger.py`

当前已有：

#### citation_count

统计答案中出现了多少个 `[1]`、`[2]` 形式的引用。

#### context_utilization

计算被答案引用的 context 数量占总 context 数量的比例：

```text
context_utilization = cited_context_count / selected_context_count
```

示例：

```text
selected contexts = [1], [2], [3], [4]
answer cites = [1], [3]
context_utilization = 2 / 4 = 0.5
```

注意：该指标只检查是否引用了编号，不判断引用是否正确，也不是 token-level 利用率。

#### citation_validity_ratio

计算 LLM 答案中引用的 chunk 编号，有多少属于本次请求实际提供的上下文编号范围：

```text
citation_validity_ratio = valid_citation_count / all_citation_count
```

示例：

```text
selected contexts = [1], [2], [3]
answer cites = [1], [3], [9]
citation_validity_ratio = 2 / 3 = 0.667
invalid_citation_indices = [9]
```

该指标回答的是：

```text
LLM 返回引用的 chunk 中，有多少是本次请求 chunk 里的有效编号？
```

注意：它只判断引用编号是否有效，不判断引用内容是否真正支撑答案。后者需要 citation correctness / faithfulness judge。

## 当前实验入口

### 建立知识库

```bash
python app.py index --docs ./data
```

### 单次问答

```bash
python app.py ask --query "问题"
```

### 指定策略和候选池

```bash
python app.py ask --query "问题" --strategy baseline --top-k 20
python app.py ask --query "问题" --strategy dynamic --top-k 20
```

### 固定上下文预算

```bash
python app.py ask --query "问题" --context-budget 2000
```

## 当前方法的主要局限

### 1. Token 估算不精确

当前上下文预算使用粗略估算：

```python
estimate_tokens(text) = len(text) // 2
```

这对中英文混合文本只是近似，不等于实际模型 tokenizer。

影响：

- 预算控制不精确
- 不同策略的 token 使用量可能不可比
- 难以做严格固定 budget 实验

### 2. 缺少 Reranker

当前只用 embedding 相似度取 top-k。

问题：

- 语义相近但不回答问题的 chunk 可能被选中
- 真正包含答案的 chunk 可能排在 top-k 之外
- dynamic allocation 只能在已有候选池中重排，不能纠正召回不足

### 3. 缺少更细粒度的信息选择

当前选择单位是 chunk。

问题：

- 一个 chunk 里可能只有一句话相关，其余 token 浪费
- 无法在固定 budget 下做 sentence-level 或 claim-level selection

### 4. 压缩模块仍是占位

当前 `compression/summarize.py` 只是截断文本。

问题：

- 没有 query-aware summary
- 没有 evidence-preserving compression
- 无法比较压缩前后答案准确率变化

### 5. 评估指标不足

当前没有直接记录：

- selected chunk ids
- selected context text
- context token count
- redundancy score
- answer accuracy
- faithfulness
- gold answer
- judge score

这会限制实验复现和策略对比。

## 建议新增与优化方向

### 优先级 P0：让实验可严格比较

#### 1. 精确 Token 计数

当前状态：已实现第一版。

位置：

- `evaluation/token_counter.py`

实现方式：

- 优先使用 `tiktoken` 的 `cl100k_base`
- 如果未安装 `tiktoken`，退化为确定性的正则 token 估算
- 记录 `context_tokens` 和 `prompt_token_estimate`

新增模块：

```text
evaluation/token_counter.py
```

目标：

- 记录 selected context tokens
- 记录 full prompt tokens
- 让所有策略在同一个 budget 下公平比较

可选实现：

- 如果模型 tokenizer 可用，使用 MiniMax 对应 tokenizer
- 如果不可用，先用 `tiktoken` 作为近似
- 同时保留 API 返回的真实 `prompt_tokens`

已新增日志字段：

```csv
context_tokens,prompt_tokens,completion_tokens,total_tokens
```

#### 2. 记录被选中的上下文

当前状态：已实现第一版。

位置：

- `evaluation/logger.py`
- `experiments/runs/*.json`

每次 `ask` 都会保存 run detail JSON，包括 dry-run。

建议新增：

```text
experiments/runs/
  run_001.json
```

每次运行保存：

```json
{
  "query": "...",
  "strategy": "dynamic",
  "top_k": 20,
  "budget": 2000,
  "selected_chunks": [
    {
      "id": "...",
      "source": "...",
      "page": 3,
      "score": 0.82,
      "estimated_tokens": 240,
      "text": "..."
    }
  ],
  "prompt": "...",
  "answer": "...",
  "usage": {
    "prompt_tokens": 1800,
    "completion_tokens": 300,
    "total_tokens": 2100
  }
}
```

这对于后续分析“为什么某个策略失败”非常重要。

#### 3. 批量实验 runner

新增：

```text
experiments/run_budget_sweep.py
```

支持批量比较：

```text
strategies = baseline, dynamic
top_k = 8, 20, 50
budgets = 500, 1000, 2000, 4000
```

输出：

```text
results.csv
run_details/*.json
```

### 优先级 P1：改进信息选择质量

#### 4. Reranker

当前状态：已实现轻量 reranker 第一版。

位置：

```text
scoring/rerank.py
allocation/rerank_budget.py
```

当前实现是无额外模型依赖的 lexical + semantic reranker。后续仍可升级为 cross-encoder 或 LLM judge reranker。

新增：

```text
scoring/rerank.py
```

推荐流程：

```text
FAISS top 50
-> reranker score(query, chunk)
-> select top candidates under budget
```

可选方法：

- 本地 cross-encoder reranker
- BAAI/bge-reranker-base
- MiniMax judge reranker
- 简单 LLM 打分：0-5 分判断 chunk 是否回答 query

实验价值：

```text
比较 embedding-only vs embedding + reranker 在固定 budget 下的准确率
```

#### 5. MMR 多样性选择

新增：

```text
allocation/mmr.py
```

目标：

在保证相关性的同时减少重复信息：

```text
score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
```

适合比较：

```text
baseline vs dynamic vs MMR
```

#### 6. Sentence-level Selection

新增：

```text
compression/sentence_select.py
```

流程：

```text
chunk -> split sentences
-> score each sentence against query
-> select best sentences under budget
```

研究价值：

- 固定 token 下减少无关文本
- 提高 evidence density
- 适合回答事实型和列表型问题

#### 7. Query-aware Compression

新增：

```text
compression/query_aware.py
```

目标：

不是通用摘要，而是针对 query 压缩：

```text
请保留回答该问题所需的事实、定义、数字、方法名和证据。
删除无关背景。
```

实验价值：

```text
raw chunk vs truncated chunk vs query-aware summary
```

### 优先级 P2：增强评估体系

#### 8. Gold Answer 数据集

新增：

```text
evaluation/questions.jsonl
```

格式：

```json
{"id": "q1", "query": "...", "gold_answer": "...", "evidence_sources": ["..."]}
```

可以支持：

- 人工标注准确率
- LLM-as-judge
- evidence recall

#### 9. LLM-as-Judge

新增：

```text
evaluation/judge.py
```

评估维度：

- answer correctness
- faithfulness to provided context
- completeness
- citation correctness

输出：

```json
{
  "correctness": 4,
  "faithfulness": 5,
  "completeness": 3,
  "comment": "..."
}
```

#### 10. Redundancy 与 Density 指标升级

当前 density 和 redundancy 是启发式。

可优化为：

- embedding-based redundancy
- sentence overlap
- named entity density
- query term coverage
- evidence sentence ratio

## 推荐实验矩阵

### 实验 1：Budget Sweep

目标：

```text
比较不同策略在固定 token budget 下的准确率变化。
```

变量：

```text
strategy = baseline, dynamic
budget = 500, 1000, 2000, 4000
top_k = 20
```

观察：

- accuracy 是否随 budget 增长
- dynamic 是否在小 budget 下优于 baseline
- context_utilization 是否随 budget 下降

### 实验 2：Candidate Pool Size

目标：

```text
研究 top_k 增大是否提高最终答案准确率。
```

变量：

```text
top_k = 8, 20, 50, 100
budget = 2000
strategy = dynamic
```

观察：

- top_k 增大是否带来更好候选
- 是否引入更多噪声
- reranker 是否能缓解噪声

### 实验 3：Chunk Size Ablation

目标：

```text
研究 chunk 粒度对固定 budget 效率的影响。
```

变量：

```text
CHUNK_SIZE = 300, 500, 800, 1200
CHUNK_OVERLAP = 50, 80, 150
```

注意：

修改 chunk 参数后必须重新建索引。

### 实验 4：Compression Ablation

目标：

```text
比较原文 chunk、截断、句子选择、query-aware summary 的效果。
```

变量：

```text
compression = none, truncate, sentence_select, query_aware_summary
budget = 1000, 2000
```

观察：

- 压缩是否提高 answer accuracy
- 压缩是否损失关键证据
- token 使用是否更高效

### 实验 5：Reranker Ablation

目标：

```text
比较是否加入 reranker 对最终答案准确率的影响。
```

变量：

```text
retrieval = embedding_only, embedding_plus_reranker
top_k_initial = 50
final_budget = 2000
```

## 推荐实现路线

### Step 1：完善日志

当前状态：已完成 P0 第一版。

优先新增：

- selected chunk ids
- selected context tokens
- context text snapshot
- prompt snapshot
- answer
- usage

这是后续所有实验分析的基础。

### Step 2：精确 token budget

当前状态：已完成 P0 第一版。

`allocation/baseline.py` 和 `allocation/dynamic_budget.py` 已统一使用 `evaluation/token_counter.py` 的计数结果控制 budget，日志中的 `context_tokens` 也来自同一套 counter。

注意：当 MiniMax 没有公开 tokenizer 时，当前实现使用 `tiktoken` 的 `cl100k_base` 作为近似。API 返回的真实 `prompt_tokens` 仍会记录到 CSV。

### Step 2.5：P0 budget sweep runner

当前状态：已完成。

位置：

```text
experiments/run_budget_sweep.py
```

示例：

```bash
python experiments/run_budget_sweep.py \
  --query "问题" \
  --strategies baseline,dynamic \
  --budgets 500,1000,2000,4000 \
  --top-k 20
```

该 runner 用同一 query、同一 top-k、不同 strategy 和 budget 批量执行实验。每次运行都会写入：

- `experiments/results.csv`
- `experiments/runs/*.json`

### Step 3：新增 MMR allocation

作为比 dynamic 更标准的多样性选择 baseline。

### Step 4：新增 batch experiment runner

支持一次性跑不同：

- strategy
- top_k
- budget
- chunk_size
- compression

### Step 5：新增 reranker

先用本地 reranker 或 LLM judge reranker 都可以，关键是建立可对比实验。

### Step 6：新增 compression 策略

从 sentence-level selection 开始，比直接 LLM 摘要更可控。

## 不建议优先做的部分

### 论文元数据抽取层

论文元数据抽取对“论文助手”有价值，但对当前研究问题不是优先项。

它可以作为后续一种对照策略：

```text
metadata-enhanced selection
```

但不应该先于：

- 精确 token 计数
- selected context logging
- batch experiment runner
- reranker
- compression strategy

## 当前项目定位

当前系统已经具备一个最小可用实验闭环：

```text
PDF / text documents
-> chunking
-> embedding + FAISS retrieval
-> baseline / dynamic allocation
-> prompt construction
-> MiniMax answer
-> usage + simple metrics logging
```

下一阶段应将其从“可问答系统”升级为“可复现实验框架”：

```text
fixed budget
controlled strategy
recorded selected context
repeatable evaluation
comparable metrics
```
