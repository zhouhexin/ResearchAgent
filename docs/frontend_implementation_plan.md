# 前端实现计划

本文档记录如何基于当前 ResearchAgent 项目搭建一个可解释的知识库问答前端。

## 目标

构建一个前端页面，先支持关键词检索，后续扩展为可解释的论文知识库问答系统。

这个前端不应该只展示最终回答，还应该展示：

```text
检索到的证据
进入 context 的内容
token 使用情况
实验结果指标
```

## 产品定位

第一版应该定位为内部研究工具，而不是普通聊天机器人。

核心价值是让完整流程可观察：

```text
query -> retrieval evidence -> selected context -> answer -> metrics
```

这样可以帮助判断 retrieval、context selection、compression、granularity 等环节是否真的有效。

## 推荐架构

建议在前端和现有 Python 实验管线之间增加一个轻量 API 后端。

```text
Frontend UI
  -> FastAPI backend
    -> storage/metadata.json 关键词检索
    -> FAISS semantic retrieval
    -> app.answer_query 问答流程
    -> experiments/*.csv 实验结果查看
```

推荐前端技术栈：

```text
React + Vite + TypeScript
Ant Design
```

Ant Design 比较适合这个项目，因为当前页面更像实验工作台，需要大量表单、表格、筛选器和密集结果展示。

## 实现顺序

### 1. 后端关键词检索 API

先增加一个 FastAPI 后端，并实现关键词检索接口：

```text
POST /search
```

第一版行为：

- 读取 `storage/metadata.json`。
- 在 `text`、`source`、`page` 和论文标题中检索关键词。
- 返回命中的 chunks。
- 对命中的关键词进行高亮。
- 返回 `source`、`page`、`score`、`chunk_id` 和文本片段。

这一步不调用 LLM，只用于验证语料是否可检索、证据是否存在。

### 2. 前端关键词检索页面

创建第一个前端页面：

```text
关键词输入框
搜索按钮
结果列表 / 表格
关键词高亮
source 和 page 信息
```

建议提供的控制项：

- keyword query
- max results
- source filter
- 是否区分大小写

这个页面用于检查知识库内容是否被正确索引，以及用户能否快速定位论文证据。

### 3. QA API

增加问答接口：

```text
POST /ask
```

该接口调用现有 `answer_query` 流程，并返回：

- final answer
- retrieved chunks
- selected chunks
- context tokens
- prompt / completion / total token usage
- details path 或 run id

第一版只暴露少量安全参数：

- `strategy`
- `top_k`
- `context_budget`
- `compression`
- `compression_stage`

### 4. 可解释问答页面

创建问答页面，建议布局为三块：

```text
左侧：问题输入和参数设置
中间：最终回答
右侧：retrieved evidence 和 selected context
```

证据面板需要展示：

- retrieved chunks
- selected chunks
- source file
- page
- score
- token estimate
- text preview

这个页面比普通聊天 UI 更适合当前项目，因为它能展示回答为什么成立，或者为什么没有被 context 支撑。

### 5. 实验结果页面

增加一个实验结果查看页面，读取 CSV 文件，例如：

```text
experiments/qa_v1_densex_summary.csv
experiments/qa_parent_v1_densex_results.csv
```

页面需要支持排序和筛选，重点展示：

- `granularity`
- `budget`
- `answer_f1`
- `answer_recall`
- `context_tokens`
- `selected_gold_recall`
- `selected_relevance_precision`
- `token_efficiency`

这个页面可以减少手动打开 CSV 的成本，也方便横向比较 chunk、sentence、proposition 和 fine-to-chunk。

### 6. Hybrid Retrieval 和高级模式

基础关键词检索和问答页面稳定后，再增加高级检索模式：

```text
keyword
semantic
hybrid
sentence-to-chunk
proposition-to-chunk
```

Hybrid retrieval 可以组合：

```text
keyword score + embedding similarity score
```

fine-to-chunk 模式复用当前 parent aggregation 逻辑：

```text
sentence/proposition retrieve -> parent chunk aggregation -> context selection
```

### 7. Evaluation Overlay

对于已经存在于 `evaluation/questions.jsonl` 中的问题，可以在回答后展示实验评估信息：

- matched gold items
- selected gold recall
- selected relevance precision
- answer F1 / recall

这部分应该作为实验调试功能，而不是面向普通用户的回答质量保证。

## 第一个里程碑

第一版可用功能应该包括：

```text
FastAPI /search
React 关键词检索页面
命中关键词高亮
source/page 展示
```

不要一开始就做聊天页面。关键词检索更简单、更容易验证，也更适合检查语料中是否存在预期证据。

## 第二个里程碑

增加：

```text
POST /ask
问答页面
retrieved/selected context 面板
token usage 展示
```

完成这一步后，系统就具备“可解释知识库问答”的基本形态。

## 第三个里程碑

增加：

```text
实验结果 CSV 查看页面
granularity 对比表格
fine-to-chunk 模式控制
```

完成这一步后，前端可以作为当前研究流程的实验分析工作台。
