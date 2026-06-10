# Relevance 准确性诊断与提升步骤

本文档用于分析 fine-to-chunk 实验中 relevance 不高的问题，重点回答两个问题：

```text
1. 正确证据为什么没有进入 context？
2. 应该从 retrieval、aggregation、selection 哪一步继续优化？
```

当前建议对比：

```text
之前结果：experiments/qa_parent_v1_densex_results.csv
本次结果：experiments/qa_chunkaware_dedup_v1_results.csv
```

其中 `qa_chunkaware_dedup_v1` 使用的是：

```text
原始 sentence/proposition 索引
-> fine-top-m=300
-> parent chunk 内 exact dedup
-> fine-to-chunk aggregation
```

## 1. 先不要只看最终 F1

最终 F1 只能说明答案对不对，不能说明问题出在哪里。需要按下面三层观察：

```text
fine hit 分数
parent aggregation 分数
selected context
```

如果 ACDepth 的 F1 没提升，不能直接判断 dedup 无效。要先确认：

```text
正确 evidence 是否被 sentence/proposition 检索到？
正确 parent chunk 是否被聚合出来？
正确 parent chunk 是否进入 token budget 内的 context？
```

## 2. 观察 Fine Hit 分数

第一层看 sentence/proposition 检索结果。

建议导出每个问题的 top fine hits，字段包括：

```text
question_id
granularity
rank
score
parent_chunk_id
source
page
text
matched_gold_ids
```

重点观察：

```text
gold evidence 是否出现在 top 300？
gold evidence 排名大概是多少？
gold evidence 的 score 和前几名噪声 score 差多少？
top 300 是否被同一篇论文、同一页、同一类重复内容占满？
```

判断方式：

```text
如果 gold evidence 不在 top 300：
    问题主要在 embedding retrieval。

如果 gold evidence 在 top 300，但排名很低：
    需要 query expansion 或 rerank。

如果 top 300 有 gold evidence，但 parent chunk 没排上去：
    问题在 parent aggregation。
```

## 3. 观察 Parent Aggregation 分数

第二层看 fine hits 如何聚合回 parent chunk。

当前 chunk-aware dedup 的聚合流程是：

```text
fine hits
-> 按 parent_chunk_id 分组
-> 每个 parent chunk 内部 exact dedup
-> 计算 parent chunk score
```

当前默认聚合公式：

```text
aggregate_score = max_child_score + 0.1 * sum(top_n_child_scores)
```

建议导出 parent candidates，字段包括：

```text
question_id
granularity
parent_rank
aggregate_score
max_child_score
top_child_score_sum
matched_child_count
deduplicated_child_count
top_child_ids
top_child_texts
source
page
matched_gold_ids
```

重点观察：

```text
正确 chunk 的 max_child_score 是否偏低？
噪声 chunk 是否因为命中数量多被抬高？
deduplicated_child_count 是否明显小于 matched_child_count？
正确 chunk 是否排在 parent_top_k 之外？
正确 chunk 进入 parent candidates 后，是否又被 token budget selection 挤掉？
```

判断方式：

```text
如果正确 chunk 的 max_child_score 很低：
    fine retrieval 或 query 表达有问题。

如果噪声 chunk matched_child_count 很高：
    当前聚合公式可能过度奖励重复命中。

如果正确 chunk 排名在 parent_top_k 之外：
    需要调聚合公式或增加 fine-top-m。
```

## 4. 观察 Selected Context

第三层看最终进入 LLM 的 context。

建议导出 selected chunks，字段包括：

```text
question_id
granularity
budget
selected_rank
score
tokens
source
page
text
matched_gold_ids
```

判断方式：

```text
如果 gold chunk 已进入 parent candidates，但没有进入 selected context：
    问题在 token budget allocation。

如果 selected context 有 gold chunk，但 LLM 没答对：
    问题可能在答案生成、prompt 或 gold alias 匹配。

如果 selected context 没有 gold chunk：
    不应该优先调 LLM，应该先提升 retrieval/aggregation/selection。
```

## 5. Relevance 提升方向

建议按以下顺序优化，不要同时改多个变量。

### 5.1 调整 Fine-Top-M

当前本次实验使用：

```text
fine-top-m=300
```

建议继续对比：

```text
fine-top-m=150
fine-top-m=300
fine-top-m=500
fine-top-m=800
```

观察指标：

```text
retrieved_gold_recall
selected_gold_recall
selected_relevance_precision
answer_f1
token_efficiency
```

理想情况：

```text
retrieved_gold_recall 上升
selected_gold_recall 上升
selected_relevance_precision 不明显下降
answer_f1 上升
```

如果 top-M 增大后：

```text
retrieved_gold_recall 上升，但 selected_relevance_precision 下降
```

说明召回了更多证据，但噪声也增加了，需要 rerank 或更好的 aggregation。

### 5.2 调整 Parent Aggregation 公式

当前公式：

```text
score = max_child_score + 0.1 * sum(top_n_child_scores)
```

可以测试以下版本：

```text
max only:
score = max_child_score

max + smaller child sum:
score = max_child_score + 0.05 * sum(top_3_scores)

max + coverage:
score = max_child_score + 0.05 * log(1 + deduplicated_child_count)

top-k average:
score = average(top_3_scores)
```

观察重点：

```text
是否减少噪声 chunk 排名前置？
是否让 gold parent chunk 排名上升？
是否提升 selected_gold_recall？
```

### 5.3 增加 Source / Paper Diversity

如果 top parent chunks 被同一篇论文占满，可以增加来源多样性限制：

```text
每篇论文最多保留 N 个 parent chunks
```

例如：

```text
max_chunks_per_source = 5
```

适用场景：

```text
问题需要找多篇论文；
top chunks 被同一篇论文或同一页大量占据；
selected context 缺少其他论文证据。
```

### 5.4 Query Expansion

ACDepth 的问题更像是 query 和 evidence 表达不一致。可以先不用 LLM，手工给 QA 对增加
`retrieval_query` 字段。

示例：

```text
原始问题：
Always Clear Depth 论文的主要贡献包括哪些？

retrieval_query：
Always Clear Depth contributions proposed method components adverse weather robust monocular depth estimation
```

数据集问题示例：

```text
原始问题：
Always Clear Depth 使用了哪些数据集进行实验？

retrieval_query：
Always Clear Depth evaluated datasets experiments benchmark adverse weather dataset
```

原则：

```text
回答仍使用原始 query；
检索使用 retrieval_query；
评估仍按原 QA gold items。
```

### 5.5 Section-Aware Boost

论文 QA 的证据通常出现在固定 section。

常见映射：

```text
贡献问题：
Abstract / Introduction / Contributions / Conclusion

数据集问题：
Experiments / Datasets / Experimental Setup

消融问题：
Ablation Study / Ablation Experiments

对比方法问题：
Comparison with SoTAs / Experiments / Baselines
```

如果没有 section metadata，可以用关键词近似：

```text
contribution / propose / introduce / main contributions
dataset / benchmark / evaluated on
ablation / components / variants
comparison / baseline / state-of-the-art
```

### 5.6 Rerank

如果正确 evidence 已经出现在 top-M，但排名不够高，可以加 rerank。

先从轻量规则 rerank 开始：

```text
final_score =
    embedding_score
    + query_keyword_overlap
    + paper_title_match
    + section_keyword_match
    - source_overuse_penalty
```

暂时不建议一开始就上复杂 cross-encoder。先确认规则 rerank 是否能提升
`selected_gold_recall` 和 `selected_relevance_precision`。

## 6. 当前实验结论

基于目前结果：

```text
DepthDark：
chunk-aware dedup 有明显帮助，尤其是 sentence-to-chunk @500/@1000。

ACDepth：
提升不明显，主要问题是正确 evidence 没有被 fine retrieval 找到。
```

因此下一步建议分开处理：

```text
DepthDark：
继续优化 fine-top-m、aggregation score、token efficiency。

ACDepth：
优先做 retrieval relevance 诊断，重点检查 fine hits 和 query/evidence 表达差异。
```

## 7. 推荐下一步工具

建议新增一个诊断脚本：

```text
experiments/inspect_densex_relevance.py
```

目标输入：

```powershell
python experiments\inspect_densex_relevance.py `
  --question-id always_clear_depth_contributions `
  --granularity sentence `
  --top-m 300
```

目标输出：

```text
experiments\diagnostics\fine_hits_<question_id>_<granularity>.csv
experiments\diagnostics\parent_candidates_<question_id>_<granularity>.csv
```

先通过这个脚本确认 ACDepth 的瓶颈，再决定是否实现 query expansion、section-aware boost
或 rerank。
