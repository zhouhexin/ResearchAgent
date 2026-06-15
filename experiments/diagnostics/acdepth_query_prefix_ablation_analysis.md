# ACDepth 问题前缀消融检索分析

分析对象：

- `experiments/diagnostics/acdepth_query_ablation_summary.csv`
- `experiments/diagnostics/acdepth_query_ablation_hits.csv`

分析目的：

判断在 ACDepth 相关问题中，去掉 `Always Clear Depth ...` 这类论文名前缀后，是否有助于 top-k 检索。

## 结论

去掉论文名前缀后，对 ACDepth 的 top-k 检索有明显帮助，尤其体现在 `chunk` 和 `sentence` 粒度上。

当前诊断结果显示，`content_only` 查询相比 `original` 查询：

- gold alias 命中率从 `0.417` 提升到 `1.000`
- 平均 gold alias 命中数量从 `1.083` 提升到 `39.333`
- 首个 gold alias 命中平均排名从 `43.4` 提前到 `1.33`

这说明原始 query 中的论文名和泛化领域词会干扰 embedding 检索，使 top-k 更容易返回标题页、泛化深度估计内容，甚至其他论文的相关文本。

## 总体对比

| query variant | 样本数 | gold 命中率 | 平均 gold 命中数 | 平均首个 gold 排名 | evidence 命中率 | 平均 evidence 命中数 | 平均首个 evidence 排名 | title junk rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 12 | 0.417 | 1.083 | 43.4 | 0.333 | 4.917 | 5.0 | 0.257 |
| title_only | 12 | 0.667 | 10.500 | 22.125 | 0.250 | 0.500 | 4.0 | 0.000 |
| content_only | 12 | 1.000 | 39.333 | 1.333 | 0.333 | 1.417 | 27.0 | 0.001 |
| source_filtered | 12 | 0.333 | 5.917 | 1.250 | 0.333 | 5.500 | 3.75 | 0.079 |

## 按检索粒度对比

| granularity | variant | gold 命中率 | 平均 gold 命中数 | 平均首个 gold 排名 | evidence 命中率 | 平均 evidence 命中数 | 平均首个 evidence 排名 | title junk rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chunk | original | 0.500 | 1.250 | 21.500 | 0.000 | 0.000 | - | 0.000 |
| chunk | content_only | 1.000 | 35.750 | 1.000 | 0.000 | 0.000 | - | 0.000 |
| sentence | original | 0.500 | 1.000 | 80.000 | 0.000 | 0.000 | - | 0.000 |
| sentence | content_only | 1.000 | 38.250 | 1.500 | 0.000 | 0.000 | - | 0.000 |
| proposition | original | 0.250 | 1.000 | 14.000 | 1.000 | 14.750 | 5.000 | 0.770 |
| proposition | content_only | 1.000 | 44.000 | 1.500 | 1.000 | 4.250 | 27.000 | 0.003 |

## 具体问题对比

| question_id | granularity | original gold hits | content_only gold hits | original first rank | content_only first rank | original evidence hits | content_only evidence hits |
|---|---|---:|---:|---:|---:|---:|---:|
| always_clear_depth_ablation_components | chunk | 1 | 24 | 20 | 1 | 0 | 0 |
| always_clear_depth_ablation_components | sentence | 1 | 28 | 94 | 1 | 0 | 0 |
| always_clear_depth_ablation_components | proposition | 0 | 48 | - | 1 | 11 | 7 |
| always_clear_depth_contributions | chunk | 0 | 6 | - | 1 | 0 | 0 |
| always_clear_depth_contributions | sentence | 0 | 4 | - | 2 | 0 | 0 |
| always_clear_depth_contributions | proposition | 0 | 2 | - | 3 | 25 | 3 |
| always_clear_depth_eval_datasets | chunk | 0 | 64 | - | 1 | 0 | 0 |
| always_clear_depth_eval_datasets | sentence | 0 | 65 | - | 2 | 0 | 0 |
| always_clear_depth_eval_datasets | proposition | 0 | 67 | - | 1 | 11 | 4 |
| always_clear_depth_sota_comparison_methods | chunk | 4 | 49 | 23 | 1 | 0 | 0 |
| always_clear_depth_sota_comparison_methods | sentence | 3 | 56 | 66 | 1 | 0 | 0 |
| always_clear_depth_sota_comparison_methods | proposition | 4 | 59 | 14 | 1 | 12 | 3 |

## 现象解释

以 `always_clear_depth_contributions / chunk` 为例：

- original query：`Always Clear Depth 论文的主要贡献包括哪些？`
- content_only query：`main contributions proposed framework adverse weather robust monocular depth estimation multi tuple degradation multi granularity knowledge distillation ordinal guidance`

在 original 查询下，top-k 前几位会混入 `SSRDepth.pdf`、`Repurposing Diffusion-Based Image Generators...pdf`、`Robust Monocular Depth Estimation...pdf` 等其他论文。`ACD.pdf` 虽然也出现，但更容易命中标题页或泛化内容。

在 content_only 查询下，top1 就能命中 `ACD.pdf` 中包含贡献描述的文本块，例如 ACDepth framework、multi-granularity knowledge distillation、ordinal guidance 等相关内容。

因此，问题前缀中的论文名不适合直接作为 embedding query 的主体。它更适合作为 source filter 或 source boost 使用。

## 对 proposition 的补充说明

`proposition` 粒度下，`content_only` 同样显著提升 gold alias 命中：

- gold 命中率：`0.250` -> `1.000`
- 平均 gold 命中数：`1.000` -> `44.000`
- 平均首个 gold 排名：`14.000` -> `1.500`

但 proposition 的 evidence 命中数量从 `14.750` 降到 `4.250`，平均首个 evidence 排名从 `5.000` 变差到 `27.000`。

这说明 proposition 粒度存在额外问题：只优化 query 不能完全解决证据选择，需要继续结合 source-aware reranking、parent chunk aggregation 或 proposition 质量过滤。

更准确地说，proposition 的问题不是“找不到 evidence”。从当前诊断结果看，多个 query variant 在 proposition 粒度下都能较好命中 evidence page：

| variant | evidence 命中率 | 平均 evidence 命中数 | 平均首个 evidence 排名 | 主要问题 |
|---|---:|---:|---:|---|
| original | 1.000 | 14.750 | 5.000 | title junk 高，gold alias 命中差 |
| title_only | 0.750 | 1.500 | 4.000 | 证据少，query 不针对具体问题 |
| content_only | 1.000 | 4.250 | 27.000 | gold alias 好，但部分 evidence 排名靠后 |
| source_filtered | 1.000 | 16.500 | 3.750 | evidence 最强，但仍有一定 title junk |

因此，当前更准确的判断是：

```text
proposition 能在 top-k 中找到正确 evidence page，
但 evidence page 没有稳定转化为最终 context 中的有效答案信息。
```

## 为什么 evidence 命中好但 recall 仍低

之前 ACDepth 正式实验中，proposition 相关问题的 `answer_recall` 和 `selected_gold_recall` 很低，这和 diagnostics 中 proposition evidence 命中较好并不矛盾，原因是两个指标口径不同。

当前 diagnostics 的 `matched_evidence_refs` 是 page-level 判断：

```text
hit.source == gold_evidence.source
并且
hit.page == gold_evidence.page
```

也就是说，只要 proposition 来自标注 evidence 的同一页，就算命中 evidence。它不要求该 proposition 本身包含 gold answer alias，也不要求它进入最终 prompt。

而正式实验中的 `retrieved_gold_recall`、`selected_gold_recall` 和 `answer_recall` 分别关注：

```text
retrieved_gold_recall:
gold answer alias 是否出现在 retrieved chunks 中

selected_gold_recall:
gold answer alias 是否出现在 selected context 中

answer_recall:
LLM 最终回答是否覆盖 gold answer items
```

因此会出现这种情况：

```text
top-k proposition 命中了 evidence page
但该 proposition 不包含答案项，或没有进入 selected context
最终 answer recall 仍然接近 0
```

另外，之前 `qa_v2_densex_results.csv` 这类普通 DenseX sweep 使用的是原始问题：

```text
Always Clear Depth 的消融实验验证了哪些组件？
```

而不是 diagnostics 中表现更好的 `content_only` query。因此不能直接用 `content_only` 的诊断表现解释之前普通 sweep 的低 recall。

## 当前 parent aggregation 状态

`proposition/sentence -> parent chunk` 聚合已经实现，但只在以下脚本中生效：

```text
experiments/run_densex_parent_sweep.py
```

普通 DenseX sweep：

```text
experiments/run_densex_sweep.py
```

不会执行 parent aggregation，而是直接把 chunk、sentence 或 proposition 本身作为候选传给后续 context selection。

当前 parent aggregation 链路是：

```text
fine hits
-> 按 parent_chunk_id 聚合
-> 生成 parent_candidates
-> answer_query 使用 retrieved_chunks_override=parent_candidates
-> allocation/budget 选择 selected context
```

所以接下来不应该重复实现 `sentence/proposition -> parent chunk`，而应该重点分析和优化：

```text
parent_candidates 的排序
selected context 的选择
```

## 下一步应该判断的核心指标

为了避免指标过多，下一步建议只保留以下关键指标：

| 阶段 | 指标 | 判断内容 |
|---|---|---|
| parent aggregation 后 | `parent_candidate_gold_recall` | 正确答案项是否进入 parent candidates |
| parent aggregation 后 | `first_gold_parent_rank` | 首个正确 parent chunk 排名是否靠前 |
| selected 后 | `selected_gold_recall` | 正确答案项是否进入最终 context |
| selected 后 | `selected_evidence_recall` | 正确 evidence page 是否进入最终 context |
| 最终回答 | `answer_recall` / `answer_f1` | LLM 最终是否答出 gold answer items |

判断链路如下：

```text
parent_candidate_gold_recall 低
=> 问题在 fine retrieval、query 或 parent aggregation score。

parent_candidate_gold_recall 高，但 selected_gold_recall / selected_evidence_recall 低
=> 问题在 selected context 选择和 budget packing。

selected_gold_recall / selected_evidence_recall 高，但 answer_recall 低
=> 问题在 prompt、LLM 回答格式或 gold alias 匹配。
```

其中需要新增或重点观察的是：

```text
selected_evidence_recall
```

它用于判断最终传给 LLM 的 selected context 是否包含 gold evidence 对应的 source/page。这个指标比 top-k evidence hit 更接近最终回答链路。

## 如何提高 parent aggregation 后的排序和 selected 选择

如果正确 parent chunk 没有进入 parent candidates，应优先优化 aggregation 之前和 aggregation 本身：

```text
1. 提高 fine_top_m，扩大 sentence/proposition 初始召回池。
2. 使用 content_only query，降低论文名前缀和标题语义干扰。
3. 对目标 source 做 boost，而不是把论文名直接拼进 embedding query。
4. 调整 parent_score，使多个高质量 child hit 能提高 parent chunk 排名。
```

当前 parent score 可以继续沿用类似形式：

```text
parent_score = max(child_score) + 0.1 * sum(top_3_child_scores)
```

后续可测试更偏 evidence 的版本：

```text
parent_score =
  max_child_score
  + child_sum_weight * sum(top_k_child_scores)
  + source_match_bonus
  + query_term_overlap_bonus
  - title_junk_penalty
```

如果正确 parent chunk 已经进入 parent candidates，但没有进入 selected context，应优化 selected 阶段：

```text
1. 不只按 parent_score 顺序塞满 budget。
2. 优先选择高分、低重复、来自目标 source 的 parent chunk。
3. 限制同一 source/page 的重复 chunk 数量。
4. 对 title page、参考文献、纯表格残片等低质量 chunk 降权。
5. 在 budget 较小时优先保证 evidence-like parent chunk 进入 context。
```

更具体的 selected 策略可以是：

```text
第一步：选择 top evidence-like parent chunks。
第二步：补充高 parent_score 且与已选内容不重复的 parent chunks。
第三步：如果超出 budget，优先删除重复页、标题页、低 query overlap 的 chunk。
```

最终目标不是单纯提高 top-k 命中，而是提高：

```text
selected_evidence_recall
selected_gold_recall
answer_recall
```

## 后续建议

对于指定论文内问答，推荐采用两段式检索：

1. 用去掉论文名后的问题核心内容作为向量检索 query。
2. 用论文名、文件名或 source metadata 做过滤或加权，而不是把论文名前缀直接拼进 embedding query。

推荐形式：

```text
retrieval_query = 去掉论文名后的问题核心内容
source_constraint = Always Clear Depth / ACD.pdf
```

对于开放式问题，不应强制 source filter，但仍建议去掉模板化前缀，保留真正表达检索意图的关键词。
