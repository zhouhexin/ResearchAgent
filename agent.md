# Agent Notes

本文档用于约束后续基于 ResearchAgent 项目进行前端开发时的行为边界。

## 当前目标

基于现有 RAG 实验项目，开发第一版实验室知识问答系统前端。

第一版面向实验室其他用户时，应表现为一个简单的问答系统：

```text
用户输入问题或关键词
系统返回最终回答
```

前端只展示：

- 用户问题
- 最终回答
- 加载状态
- 错误状态
- 空输入提示
- 可选的当前会话问答历史

前端不展示：

- retrieved chunks
- selected context
- token usage
- compression info
- answer_f1 / recall / token_efficiency 等实验指标
- gold items / evidence 匹配结果

这些研究信息仍然保留在后端日志、run details 和实验 CSV 中，供个人研究使用。

## 双重产品定位

该系统有两层定位：

```text
对实验室其他用户：知识问答系统
对项目作者个人：基于当前研究方法的实验工具
```

因此，普通用户页面必须保持简洁，不暴露复杂实验参数和检索细节。

## 不要改动实验主流程

除非用户明确要求，不要修改以下实验核心流程的行为：

- `app.py` 中现有 `index` / `ask` CLI 行为
- `answer_query` 的默认检索、allocation、compression、logging 行为
- `experiments/run_budget_sweep.py`
- `experiments/run_densex_sweep.py`
- `experiments/run_densex_parent_sweep.py`
- `experiments/evaluate_densex_runs.py`
- `experiments/densex_prepare_corpus.py`
- `experiments/densex_build_index.py`
- `evaluation/questions.jsonl`
- `evaluation/questions_review.md`
- `densex/parent_aggregation.py`
- `allocation/`
- `compression/`
- `retrieval/`
- `scoring/`

如果前端需要调用现有能力，优先新增 API wrapper 或 service 层，不要直接重构实验代码。

推荐新增位置：

```text
api/
frontend/
```

例如：

```text
api/server.py
api/schemas.py
api/services/qa_service.py
frontend/
```

## 后端 API 第一版

第一版只需要一个问答接口：

```text
POST /ask
```

请求：

```json
{
  "query": "DepthDark 在哪些数据集上进行了训练？"
}
```

返回给前端：

```json
{
  "answer": "...",
  "run_id": "...",
  "error": null
}
```

后端内部可以继续保存：

- retrieved chunks
- selected chunks
- context tokens
- usage
- details path

但这些字段不返回给普通前端。

## 默认问答策略

第一版不要在前端暴露复杂参数。

后端使用固定默认配置，例如：

```text
strategy = baseline
granularity = chunk
top_k = 固定值
context_budget = 固定值
compression = none 或当前实验验证后的稳定配置
```

后续如果 fine-to-chunk 或其他方法在实验中表现更好，再由后端替换默认策略。

## 前端第一版

推荐技术栈：

```text
React + Vite + TypeScript + Ant Design
```

第一版页面结构：

```text
顶部：系统名称
中间：问题输入框
按钮：提交 / 清空
下方：最终回答
侧边或下方：当前会话历史（可选）
```

页面状态：

- idle
- loading
- answered
- error
- empty query

## 研究信息保留方式

虽然前端不展示研究信息，但后端仍然应保持 run details 和 CSV 记录。

现有实验输出位置：

```text
experiments/runs/
experiments/results.csv
experiments/*_densex_results.csv
experiments/*_densex_summary.csv
```

不要为了前端展示而删除或削弱这些记录。

## 开发原则

- 前端开发应与实验管线解耦。
- 新增功能优先新增文件，不要大范围重构已有实验模块。
- 不要把 API key 暴露到前端。
- 前端只能调用后端 API。
- `.env` 不提交到 git。
- 不提交 `experiments/runs/`、`experiments/densex_corpus/`、`storage/densex/` 等生成物。
- 修改依赖时同步更新 `requirements.txt` 或前端 package 文件。

## 验证要求

每次涉及前端或 API 开发，至少验证：

```text
后端 API 可启动
POST /ask 能返回 answer
前端页面可启动
输入问题后能展示最终回答
错误状态能正常展示
```

如果修改了 Python 文件，至少运行：

```bash
python -m py_compile <modified_python_files>
```

如果修改了前端文件，运行对应的：

```bash
npm run build
```

或者项目中实际配置的前端检查命令。

## Git 提交

前端开发建议按阶段提交：

```text
Add QA API
Add frontend QA page
Connect frontend to QA API
Add session history
```

不要把实验结果 JSON、模型文件、缓存文件和本地配置提交到 git。
