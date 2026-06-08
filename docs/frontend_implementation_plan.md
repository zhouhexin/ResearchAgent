# 前端实现计划

本文档记录如何基于当前 ResearchAgent 项目搭建一个实验室知识问答系统。

## 目标

构建一个面向实验室成员开放的知识问答前端。用户输入关键词或自然语言问题后，系统基于当前论文知识库和已有 RAG 方法生成最终回答。

前端只需要展示：

```text
用户问题
最终回答
必要的加载、错误和空结果状态
```

前端不展示：

```text
retrieved chunks
selected context
token 使用情况
实验评估指标
gold items / evidence 匹配结果
```

这些信息仍然可以在后端日志、run details 和实验 CSV 中保留，供个人研究和调试使用。

## 产品定位

这个系统有两层定位：

```text
对实验室其他用户：知识问答系统
对我个人：基于当前研究方法的实验工具
```

因此，普通前端应该保持简洁，像一个可直接使用的问答系统；复杂的检索过程、context selection、compression、granularity 对比和评估指标不暴露给普通用户。

当前研究方法仍然作为后端能力存在：

```text
query -> retrieval -> context selection -> optional compression -> LLM answer
```

但用户只看到最终回答。

## 推荐架构

建议在前端和现有 Python 实验管线之间增加一个轻量 API 后端。

```text
Frontend UI
  -> FastAPI backend
    -> storage/metadata.json / FAISS index
    -> app.answer_query 问答流程
    -> experiments/runs 后端调试记录
    -> experiments/*.csv 个人实验分析
```

推荐前端技术栈：

```text
React + Vite + TypeScript
Ant Design
```

Ant Design 适合快速构建稳定的输入框、按钮、结果区域、历史记录列表和基础管理页面。

## 用户侧功能范围

第一版用户侧只保留最小功能：

- 输入关键词或自然语言问题。
- 点击提交。
- 展示最终回答。
- 展示加载状态。
- 展示错误提示。
- 可选展示最近几次问答历史。

不在用户侧展示检索证据和实验指标，避免普通用户理解成本过高。

## 研究侧保留能力

虽然前端不展示证据和指标，但后端仍然需要保留研究所需信息：

- run id
- run label
- retrieved chunks
- selected chunks
- context tokens
- compression info
- details path
- LLM usage

这些信息继续写入：

```text
experiments/runs/
experiments/results.csv
```

这样前端对外是问答系统，对个人研究仍然可追踪和可评估。

## 实现顺序

### 1. 后端问答 API

先增加一个 FastAPI 后端，并实现问答接口：

```text
POST /ask
```

请求参数第一版保持简单：

```json
{
  "query": "DepthDark 在哪些数据集上进行了训练？"
}
```

后端内部使用默认配置调用当前 RAG 流程：

```text
retrieval -> allocation -> prompt -> LLM -> answer
```

接口返回给前端的内容只包括：

```json
{
  "answer": "...",
  "run_id": "...",
  "error": null
}
```

`run_id` 可以用于后端排查，但前端不需要展示复杂细节。

### 2. 前端问答页面

创建第一版问答页面：

```text
顶部：系统名称
中间：问题输入框
按钮：提交 / 清空
下方：最终回答
```

页面状态需要包括：

- idle
- loading
- answered
- error
- empty query

普通用户只关心问题和回答，所以页面应保持简洁。

### 3. 基础问答历史

增加本地历史记录，方便用户查看当前会话内问过的问题。

第一版可以只存在浏览器内存中，不需要数据库：

```text
question
answer
timestamp
```

后续如果需要多人使用记录，再增加持久化存储。

### 4. 后端默认参数配置

为了对其他人开放使用，前端不暴露复杂实验参数。

默认参数由后端配置控制，例如：

```text
strategy
top_k
context_budget
compression
compression_stage
granularity / retrieval mode
```

这些参数可以先写在后端配置中，由你根据实验结果选择当前最稳定的方案。

建议第一版默认使用当前表现最稳的配置，例如：

```text
chunk retrieval
baseline allocation
固定 top_k
固定 context_budget
```

等 fine-to-chunk 或其他方法验证更好后，再替换后端默认策略。

### 5. 管理 / 调试入口

如果需要保留个人研究入口，可以增加一个不面向普通用户的调试页面或后端接口。

这部分不作为第一版公开前端：

```text
/admin/runs
/admin/results
/admin/search-debug
```

调试入口可以展示：

- retrieved chunks
- selected context
- token usage
- run details
- CSV 结果

但默认用户页面不显示这些内容。

### 6. 关键词检索能力

用户可以输入关键词，也可以输入自然语言问题。

第一版不单独做“检索结果列表页”，而是统一进入问答流程：

```text
关键词 / 问题 -> 后端检索 -> LLM 生成最终回答 -> 前端展示最终回答
```

如果后续发现用户需要查原文，再考虑增加“查看相关资料”按钮，但默认仍然隐藏证据细节。

### 7. 后续增强

基础问答系统稳定后，再考虑：

- 用户登录
- 问答历史持久化
- 多知识库切换
- 管理端上传论文
- 后台重新构建索引
- 基于实验结果切换 retrieval mode

这些都不是第一版必须功能。

## 第一个里程碑

第一版可用功能：

```text
FastAPI /ask
React 问答页面
输入问题
展示最终回答
加载和错误状态
```

这一步完成后，系统就可以作为实验室内部知识问答系统试用。

第一版启动方式：

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

前端默认调用：

```text
http://127.0.0.1:8000
```

如果需要修改 API 地址，在 `frontend/.env` 中设置：

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 第二个里程碑

增加：

```text
当前会话问答历史
后端 run_id 记录
更稳定的错误处理
默认参数配置文件
```

这一步让系统更适合给其他人持续使用。

## 第三个里程碑

增加个人研究调试能力，但不暴露给普通用户：

```text
调试页面 / 管理入口
run details 查看
实验 CSV 查看
retrieval mode 切换
```

这一步让系统同时满足两类需求：

```text
普通用户：只使用问答系统
个人研究：继续分析检索、context 和实验效果
```
