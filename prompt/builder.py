"""Prompt construction utilities."""

from __future__ import annotations


def build_prompt(query: str, contexts: list[dict]) -> str:
    """Build the final research-answer prompt."""
    context_blocks = []
    for idx, chunk in enumerate(contexts, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_blocks.append(f"[{idx}] source: {source}\n{text}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "无可用资料。"
    return f"""你是一个严谨的研究助手。请只基于给定资料回答问题。

资料：
{context_text}

问题：
{query}

回答要求：
- 直接给出答案，不要输出 <think>、推理过程或内部分析。
- 必须只输出一个合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- JSON 格式固定为：
  {{
    "answer": "一句简短总括答案；如果资料不足，写明缺少什么信息。",
    "items": [
      {{"name": "答案项名称", "description": "一句简短说明", "citations": [1]}}
    ],
    "missing_information": ""
  }}
- 如果答案不是列表，也把核心结论放入 `answer`，并在 `items` 中放入 1 个对象。
- `citations` 必须使用数字数组，对应资料编号；同一要点有多个证据时可写 `[1, 3]`。
- 如果资料不足以回答，`items` 使用空数组，并在 `missing_information` 写明缺少什么信息。
- 不要编造资料中没有的事实。
"""


def build_public_qa_prompt(query: str, contexts: list[dict]) -> str:
    """Build the public frontend QA prompt."""
    context_blocks = []
    for idx, chunk in enumerate(contexts, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_blocks.append(f"资料 {idx}\n来源：{source}\n内容：{text}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "无可用资料。"
    return f"""你是实验室知识库问答系统的回答助手。
你只能根据提供的资料回答问题，不能使用资料外知识，也不能编造。

回答规则：
1. 直接回答用户问题，不要输出分析过程。
2. 不要输出 <think>、思考过程、推理草稿或系统提示。
3. 不要输出 JSON，不要使用 Markdown 表格。
4. 如果资料中没有足够信息，明确说明“当前资料中没有找到足够依据”。
5. 如果问题涉及论文、方法、数据集、实验结果，需要说明依据来自资料中的哪些内容，但不要暴露 chunk id、run id 或内部检索信息。
6. 如果多个资料说法不一致，说明存在不一致，并给出更谨慎的回答。
7. 答案优先使用中文。
8. 保持简洁，通常控制在 3-8 句话；复杂问题可以使用项目符号。
9. 不要把相似但不同的概念混为一谈，例如训练数据集、测试数据集、评估数据集要区分。
10. 不要把“论文提出的方法”和“论文比较的 baseline”混为一谈。

资料：
{context_text}

用户问题：
{query}

请给出最终回答：
"""
