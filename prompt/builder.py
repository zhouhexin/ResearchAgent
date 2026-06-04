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
