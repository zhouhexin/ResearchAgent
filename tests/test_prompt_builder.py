from __future__ import annotations

import unittest

from prompt.builder import build_prompt, build_public_qa_prompt


class PromptBuilderTest(unittest.TestCase):
    def test_public_qa_prompt_is_plain_answer_oriented(self) -> None:
        prompt = build_public_qa_prompt(
            "DepthDark 在哪些数据集上训练？",
            [{"source": "paper.pdf", "text": "DepthDark uses low-light data."}],
        )

        self.assertIn("实验室知识库问答系统", prompt)
        self.assertIn("不要输出 JSON", prompt)
        self.assertIn("不要输出 <think>", prompt)
        self.assertIn("不要使用 Markdown 加粗", prompt)
        self.assertNotIn("**", prompt)
        self.assertIn("首次提到论文时必须使用论文全称", prompt)
        self.assertIn("当前资料中没有找到足够依据", prompt)
        self.assertIn("DepthDark uses low-light data.", prompt)

    def test_experiment_prompt_still_requires_json(self) -> None:
        prompt = build_prompt("问题", [{"source": "paper.pdf", "text": "content"}])

        self.assertIn("必须只输出一个合法 JSON 对象", prompt)
        self.assertIn('"items"', prompt)


if __name__ == "__main__":
    unittest.main()
