from __future__ import annotations

import unittest

from api.services.qa_service import _extract_final_answer


class QaAnswerCleaningTest(unittest.TestCase):
    def test_extract_final_answer_removes_markdown_bold(self) -> None:
        raw_answer = (
            '{"answer": "推荐 DepthDark: Robust Monocular Depth Estimation '
            'for Low-Light Environments（**DepthDark**）。"}'
        )

        answer = _extract_final_answer(raw_answer)

        self.assertEqual(
            answer,
            "推荐 DepthDark: Robust Monocular Depth Estimation for Low-Light Environments（DepthDark）。",
        )


if __name__ == "__main__":
    unittest.main()
