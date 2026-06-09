from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from api.services.feedback_service import record_feedback


class FeedbackServiceTest(unittest.TestCase):
    def test_record_feedback_appends_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "web_feedback.jsonl"

            saved = record_feedback(
                run_id="web_run",
                query="question",
                answer="answer",
                rating="accurate",
                feedback_path=feedback_path,
            )

            rows = [
                json.loads(line)
                for line in feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(saved["run_id"], "web_run")
        self.assertEqual(saved["rating"], "accurate")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "question")
        self.assertEqual(rows[0]["answer"], "answer")
        self.assertIn("created_at", rows[0])


if __name__ == "__main__":
    unittest.main()
