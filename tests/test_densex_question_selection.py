import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_questions(path: Path) -> None:
    rows = [
        {"id": "q_one", "query": "Question one?"},
        {"id": "q_two", "query": "Question two?"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class DenseXQuestionSelectionTests(unittest.TestCase):
    def test_fixed_batch_supports_all_question_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            questions = Path(tmpdir) / "questions.jsonl"
            _write_questions(questions)

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/run_fixed_qa_batch.py",
                    "--question-set",
                    "all",
                    "--questions",
                    str(questions),
                    "--granularities",
                    "chunk",
                    "--budgets",
                    "500",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--question-ids q_one,q_two", result.stdout)
        self.assertIn(f"--questions {questions}", result.stdout)

    def test_direct_sweep_supports_all_questions_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            questions = Path(tmpdir) / "questions.jsonl"
            _write_questions(questions)

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/run_densex_sweep.py",
                    "--all-questions",
                    "--questions",
                    str(questions),
                    "--granularities",
                    "chunk",
                    "--budgets",
                    "500",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qid=q_one", result.stdout)
        self.assertIn("qid=q_two", result.stdout)

    def test_parent_sweep_supports_all_question_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            questions = Path(tmpdir) / "questions.jsonl"
            _write_questions(questions)

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/run_densex_parent_sweep.py",
                    "--question-set",
                    "all",
                    "--questions",
                    str(questions),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
