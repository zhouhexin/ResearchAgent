import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SageScriptsTests(unittest.TestCase):
    def test_sage_build_pairs_writes_sentence_pair_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = tmp / "metadata.json"
            output = tmp / "pairs.jsonl"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "id": "chunk-1",
                            "source": "data/ACD.pdf",
                            "page": 1,
                            "text": "First same paragraph sentence. Second same paragraph sentence.\n\nNew paragraph starts here.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/sage_build_pairs.py",
                    "--metadata",
                    str(metadata),
                    "--output",
                    str(output),
                    "--min-sentence-chars",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["label"] for row in rows}, {0, 1})

    def test_sage_sweep_dry_run_prints_chunk_and_semantic_chunk_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            questions = Path(tmpdir) / "questions.jsonl"
            questions.write_text(json.dumps({"id": "q_one", "query": "Question one?"}) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/run_sage_semantic_chunk_sweep.py",
                    "--questions",
                    str(questions),
                    "--question-ids",
                    "q_one",
                    "--budgets",
                    "300",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("granularity=chunk", result.stdout)
            self.assertIn("granularity=semantic_chunk", result.stdout)


if __name__ == "__main__":
    unittest.main()
