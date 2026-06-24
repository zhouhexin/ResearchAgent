import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_sage_semantic_chunk_loop import (
    Job,
    completed_run_labels,
    is_retryable_exception,
    run_job_with_retries,
)


class SageLoopRunnerTests(unittest.TestCase):
    def test_completed_run_labels_keeps_only_successful_answer_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            (runs_dir / "ok.json").write_text(
                json.dumps({"run_label": "sage_chunk_q1", "answer": "done"}),
                encoding="utf-8",
            )
            (runs_dir / "dry.json").write_text(
                json.dumps({"run_label": "sage_chunk_q2", "dry_run": True}),
                encoding="utf-8",
            )
            (runs_dir / "empty.json").write_text(
                json.dumps({"run_label": "sage_chunk_q3", "answer": ""}),
                encoding="utf-8",
            )

            labels = completed_run_labels(runs_dir, require_answer=True)

        self.assertEqual(labels, {"sage_chunk_q1"})

    def test_connection_errors_are_retryable(self):
        self.assertTrue(is_retryable_exception(RuntimeError("openai.APIConnectionError: Connection error")))
        self.assertTrue(is_retryable_exception(RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING]")))
        self.assertFalse(is_retryable_exception(ValueError("Question ids not found")))

    def test_run_job_with_retries_retries_then_succeeds(self):
        calls = []
        waits = []
        job = Job(
            question={"id": "q1", "query": "question?"},
            granularity="chunk",
            index_dir=Path("storage/sage/chunk"),
            budget=500,
            run_label="sage_chunk_q1",
        )

        def flaky_answer(**kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise RuntimeError("APIConnectionError: Connection error")
            return "ok"

        result = run_job_with_retries(
            job,
            answer_query=flaky_answer,
            strategy="baseline",
            top_k=50,
            max_retries=3,
            retry_wait_seconds=2.0,
            retry_backoff=2.0,
            dry_run=False,
            sleep=lambda seconds: waits.append(seconds),
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(calls), 3)
        self.assertEqual(waits, [2.0, 4.0])

    def test_run_job_with_retries_stops_after_non_retryable_error(self):
        job = Job(
            question={"id": "q1", "query": "question?"},
            granularity="chunk",
            index_dir=Path("storage/sage/chunk"),
            budget=500,
            run_label="sage_chunk_q1",
        )

        result = run_job_with_retries(
            job,
            answer_query=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
            strategy="baseline",
            top_k=50,
            max_retries=3,
            retry_wait_seconds=2.0,
            retry_backoff=2.0,
            dry_run=False,
            sleep=lambda seconds: None,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 1)
        self.assertIn("bad config", result.error)


if __name__ == "__main__":
    unittest.main()
