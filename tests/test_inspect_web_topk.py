from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "inspect_web_topk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("inspect_web_topk", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["inspect_web_topk"] = module
    spec.loader.exec_module(module)
    return module


class InspectWebTopkTest(unittest.TestCase):
    def test_summarizes_overlap_and_order_against_first_run(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            self._write_run(
                runs_dir / "web_a.json",
                run_id="web_a",
                query="same question",
                chunk_ids=["c1", "c2", "c3"],
                selected_ids=["c1", "c2"],
            )
            self._write_run(
                runs_dir / "web_b.json",
                run_id="web_b",
                query="same question",
                chunk_ids=["c1", "c3", "c2"],
                selected_ids=["c2", "c1"],
            )
            self._write_run(
                runs_dir / "web_c.json",
                run_id="web_c",
                query="other question",
                chunk_ids=["x1"],
                selected_ids=["x1"],
            )
            os.utime(runs_dir / "web_a.json", (3, 3))
            os.utime(runs_dir / "web_b.json", (2, 2))
            os.utime(runs_dir / "web_c.json", (1, 1))

            summaries = module.summarize_runs(
                runs_dir=runs_dir,
                query="same question",
                limit=10,
            )

        self.assertEqual([item.run_id for item in summaries], ["web_a", "web_b"])
        self.assertEqual(summaries[0].retrieved_ids, ["c1", "c2", "c3"])
        self.assertEqual(summaries[0].overlap_with_first, 3)
        self.assertTrue(summaries[0].same_order_as_first)
        self.assertEqual(summaries[0].selected_overlap_with_first, 2)
        self.assertTrue(summaries[0].selected_same_order_as_first)
        self.assertEqual(summaries[1].overlap_with_first, 3)
        self.assertFalse(summaries[1].same_order_as_first)
        self.assertEqual(summaries[1].selected_overlap_with_first, 2)
        self.assertFalse(summaries[1].selected_same_order_as_first)

        report = module.format_report(summaries, show_chunks=False)
        self.assertIn("selected_overlap_with_first", report)
        self.assertIn("selected_same_order_as_first", report)

    def _write_run(
        self,
        path: Path,
        *,
        run_id: str,
        query: str,
        chunk_ids: list[str],
        selected_ids: list[str],
    ) -> None:
        payload = {
            "run_id": run_id,
            "query": query,
            "top_k": len(chunk_ids),
            "retrieved_chunks": [
                {
                    "id": chunk_id,
                    "source": "data/example.pdf",
                    "page": index + 1,
                    "score": 1.0 - index * 0.1,
                }
                for index, chunk_id in enumerate(chunk_ids)
            ],
            "selected_chunks": [{"id": chunk_id} for chunk_id in selected_ids],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
