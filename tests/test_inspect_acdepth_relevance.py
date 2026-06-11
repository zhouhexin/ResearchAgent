import unittest
from pathlib import Path

from experiments.inspect_acdepth_relevance import (
    _format_evidence_ref,
    _matched_evidence_refs,
    _resolve_index_dir,
    _summarize_hits,
)


class InspectACDepthRelevanceTests(unittest.TestCase):
    def test_matched_evidence_refs_use_source_basename_and_page(self):
        question = {
            "gold_evidence": [
                {
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 7,
                    "evidence_for": "acdepth_framework",
                },
                {
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 2,
                    "evidence_for": "ordinal_guidance_distillation",
                },
            ]
        }
        hit = {
            "source": "Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
            "page": 7,
        }

        self.assertEqual(
            _matched_evidence_refs(hit, question),
            ["Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf:p7:acdepth_framework"],
        )

    def test_summarize_hits_reports_first_gold_and_evidence_rank(self):
        hits = [
            {"rank": 1, "matched_gold_ids": "", "matched_evidence_refs": ""},
            {"rank": 2, "matched_gold_ids": "acdepth_framework", "matched_evidence_refs": ""},
            {
                "rank": 3,
                "matched_gold_ids": "",
                "matched_evidence_refs": "Always Clear Depth.pdf:p7:acdepth_framework",
            },
        ]

        summary = _summarize_hits("always_clear_depth_contributions", "sentence", hits)

        self.assertEqual(summary["first_gold_alias_rank"], 2)
        self.assertEqual(summary["first_evidence_page_rank"], 3)
        self.assertEqual(summary["gold_alias_hit_count"], 1)
        self.assertEqual(summary["evidence_page_hit_count"], 1)
        self.assertEqual(summary["matched_gold_ids_top_m"], "acdepth_framework")

    def test_resolve_index_dir_uses_main_storage_for_chunk(self):
        root = Path("/project")

        self.assertEqual(
            _resolve_index_dir("chunk", root / "storage" / "densex", root / "storage", allow_dedup_fallback=False),
            root / "storage",
        )

    def test_resolve_index_dir_can_fallback_to_dedup_index(self):
        with self.subTest("sentence fallback"):
            self.assertEqual(
                _resolve_index_dir(
                    "sentence",
                    Path("storage/densex"),
                    Path("storage"),
                    allow_dedup_fallback=True,
                    exists=lambda path: str(path).endswith("sentence_dedup"),
                ),
                Path("storage/densex/sentence_dedup"),
            )


if __name__ == "__main__":
    unittest.main()
