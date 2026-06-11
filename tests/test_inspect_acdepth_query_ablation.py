import unittest

from experiments.inspect_acdepth_query_ablation import (
    _build_query_variants,
    _is_title_junk,
    _overlap_rate,
    _source_matches,
    _summarize_variant_hits,
)


class InspectACDepthQueryAblationTests(unittest.TestCase):
    def test_build_query_variants_uses_original_title_content_and_source_filtered(self):
        question = {
            "id": "always_clear_depth_contributions",
            "query": "Always Clear Depth 论文的主要贡献包括哪些？",
        }

        variants = _build_query_variants(question, "Always Clear Depth")

        self.assertEqual(variants["original"], "Always Clear Depth 论文的主要贡献包括哪些？")
        self.assertEqual(variants["title_only"], "Always Clear Depth")
        self.assertIn("main contributions", variants["content_only"])
        self.assertEqual(variants["source_filtered"], variants["content_only"])

    def test_is_title_junk_detects_title_page_and_table_fragments(self):
        self.assertTrue(_is_title_junk("The title of the paper is Always Clear Depth."))
        self.assertTrue(_is_title_junk("The paper has a section called page 7."))
        self.assertTrue(_is_title_junk("absRel RMSE sqRel R absRel RMSE sqRel R"))
        self.assertFalse(
            _is_title_junk(
                "Always Clear Depth proposes multi-granularity knowledge distillation."
            )
        )

    def test_overlap_rate_compares_unit_ids(self):
        left = [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}]
        right = [{"unit_id": "b"}, {"unit_id": "d"}]

        self.assertEqual(_overlap_rate(left, right), 1 / 3)

    def test_source_matches_accepts_basename_with_different_path_separators(self):
        hit = {"source": "data\\Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf"}

        self.assertTrue(
            _source_matches(
                hit,
                "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
            )
        )

    def test_summarize_variant_hits_reports_title_junk_and_overlap(self):
        rows = [
            {
                "rank": 1,
                "unit_id": "title",
                "matched_gold_ids": "",
                "matched_evidence_refs": "",
                "is_title_junk": True,
            },
            {
                "rank": 2,
                "unit_id": "gold",
                "matched_gold_ids": "acdepth_framework",
                "matched_evidence_refs": "Always Clear Depth.pdf:p7:acdepth_framework",
                "is_title_junk": False,
            },
        ]
        title_only_rows = [{"unit_id": "title"}, {"unit_id": "other"}]

        summary = _summarize_variant_hits(
            "always_clear_depth_contributions",
            "sentence",
            "original",
            rows,
            title_only_rows=title_only_rows,
        )

        self.assertEqual(summary["first_gold_alias_rank"], 2)
        self.assertEqual(summary["first_evidence_page_rank"], 2)
        self.assertEqual(summary["title_junk_count"], 1)
        self.assertEqual(summary["title_junk_rate"], 0.5)
        self.assertEqual(summary["overlap_with_title_only"], 0.5)


if __name__ == "__main__":
    unittest.main()
