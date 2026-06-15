import unittest
import tempfile
from pathlib import Path

from experiments.diagnose_parent_selection import (
    _matched_evidence_refs,
    _resolve_fine_index_dir,
    _summarize,
)


class DiagnoseParentSelectionTests(unittest.TestCase):
    def test_evidence_matching_uses_source_basename_and_page(self):
        question = {
            "gold_evidence": [
                {
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 7,
                    "evidence_for": "ordinal_guidance_distillation",
                }
            ]
        }
        chunk = {
            "source": r"E:\PycharmProject\ResearchAgent\data\Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
            "page": "7",
        }

        self.assertEqual(
            _matched_evidence_refs(chunk, question),
            [
                "Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf:p7:ordinal_guidance_distillation"
            ],
        )

    def test_evidence_matching_accepts_acd_source_alias(self):
        question = {
            "gold_evidence": [
                {
                    "paper_id": "always_clear_depth",
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 7,
                    "evidence_for": "ordinal_guidance_distillation",
                }
            ]
        }
        chunk = {
            "source": r"data\ACD.pdf",
            "page": 7,
        }

        self.assertEqual(
            _matched_evidence_refs(chunk, question),
            [
                "Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf:p7:ordinal_guidance_distillation"
            ],
        )

    def test_summarize_separates_parent_candidate_and_selected_recall(self):
        question = {
            "id": "always_clear_depth_ablation_components",
            "gold_items": [
                {"id": "distillation_learning", "name": "distillation learning", "aliases": []},
                {
                    "id": "feature_consistency_constraint",
                    "name": "feature consistency constraint",
                    "aliases": [],
                },
            ],
            "gold_evidence": [
                {
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 7,
                    "evidence_for": "distillation_learning",
                },
                {
                    "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                    "page": 7,
                    "evidence_for": "feature_consistency_constraint",
                },
            ],
        }
        annotated_candidates = [
            {
                "id": "chunk-a",
                "parent_rank": 1,
                "source": "data/Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf",
                "page": 7,
                "text": "The ablation includes distillation learning.",
                "matched_gold_ids": ["distillation_learning"],
                "matched_evidence_refs": [
                    "Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf:p7:distillation_learning",
                    "Always Clear Depth- Robust Monocular Depth Estimation Under Adverse Weather.pdf:p7:feature_consistency_constraint",
                ],
            },
            {
                "id": "chunk-b",
                "parent_rank": 2,
                "source": "data/Other.pdf",
                "page": 1,
                "text": "Unrelated text.",
                "matched_gold_ids": [],
                "matched_evidence_refs": [],
            },
        ]
        selected = [annotated_candidates[1]]

        summary = _summarize(
            question=question,
            query_mode="original",
            query_used="Always Clear Depth 的消融实验验证了哪些组件？",
            granularity="proposition",
            strategy="baseline",
            budget=500,
            fine_top_m=150,
            parent_top_k=50,
            annotated_candidates=annotated_candidates,
            selected=selected,
        )

        self.assertEqual(summary["parent_candidate_gold_recall"], 0.5)
        self.assertEqual(summary["parent_candidate_evidence_recall"], 1.0)
        self.assertEqual(summary["selected_gold_recall"], 0.0)
        self.assertEqual(summary["selected_evidence_recall"], 0.0)
        self.assertEqual(summary["first_gold_parent_rank"], 1)
        self.assertEqual(summary["selected_first_gold_parent_rank"], "")

    def test_resolve_fine_index_dir_can_use_dedup_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "proposition_dedup").mkdir()

            self.assertEqual(
                _resolve_fine_index_dir(
                    "proposition",
                    root,
                    allow_dedup_fallback=True,
                ),
                root / "proposition_dedup",
            )


if __name__ == "__main__":
    unittest.main()
