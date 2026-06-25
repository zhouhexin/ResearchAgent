import unittest

from sage_segmenter.segmenter import make_semantic_chunk_units, split_by_scores


class SageSegmenterTests(unittest.TestCase):
    def test_split_by_scores_splits_when_score_falls_below_threshold(self):
        sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
        scores = [0.9, 0.2]

        segments = split_by_scores(sentences, scores, threshold=0.55, min_chars=1, max_chars=1000)

        self.assertEqual([segment.text for segment in segments], ["Sentence one. Sentence two.", "Sentence three."])
        self.assertEqual(segments[0].score_min, 0.9)
        self.assertIsNone(segments[1].score_min)

    def test_make_semantic_chunk_units_uses_segment_metadata(self):
        chunks = [
            {
                "source": "data/ACD.pdf",
                "page": 1,
                "text": "First sentence. Second sentence. Third sentence.",
            }
        ]

        units = make_semantic_chunk_units(
            chunks,
            scorer=lambda sentences: [0.8, 0.1],
            threshold=0.55,
            min_sentence_chars=5,
            min_chars=1,
            max_chars=1000,
        )

        self.assertEqual([unit["text"] for unit in units], ["First sentence. Second sentence.", "Third sentence."])
        self.assertEqual(units[0]["granularity"], "semantic_chunk")
        self.assertEqual(units[0]["source"], "data/ACD.pdf")
        self.assertEqual(units[0]["page"], 1)
        self.assertEqual(units[0]["sentence_count"], 2)
        self.assertEqual(units[0]["segment_score_min"], 0.8)

    def test_make_semantic_chunk_units_keeps_paragraphs_as_hard_boundaries(self):
        calls = []

        def scorer(sentences):
            calls.append(sentences)
            return [0.95 for _ in range(len(sentences) - 1)]

        chunks = [
            {
                "source": "data/ACD.pdf",
                "page": 1,
                "text": (
                    "First paragraph sentence one. First paragraph sentence two.\n\n"
                    "Second paragraph sentence one. Second paragraph sentence two."
                ),
            }
        ]

        units = make_semantic_chunk_units(
            chunks,
            scorer=scorer,
            threshold=0.55,
            min_sentence_chars=5,
            min_chars=1,
            max_chars=1000,
        )

        self.assertEqual(
            [unit["text"] for unit in units],
            [
                "First paragraph sentence one. First paragraph sentence two.",
                "Second paragraph sentence one. Second paragraph sentence two.",
            ],
        )
        self.assertEqual([unit["paragraph_index"] for unit in units], [0, 1])
        self.assertEqual(
            calls,
            [
                ["First paragraph sentence one.", "First paragraph sentence two."],
                ["Second paragraph sentence one.", "Second paragraph sentence two."],
            ],
        )


if __name__ == "__main__":
    unittest.main()
