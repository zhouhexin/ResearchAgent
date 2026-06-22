import unittest

from sage_segmenter.dataset import build_sentence_pair_rows


class SageDatasetTests(unittest.TestCase):
    def test_build_sentence_pair_rows_labels_same_and_cross_paragraph_pairs(self):
        chunks = [
            {
                "id": "chunk-1",
                "source": "data/ACD.pdf",
                "page": 2,
                "text": (
                    "ACDepth introduces a new restoration module. "
                    "The module improves depth estimation.\n\n"
                    "ACDepth evaluates the design with ablations. "
                    "The ablation confirms each component."
                ),
            }
        ]

        rows = build_sentence_pair_rows(chunks, min_sentence_chars=5)

        self.assertEqual([row["label"] for row in rows], [1, 0, 1])
        self.assertEqual(rows[0]["s1"], "ACDepth introduces a new restoration module.")
        self.assertEqual(rows[0]["s2"], "The module improves depth estimation.")
        self.assertEqual(rows[1]["s1"], "The module improves depth estimation.")
        self.assertEqual(rows[1]["s2"], "ACDepth evaluates the design with ablations.")
        self.assertEqual(rows[1]["pair_type"], "cross_paragraph")
        self.assertEqual(rows[0]["source"], "data/ACD.pdf")


if __name__ == "__main__":
    unittest.main()
