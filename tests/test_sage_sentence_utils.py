import unittest

from sage_segmenter.sentence_utils import (
    merge_ordered_chunk_texts,
    split_paragraphs,
    split_sentences,
)


class SageSentenceUtilsTests(unittest.TestCase):
    def test_split_paragraphs_normalizes_blank_lines(self):
        text = " First paragraph line. \n still first. \n\n\n Second paragraph. "

        paragraphs = split_paragraphs(text)

        self.assertEqual(paragraphs, ["First paragraph line. still first.", "Second paragraph."])

    def test_split_sentences_keeps_reasonable_units(self):
        text = "DepthDark proposes LLDG. It also proposes LLPEFT! Does it evaluate on RobotCar-Night? Yes."

        sentences = split_sentences(text, min_chars=5)

        self.assertEqual(
            sentences,
            [
                "DepthDark proposes LLDG.",
                "It also proposes LLPEFT!",
                "Does it evaluate on RobotCar-Night?",
                "Yes.",
            ],
        )

    def test_merge_ordered_chunk_texts_removes_prefix_suffix_overlap(self):
        chunks = [
            {"text": "The first chunk has overlapping text."},
            {"text": "overlapping text. The second chunk continues."},
        ]

        merged = merge_ordered_chunk_texts(chunks, min_overlap=10)

        self.assertEqual(merged, "The first chunk has overlapping text. The second chunk continues.")


if __name__ == "__main__":
    unittest.main()
