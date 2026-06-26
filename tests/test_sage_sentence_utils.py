import unittest
import tempfile
from pathlib import Path

from sage_segmenter.sentence_utils import (
    load_page_records_from_docs,
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

    def test_load_page_records_from_docs_preserves_paragraph_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "data"
            docs.mkdir()
            path = docs / "sample.txt"
            path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

            records = load_page_records_from_docs(docs, extensions={".txt"})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["page"], None)
        self.assertEqual(records[0]["source"], str(path))
        self.assertIn("\n\n", records[0]["text"])

    def test_load_page_records_from_docs_filters_by_source_contains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir) / "data"
            docs.mkdir()
            (docs / "keep.txt").write_text("Keep this paragraph.", encoding="utf-8")
            (docs / "drop.txt").write_text("Drop this paragraph.", encoding="utf-8")

            records = load_page_records_from_docs(
                docs,
                extensions={".txt"},
                source_contains="keep",
            )

        self.assertEqual([Path(record["source"]).name for record in records], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
