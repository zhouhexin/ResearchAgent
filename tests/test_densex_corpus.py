import unittest

from densex.corpus import deduplicate_units_by_text, make_sentence_units


class DenseXCorpusTests(unittest.TestCase):
    def test_make_sentence_units_can_deduplicate_normalized_text_globally(self):
        chunks = [
            {
                "id": "chunk-a",
                "text": "DepthDark uses LLPEFT. It evaluates on RobotCar-Night.",
                "source": "data/DepthDark.pdf",
                "page": 1,
            },
            {
                "id": "chunk-b",
                "text": "  depthdark   uses llpeft. It trains on nuScenes.",
                "source": "data/DepthDark.pdf",
                "page": 2,
            },
        ]

        units = make_sentence_units(chunks, min_chars=5, dedup=True)

        self.assertEqual(
            [unit["text"] for unit in units],
            [
                "DepthDark uses LLPEFT.",
                "It evaluates on RobotCar-Night.",
                "It trains on nuScenes.",
            ],
        )
        self.assertEqual(units[0]["parent_chunk_id"], "chunk-a")

    def test_deduplicate_units_by_text_keeps_first_normalized_proposition(self):
        units = [
            {
                "id": "proposition::chunk-a::p0",
                "text": "DepthDark proposes LLPEFT for low-light scenarios.",
                "parent_chunk_id": "chunk-a",
            },
            {
                "id": "proposition::chunk-b::p0",
                "text": " depthdark   proposes llpeft for low-light scenarios. ",
                "parent_chunk_id": "chunk-b",
            },
            {
                "id": "proposition::chunk-b::p1",
                "text": "DepthDark introduces LLDG to generate paired low-light depth data.",
                "parent_chunk_id": "chunk-b",
            },
        ]

        deduped = deduplicate_units_by_text(units)

        self.assertEqual([unit["id"] for unit in deduped], ["proposition::chunk-a::p0", "proposition::chunk-b::p1"])


if __name__ == "__main__":
    unittest.main()
