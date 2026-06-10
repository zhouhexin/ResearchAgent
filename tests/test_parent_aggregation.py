import unittest

from densex.parent_aggregation import aggregate_fine_hits_to_parent_chunks


class ParentAggregationTests(unittest.TestCase):
    def test_exact_dedup_happens_inside_each_parent_chunk(self):
        parent_chunks = {
            "chunk-a": {"id": "chunk-a", "text": "Parent A"},
            "chunk-b": {"id": "chunk-b", "text": "Parent B"},
        }
        fine_hits = [
            {
                "id": "sentence::a::s0",
                "text": "DepthDark evaluates on RobotCar-Night.",
                "parent_chunk_id": "chunk-a",
                "score": 0.9,
            },
            {
                "id": "sentence::a::s1",
                "text": " depthdark   evaluates on robotcar-night. ",
                "parent_chunk_id": "chunk-a",
                "score": 0.8,
            },
            {
                "id": "sentence::a::s2",
                "text": "DepthDark evaluates on nuScenes-Night.",
                "parent_chunk_id": "chunk-a",
                "score": 0.7,
            },
            {
                "id": "sentence::b::s0",
                "text": "DepthDark evaluates on RobotCar-Night.",
                "parent_chunk_id": "chunk-b",
                "score": 0.6,
            },
        ]

        candidates = aggregate_fine_hits_to_parent_chunks(
            fine_hits,
            parent_chunks,
            parent_top_k=10,
            top_child_count=3,
            child_sum_weight=0.1,
            fine_hit_dedup="exact-per-parent",
        )

        by_id = {candidate["id"]: candidate for candidate in candidates}
        self.assertEqual(by_id["chunk-a"]["fine_to_chunk"]["matched_child_count"], 3)
        self.assertEqual(by_id["chunk-a"]["fine_to_chunk"]["deduplicated_child_count"], 2)
        self.assertEqual(
            by_id["chunk-a"]["fine_to_chunk"]["top_child_ids"],
            ["sentence::a::s0", "sentence::a::s2"],
        )
        self.assertAlmostEqual(by_id["chunk-a"]["score"], 0.9 + 0.1 * (0.9 + 0.7))
        self.assertIn("chunk-b", by_id)


if __name__ == "__main__":
    unittest.main()
