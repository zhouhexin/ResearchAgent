import unittest

from experiments.evaluate_densex_runs import _parse_run_label


class EvaluateDenseXRunsTests(unittest.TestCase):
    def test_parse_run_label_supports_existing_granularity(self):
        granularity, question_id = _parse_run_label(
            "qa_parent_v1_sentence-to-chunk_depthdark_contributions",
            "qa_parent_v1",
            {"depthdark_contributions"},
        )

        self.assertEqual(granularity, "sentence-to-chunk")
        self.assertEqual(question_id, "depthdark_contributions")

    def test_parse_run_label_supports_granularity_with_underscore(self):
        granularity, question_id = _parse_run_label(
            "qa_dedup_v1_sentence_dedup-to-chunk_depthdark_contributions",
            "qa_dedup_v1",
            {"depthdark_contributions"},
        )

        self.assertEqual(granularity, "sentence_dedup-to-chunk")
        self.assertEqual(question_id, "depthdark_contributions")


if __name__ == "__main__":
    unittest.main()
