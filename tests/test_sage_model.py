import unittest

import torch

from sage_segmenter.model import SegmentationMLP, build_pair_features


class SageModelTests(unittest.TestCase):
    def test_build_pair_features_concatenates_expected_components(self):
        x1 = torch.tensor([[1.0, 2.0]])
        x2 = torch.tensor([[0.5, 4.0]])

        features = build_pair_features(x1, x2)

        self.assertTrue(
            torch.equal(
                features,
                torch.tensor([[1.0, 2.0, 0.5, 4.0, 0.5, -2.0, 0.5, 8.0]]),
            )
        )

    def test_segmentation_mlp_returns_one_logit_per_pair(self):
        model = SegmentationMLP(input_dim=8, hidden_dim=4, dropout=0.0)
        features = torch.ones((3, 8))

        logits = model(features)

        self.assertEqual(tuple(logits.shape), (3,))


if __name__ == "__main__":
    unittest.main()
