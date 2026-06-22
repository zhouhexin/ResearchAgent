"""Lightweight MLP model for SAGE-style sentence-pair segmentation."""

from __future__ import annotations

import torch
from torch import nn


def build_pair_features(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Build SAGE pair features: x1, x2, x1 - x2, x1 * x2."""
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")
    if x1.ndim != 2:
        raise ValueError("Expected 2D tensors shaped [pairs, embedding_dim]")
    return torch.cat([x1, x2, x1 - x2, x1 * x2], dim=1)


class SegmentationMLP(nn.Module):
    """Small binary classifier over adjacent sentence-pair features."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

