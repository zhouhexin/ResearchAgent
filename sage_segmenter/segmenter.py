"""Inference helpers for SAGE-style semantic chunking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from densex.corpus import paper_title_from_source
from retrieval.embed import Embedder
from sage_segmenter.model import SegmentationMLP, build_pair_features
from sage_segmenter.sentence_utils import split_paragraphs, split_sentences


@dataclass(frozen=True)
class SemanticSegment:
    text: str
    score_min: float | None
    sentence_count: int


def split_by_scores(
    sentences: list[str],
    scores: list[float],
    *,
    threshold: float,
    min_chars: int,
    max_chars: int,
) -> list[SemanticSegment]:
    """Split adjacent sentences where the same-chunk score falls below threshold."""
    if not sentences:
        return []
    if len(scores) != max(0, len(sentences) - 1):
        raise ValueError("scores must contain one value per adjacent sentence pair")

    segments: list[SemanticSegment] = []
    current = [sentences[0]]
    current_scores: list[float] = []

    for index, score in enumerate(scores):
        next_sentence = sentences[index + 1]
        next_text = " ".join([*current, next_sentence]).strip()
        would_exceed_max = len(next_text) > max_chars and len(" ".join(current)) >= min_chars
        should_split = score < threshold or would_exceed_max
        if should_split and len(" ".join(current)) >= min_chars:
            segments.append(
                SemanticSegment(
                    text=" ".join(current).strip(),
                    score_min=min(current_scores) if current_scores else None,
                    sentence_count=len(current),
                )
            )
            current = [next_sentence]
            current_scores = []
        else:
            current.append(next_sentence)
            current_scores.append(float(score))

    if current:
        segments.append(
            SemanticSegment(
                text=" ".join(current).strip(),
                score_min=min(current_scores) if current_scores else None,
                sentence_count=len(current),
            )
        )
    return [segment for segment in segments if segment.text]


class SemanticSegmenter:
    """Run an embedding model plus trained MLP over adjacent sentence pairs."""

    def __init__(
        self,
        *,
        embedding_model: str,
        mlp: SegmentationMLP,
        device: str = "cpu",
    ) -> None:
        self.embedder = Embedder(embedding_model)
        self.mlp = mlp.to(device)
        self.mlp.eval()
        self.device = device

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        embedding_model: str | None = None,
        device: str = "cpu",
    ) -> "SemanticSegmenter":
        import torch

        config_path = model_dir / "config.json"
        state_path = model_dir / "mlp.pt"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = embedding_model or config["embedding_model"]
        mlp = SegmentationMLP(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config.get("hidden_dim", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
        mlp.load_state_dict(torch.load(state_path, map_location=device))
        return cls(embedding_model=model_name, mlp=mlp, device=device)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        import torch

        if len(sentences) < 2:
            return []
        left = self.embedder.encode(sentences[:-1])
        right = self.embedder.encode(sentences[1:])
        x1 = torch.from_numpy(left.astype("float32")).to(self.device)
        x2 = torch.from_numpy(right.astype("float32")).to(self.device)
        with torch.no_grad():
            logits = self.mlp(build_pair_features(x1, x2))
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        return [float(score) for score in scores]


def make_semantic_chunk_units(
    chunks: list[dict],
    *,
    scorer: Callable[[list[str]], list[float]],
    threshold: float,
    min_sentence_chars: int = 30,
    min_chars: int = 120,
    max_chars: int = 1200,
) -> list[dict]:
    """Create DenseX-compatible semantic chunk units from page/text records.

    Paragraph boundaries are hard boundaries. The MLP only decides boundaries
    between adjacent sentences inside one paragraph, so a saturated same-chunk
    score cannot merge unrelated paragraphs into one long chunk.
    """
    units = []
    for chunk in chunks:
        source = str(chunk.get("source", ""))
        page = chunk.get("page")
        source_part = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem).strip("_") or "doc"
        page_part = f"p{page}" if page is not None else "p0"
        segment_index = 0
        for paragraph_index, paragraph in enumerate(split_paragraphs(str(chunk.get("text", "")))):
            sentences = split_sentences(paragraph, min_chars=min_sentence_chars)
            if not sentences:
                continue
            scores = scorer(sentences) if len(sentences) > 1 else []
            segments = split_by_scores(
                sentences,
                scores,
                threshold=threshold,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            for paragraph_segment_index, segment in enumerate(segments):
                unit_id = f"semantic_chunk::{source_part}_{page_part}_sc{segment_index}"
                unit = {
                    "id": unit_id,
                    "granularity": "semantic_chunk",
                    "text": segment.text,
                    "source": source,
                    "page": page,
                    "parent_chunk_id": f"{source_part}_{page_part}_semantic_{segment_index}",
                    "paper_title": paper_title_from_source(source),
                    "paragraph_index": paragraph_index,
                    "paragraph_segment_index": paragraph_segment_index,
                    "sentence_count": segment.sentence_count,
                }
                if segment.score_min is not None and np.isfinite(segment.score_min):
                    unit["segment_score_min"] = float(segment.score_min)
                units.append(unit)
                segment_index += 1
    return units
