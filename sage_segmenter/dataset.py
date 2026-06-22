"""Training pair builders for SAGE-style semantic segmentation."""

from __future__ import annotations

from densex.corpus import paper_title_from_source
from sage_segmenter.sentence_utils import split_paragraphs, split_sentences


def build_sentence_pair_rows(
    chunks: list[dict],
    *,
    min_sentence_chars: int = 30,
    max_pairs: int = 0,
) -> list[dict]:
    """Create same-paragraph positive and cross-paragraph negative sentence pairs."""
    rows = []
    pair_index = 0
    for chunk in chunks:
        source = str(chunk.get("source", ""))
        paragraph_sentences = [
            sentences
            for sentences in (
                split_sentences(paragraph, min_chars=min_sentence_chars)
                for paragraph in split_paragraphs(str(chunk.get("text", "")))
            )
            if sentences
        ]

        previous_last_sentence: str | None = None
        for paragraph_index, sentences in enumerate(paragraph_sentences):
            if previous_last_sentence is not None:
                rows.append(
                    _pair_row(
                        pair_index=pair_index,
                        chunk=chunk,
                        source=source,
                        s1=previous_last_sentence,
                        s2=sentences[0],
                        label=0,
                        pair_type="cross_paragraph",
                        paragraph_index=paragraph_index,
                    )
                )
                pair_index += 1
                if max_pairs and len(rows) >= max_pairs:
                    return rows

            for sentence_index in range(len(sentences) - 1):
                rows.append(
                    _pair_row(
                        pair_index=pair_index,
                        chunk=chunk,
                        source=source,
                        s1=sentences[sentence_index],
                        s2=sentences[sentence_index + 1],
                        label=1,
                        pair_type="same_paragraph",
                        paragraph_index=paragraph_index,
                    )
                )
                pair_index += 1
                if max_pairs and len(rows) >= max_pairs:
                    return rows

            previous_last_sentence = sentences[-1]
    return rows


def _pair_row(
    *,
    pair_index: int,
    chunk: dict,
    source: str,
    s1: str,
    s2: str,
    label: int,
    pair_type: str,
    paragraph_index: int,
) -> dict:
    return {
        "id": f"pair_{pair_index:06d}",
        "source": source,
        "page": chunk.get("page"),
        "parent_chunk_id": chunk.get("id"),
        "paper_title": paper_title_from_source(source),
        "paragraph_index": paragraph_index,
        "pair_type": pair_type,
        "s1": s1,
        "s2": s2,
        "label": label,
    }

