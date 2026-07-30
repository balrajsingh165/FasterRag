"""Retrieval result types.

The ``ScoredChunk`` of ``docs/data-model.md``: what retrieval returns internally and what
``--show-chunks`` exposes. It keeps **every leg's rank and score**, not just the fused
number, because the most common retrieval question is not "what came back" but "why" — and
that is unanswerable once the per-leg evidence has been collapsed into one score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DENSE_LEG", "SPARSE_LEG", "ScoredChunk"]

DENSE_LEG = "dense"
SPARSE_LEG = "bm25"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One retrieved chunk with the evidence behind its position."""

    chunk_id: str
    text: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int = 0

    @property
    def document_id(self) -> str | None:
        """Return the owning document's id."""
        value = self.payload.get("document_id")
        return value if isinstance(value, str) else None

    @property
    def source(self) -> str | None:
        """Return the source URI a citation points at."""
        value = self.payload.get("source_uri")
        return value if isinstance(value, str) else None

    @property
    def found_by_both_legs(self) -> bool:
        """Return whether dense and keyword retrieval agreed on this chunk."""
        return self.dense_rank is not None and self.bm25_rank is not None
