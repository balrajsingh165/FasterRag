"""Semantic chunker.

Splits where meaning shifts rather than where the character count runs out: adjacent
sentences are embedded, and a boundary is placed where the distance between neighbours
spikes. The spike threshold is a percentile of the distances observed *in this document*,
not a fixed number, so the strategy adapts to prose that is uniformly similar as well as
to documents that jump between topics. No configuration key is needed for it.

Embedding is supplied by the caller through :class:`SentenceEmbedder`. Chunking runs in
the CPU worker pool rather than on the event loop, so the interface is deliberately
synchronous.

The embedder contract is one vector per text, all of the same width. A provider that
breaks either half of it — a narrower vector for one sentence, or a reply that does not
answer every sentence — is rejected with a typed ``EmbedError`` rather than measured over
the part that lines up: a prefix distance is a plausible number and a short reply merges
every unanswered sentence into the last chunk, so the document would be chunked at the
wrong places and re-ided with no error anywhere. The worker pool dead-letters the document
with ``EMBED_PROVIDER_ERROR``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Final, Protocol

from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    Segment,
    TextChunk,
    TokenCounter,
    assemble,
    hard_split,
)
from fasterrag.core.chunking.recursive import pack
from fasterrag.core.parsing.models import ParsedDocument
from fasterrag.errors import EmbedError

__all__ = ["BREAKPOINT_PERCENTILE", "SemanticChunker", "SentenceEmbedder"]

BREAKPOINT_PERCENTILE: Final = 0.95
_MIN_SENTENCES: Final = 3

_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class SentenceEmbedder(Protocol):
    """Embeds sentences so boundaries can be found by similarity."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text."""
        ...


def _sentences(text: str) -> list[Segment]:
    """Return sentence-aligned tiles covering ``text`` exactly.

    The boundary needs no ``> cursor`` guard: both alternatives of ``_SENTENCE_END``
    consume at least one character, so the non-overlapping match ends strictly increase.
    """
    tiles: list[Segment] = []
    cursor = 0

    for match in _SENTENCE_END.finditer(text):
        tiles.append((cursor, match.end()))
        cursor = match.end()

    if cursor < len(text):
        tiles.append((cursor, len(text)))
    return tiles


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine distance between two vectors.

    Vectors of unequal width raise rather than yielding a prefix distance, which is a
    plausible number that would move every boundary placed after it.
    """
    try:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
    except ValueError as exc:
        raise EmbedError(
            f"the embedding provider returned a {len(left)}-dimension vector beside a "
            f"{len(right)}-dimension one for sentences of the same document; semantic "
            "boundaries cannot be placed from vectors of different widths",
            retryable=False,
        ) from exc
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def _threshold(distances: Sequence[float], percentile: float = BREAKPOINT_PERCENTILE) -> float:
    """Return the percentile of distances above which a boundary is placed.

    Lowering the percentile places more boundaries, giving smaller and more topically
    uniform chunks; raising it keeps related passages together.
    """
    if not distances:
        return 1.0
    ordered = sorted(distances)
    index = min(int(len(ordered) * percentile), len(ordered) - 1)
    return ordered[index]


class SemanticChunker:
    """Splits text where adjacent sentences stop resembling each other."""

    strategy = "semantic"

    def __init__(
        self,
        embedder: SentenceEmbedder,
        *,
        chunk_size: int = 768,
        overlap: int = 64,
        counter: TokenCounter | None = None,
        percentile: float = BREAKPOINT_PERCENTILE,
    ) -> None:
        """Build the chunker."""
        self._percentile = percentile
        self._embedder = embedder
        self._counter = counter or EstimatingTokenCounter()
        self._limit = chunk_size * self._counter.chars_per_token
        self._overlap_tokens = overlap
        self._overlap = overlap * self._counter.chars_per_token

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a parsed document at semantic boundaries."""
        text = document.text
        if not text.strip():
            return []

        tiles = _sentences(text)
        if len(tiles) < _MIN_SENTENCES:
            grouped = pack(tiles or [(0, len(text))], self._limit)
        else:
            grouped = self._group(text, tiles)

        segments: list[Segment] = []
        for start, end in grouped:
            if end - start > self._limit:
                segments.extend(hard_split(text[start:end], start, self._limit))
            else:
                segments.append((start, end))

        return assemble(
            text,
            segments,
            overlap_chars=self._overlap,
            overlap_tokens=self._overlap_tokens,
            strategy=self.strategy,
            counter=self._counter,
            page_at=document.page_at,
            section_at=document.section_at,
        )

    def _group(self, text: str, tiles: Sequence[Segment]) -> list[Segment]:
        """Merge sentence tiles into chunks, breaking where similarity drops.

        A reply of the wrong length is refused: one vector per sentence is the contract,
        and a short one would merge every unanswered sentence into the last chunk.
        """
        vectors = self._embedder.embed([text[start:end].strip() for start, end in tiles])
        if len(vectors) != len(tiles):
            raise EmbedError(
                f"the embedding provider returned {len(vectors)} vectors for "
                f"{len(tiles)} sentences of the same document; semantic boundaries cannot "
                "be placed from a reply that does not answer every sentence",
                retryable=False,
            )
        distances = [
            _distance(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)
        ]
        cutoff = _threshold(distances, self._percentile)

        segments: list[Segment] = [tiles[0]]
        for index in range(1, len(tiles)):
            start, end = tiles[index]
            shifted = distances[index - 1] >= cutoff
            if shifted or end - segments[-1][0] > self._limit:
                segments.append((start, end))
            else:
                segments[-1] = (segments[-1][0], end)

        return segments
