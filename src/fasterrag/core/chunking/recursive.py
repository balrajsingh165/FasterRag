"""Recursive chunker — the default strategy.

Splits on progressively finer separators (blank lines, then lines, then sentences, then
words) and stops as soon as a piece fits, so it breaks at the largest natural boundary
available rather than at an arbitrary offset. Pieces are then packed greedily back up to
the target size so chunks are close to uniform instead of ragged.

Separators stay attached to the piece they follow, which is what keeps the segments a
gapless tiling of the source text and therefore keeps chunk offsets exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    Segment,
    TextChunk,
    TokenCounter,
    assemble,
    hard_split,
)
from fasterrag.core.parsing.models import ParsedDocument

__all__ = ["SEPARATORS", "RecursiveChunker", "pack", "split_span"]

SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", "; ", ", ", " ")

# The character limit below which the budget pass stops halving. Without a floor a text the
# counter never reports as fitting — a single glyph tokenizing to several tokens — would
# recurse until the segments were one character wide.
_MINIMUM_LIMIT: Final = 16


def _split_on(text: str, start: int, end: int, separator: str) -> list[Segment]:
    """Cut a span after each separator occurrence, keeping the separator with its piece."""
    segments: list[Segment] = []
    cursor = start

    while cursor < end:
        found = text.find(separator, cursor, end)
        if found == -1:
            segments.append((cursor, end))
            break
        boundary = found + len(separator)
        segments.append((cursor, boundary))
        cursor = boundary

    return segments


def split_span(
    text: str,
    start: int,
    end: int,
    limit: int,
    separators: Sequence[str] = SEPARATORS,
) -> list[Segment]:
    """Split a span into pieces no longer than ``limit`` characters."""
    if end - start <= limit:
        return [(start, end)]

    for index, separator in enumerate(separators):
        pieces = _split_on(text, start, end, separator)
        if len(pieces) > 1:
            finer = separators[index + 1 :]
            resolved: list[Segment] = []
            for piece_start, piece_end in pieces:
                resolved.extend(split_span(text, piece_start, piece_end, limit, finer))
            return resolved

    return hard_split(text[start:end], start, limit)


def pack(segments: Sequence[Segment], limit: int) -> list[Segment]:
    """Merge adjacent segments while they fit, preserving the tiling."""
    packed: list[Segment] = []

    for start, end in segments:
        if packed and end - packed[-1][0] <= limit:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))

    return packed


class RecursiveChunker:
    """Splits text at the largest natural boundary that fits."""

    strategy = "recursive"

    def __init__(
        self,
        *,
        chunk_size: int = 768,
        overlap: int = 64,
        counter: TokenCounter | None = None,
    ) -> None:
        """Build the chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            overlap: Tokens each chunk repeats from its predecessor.
            counter: Token counter; defaults to the estimating counter.
        """
        self._counter = counter or EstimatingTokenCounter()
        self._budget = chunk_size
        self._limit = chunk_size * self._counter.chars_per_token
        self._overlap_tokens = overlap
        self._overlap = overlap * self._counter.chars_per_token

    def _within_budget(self, text: str, segment: Segment, limit: int) -> list[Segment]:
        """Split a segment further until the counter agrees it fits ``chunk_size`` tokens.

        Splitting is done on characters, because a character tiling is what keeps chunk
        offsets exact and gapless. Characters per token is only an assumption though, and a
        wrong one for code, CJK text, and long identifiers — all of which tokenize far
        denser than prose. This pass re-splits the segments where the assumption was wrong,
        so the configured size means tokens rather than a guess about them.

        With the estimating counter the check can never fail (the limit is derived from the
        same ratio the count uses), so this costs one count per segment and changes nothing.
        """
        start, end = segment
        if limit <= _MINIMUM_LIMIT or self._counter.count(text[start:end]) <= self._budget:
            return [segment]

        halved = max(limit // 2, _MINIMUM_LIMIT)
        return [
            piece
            for finer in split_span(text, start, end, halved)
            for piece in self._within_budget(text, finer, halved)
        ]

    def segments(self, text: str) -> list[Segment]:
        """Return the tiling segments for ``text`` before overlap is applied."""
        packed = pack(split_span(text, 0, len(text), self._limit), self._limit)
        return [
            piece for segment in packed for piece in self._within_budget(text, segment, self._limit)
        ]

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a parsed document recursively."""
        if not document.text.strip():
            return []

        return assemble(
            document.text,
            self.segments(document.text),
            overlap_chars=self._overlap,
            overlap_tokens=self._overlap_tokens,
            strategy=self.strategy,
            counter=self._counter,
            page_at=document.page_at,
            section_at=document.section_at,
        )
