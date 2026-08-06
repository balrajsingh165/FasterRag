"""Layout-aware chunker.

Uses the structure the parser recovered: chunks start at headings, and a table is kept
whole whenever it fits. A chunk that begins at a heading carries its own context, and a
table split down the middle loses the header row that gave its cells meaning — those two
properties are what layout awareness buys.

Blocks too large to fit are split recursively, so a giant table degrades to row groups
rather than to arbitrary character windows.
"""

from __future__ import annotations

from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    Segment,
    TextChunk,
    TokenCounter,
    assemble,
    within_budget,
)
from fasterrag.core.chunking.recursive import split_span
from fasterrag.core.parsing.models import ParsedDocument

__all__ = ["LayoutChunker"]


class LayoutChunker:
    """Packs whole structural blocks into chunks, breaking at headings."""

    strategy = "layout"

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

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a parsed document along its structure."""
        text = document.text
        if not text.strip():
            return []
        if not document.blocks:
            return assemble(
                text,
                within_budget(
                    text,
                    split_span(text, 0, len(text), self._limit),
                    counter=self._counter,
                    budget=self._budget,
                    limit=self._limit,
                    split=split_span,
                ),
                overlap_chars=self._overlap,
                overlap_tokens=self._overlap_tokens,
                strategy=self.strategy,
                counter=self._counter,
            )

        segments = within_budget(
            text,
            self._pack_blocks(document),
            counter=self._counter,
            budget=self._budget,
            limit=self._limit,
            split=split_span,
        )
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

    def _pack_blocks(self, document: ParsedDocument) -> list[Segment]:
        """Group block-aligned tiles into chunks, starting a chunk at every heading."""
        text = document.text
        blocks = document.blocks
        tiles: list[tuple[int, int, bool]] = []

        for index, block in enumerate(blocks):
            tile_end = blocks[index + 1].start if index + 1 < len(blocks) else len(text)
            tiles.append((block.start, tile_end, block.kind == "heading"))

        segments: list[Segment] = []
        for start, end, starts_section in tiles:
            oversized = end - start > self._limit

            if oversized:
                segments.extend(split_span(text, start, end, self._limit))
                continue

            if segments and not starts_section and end - segments[-1][0] <= self._limit:
                segments[-1] = (segments[-1][0], end)
            else:
                segments.append((start, end))

        return segments
