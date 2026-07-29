"""Fixed-size chunker.

Fixed token windows with overlap, snapping to the nearest word boundary so a window
never ends mid-word. It needs no model inference and no document structure, which makes
it the baseline every other strategy is compared against.
"""

from __future__ import annotations

from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    TextChunk,
    TokenCounter,
    assemble,
    hard_split,
)
from fasterrag.core.parsing.models import ParsedDocument

__all__ = ["FixedChunker"]


class FixedChunker:
    """Splits text into fixed-size token windows."""

    strategy = "fixed"

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
        self._limit = chunk_size * self._counter.chars_per_token
        self._overlap = overlap * self._counter.chars_per_token

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a parsed document into fixed windows."""
        if not document.text.strip():
            return []

        return assemble(
            document.text,
            hard_split(document.text, 0, self._limit),
            overlap_chars=self._overlap,
            strategy=self.strategy,
            counter=self._counter,
            page_at=document.page_at,
            section_at=document.section_at,
        )
