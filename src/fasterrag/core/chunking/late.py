"""Late chunker.

Late chunking splits the work in two: boundaries are chosen normally, but the *embedding*
is derived from a single long-context pass over the whole document, pooling the token
representations that fall inside each chunk. A chunk's vector therefore carries context
from beyond its own boundaries.

Only the boundary half belongs here. The pooling half belongs to the embedding stage,
because it needs token-level output from the model — something an embedding API generally
does not expose. This chunker produces recursive boundaries and marks each chunk so the
embedding pool knows to pool rather than to embed the chunk text on its own.
"""

from __future__ import annotations

from dataclasses import replace

from fasterrag.core.chunking.models import TextChunk, TokenCounter
from fasterrag.core.chunking.recursive import RecursiveChunker
from fasterrag.core.parsing.models import ParsedDocument

__all__ = ["LATE_POOLING_KEY", "LateChunker"]

LATE_POOLING_KEY = "late_pooling"


class LateChunker:
    """Chooses boundaries for late chunking and marks chunks for pooled embedding."""

    strategy = "late"

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
        self._inner = RecursiveChunker(chunk_size=chunk_size, overlap=overlap, counter=counter)

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a document and mark every chunk for pooled embedding.

        The marked chunks carry their exact spans, which is all the embedding stage needs
        to pool the right token range out of the long-context pass.
        """
        # TODO: TASK-0113 implements the pooling half in the embedding pool; until then the
        # embedding stage embeds the chunk text directly and the marker is inert.
        chunks = self._inner.split(document)
        return [
            replace(
                chunk,
                strategy=self.strategy,
                metadata={**chunk.metadata, LATE_POOLING_KEY: True},
            )
            for chunk in chunks
        ]
