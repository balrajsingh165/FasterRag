"""Chunking pipeline: the configurable strategies and the factory that selects one.

``chunking.strategy`` picks the strategy and ``chunking.contextual_enrichment`` composes
with any of them (``docs/architecture.md`` §5). Chunk quality is the single largest lever
on retrieval quality, which is why five strategies exist and why all five are held to the
same tested invariants (``docs/testing-strategy.md`` §1.2).
"""

from __future__ import annotations

from typing import Protocol

from fasterrag.config.schema import Settings
from fasterrag.core.chunking.fixed import FixedChunker
from fasterrag.core.chunking.late import LateChunker
from fasterrag.core.chunking.layout import LayoutChunker
from fasterrag.core.chunking.models import (
    EstimatingTokenCounter,
    TextChunk,
    TokenCounter,
)
from fasterrag.core.chunking.recursive import RecursiveChunker
from fasterrag.core.chunking.semantic import SemanticChunker, SentenceEmbedder
from fasterrag.core.parsing.models import ParsedDocument
from fasterrag.errors import ConfigError

__all__ = [
    "Chunker",
    "EstimatingTokenCounter",
    "FixedChunker",
    "LateChunker",
    "LayoutChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceEmbedder",
    "TextChunk",
    "TokenCounter",
    "create_chunker",
]


class Chunker(Protocol):
    """Splits a parsed document into chunks."""

    strategy: str

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Return the document's chunks, in order."""
        ...


def create_chunker(
    settings: Settings,
    *,
    counter: TokenCounter | None = None,
    embedder: SentenceEmbedder | None = None,
) -> Chunker:
    """Build the chunker named by ``chunking.strategy``.

    Args:
        settings: Validated configuration.
        counter: Token counter to use; defaults to the estimating counter. The embedding
            provider's real tokenizer is passed here once one is configured.
        embedder: Sentence embedder, required only by the semantic strategy.

    Returns:
        The configured chunker.

    Raises:
        ConfigError: If the semantic strategy is selected without an embedder.
    """
    size = settings.chunking.chunk_size
    overlap = settings.chunking.overlap
    strategy = settings.chunking.strategy

    if strategy == "fixed":
        return FixedChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "layout":
        return LayoutChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "late":
        return LateChunker(chunk_size=size, overlap=overlap, counter=counter)

    if embedder is None:
        raise ConfigError(
            "chunking.strategy is 'semantic', which needs an embedding model to find "
            "sentence boundaries; configure an embedding provider or choose another strategy"
        )
    return SemanticChunker(embedder, chunk_size=size, overlap=overlap, counter=counter)
