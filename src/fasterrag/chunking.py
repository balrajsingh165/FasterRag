"""Public chunking surface: ``from fasterrag.chunking import RecursiveChunker``.

The documented standalone components (``docs/python-api.md``). Every chunker is usable on
its own over a :class:`~fasterrag.parsing.ParsedDocument`, so an application can adopt
fasterRag's chunking without adopting its pipeline.
"""

from fasterrag.core.chunking import (
    Chunker,
    EstimatingTokenCounter,
    FixedChunker,
    LateChunker,
    LayoutChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceEmbedder,
    TextChunk,
    TokenCounter,
    create_chunker,
)

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
