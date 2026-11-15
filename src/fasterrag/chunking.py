"""Public chunking surface: ``from fasterrag.chunking import RecursiveChunker``.

The documented standalone components (``docs/python-api.md``). Every chunker is usable on
its own over a :class:`~fasterrag.parsing.ParsedDocument`, so an application can adopt
fasterRag's chunking without adopting its pipeline. A chunker plugin implements
:class:`Chunker`, declares ``adapter_api_version`` against :data:`CHUNKER_API_VERSION` and
registers under :data:`CHUNKERS_GROUP` (``docs/plugins.md``).
"""

from fasterrag.core.chunking import (
    CHUNKER_API_VERSION,
    CHUNKERS_GROUP,
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
    "CHUNKERS_GROUP",
    "CHUNKER_API_VERSION",
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
