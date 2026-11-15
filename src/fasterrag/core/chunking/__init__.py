"""Chunking pipeline: the configurable strategies and the factory that selects one.

``chunking.strategy`` picks the strategy and ``chunking.contextual_enrichment`` composes
with any of them (``docs/architecture.md`` §5). Chunk quality is the single largest lever
on retrieval quality, which is why five strategies exist and why all five are held to the
same tested invariants (``docs/testing-strategy.md`` §1.2).

A sixth strategy comes from a plugin (TASK-0513): a class registered under the
``fasterrag.chunkers`` entry-point group, selected by naming it in ``chunking.strategy``.
It is constructed as ``cls(settings, counter=counter)``, must define ``split`` and declare
``adapter_api_version`` compatible with :data:`CHUNKER_API_VERSION`. A built-in name always
wins, so a package cannot take over ``recursive``.
"""

from __future__ import annotations

from typing import Final, Protocol

from fasterrag.config.plugins import BUILTIN_CHUNKING_STRATEGIES, CHUNKERS_GROUP
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
from fasterrag.core.chunking.tokenizer import ModelTokenCounter, create_token_counter
from fasterrag.core.parsing.models import ParsedDocument
from fasterrag.core.plugin_api import load_plugin_class
from fasterrag.errors import ConfigError

__all__ = [
    "CHUNKERS_GROUP",
    "CHUNKER_API_VERSION",
    "Chunker",
    "EstimatingTokenCounter",
    "FixedChunker",
    "LateChunker",
    "LayoutChunker",
    "ModelTokenCounter",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceEmbedder",
    "TextChunk",
    "TokenCounter",
    "create_chunker",
    "create_token_counter",
    "resolve_chunker_class",
]

CHUNKER_API_VERSION: Final = "1.0"


class Chunker(Protocol):
    """Splits a parsed document into chunks."""

    strategy: str

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Return the document's chunks, in order."""
        ...


def resolve_chunker_class(strategy: str) -> type:
    """Return the plugin class registered as ``strategy`` under ``fasterrag.chunkers``."""
    return load_plugin_class(
        CHUNKERS_GROUP,
        strategy,
        host=CHUNKER_API_VERSION,
        interface="chunker",
        members=("split",),
    )


def create_chunker(
    settings: Settings,
    *,
    counter: TokenCounter | None = None,
    embedder: SentenceEmbedder | None = None,
) -> Chunker:
    """Build the chunker named by ``chunking.strategy``."""
    size = settings.chunking.chunk_size
    overlap = settings.chunking.overlap
    strategy = settings.chunking.strategy
    counter = counter or create_token_counter(settings)

    if strategy == "fixed":
        return FixedChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "layout":
        return LayoutChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy == "late":
        return LateChunker(chunk_size=size, overlap=overlap, counter=counter)
    if strategy not in BUILTIN_CHUNKING_STRATEGIES:
        plugin: Chunker = resolve_chunker_class(strategy)(settings, counter=counter)
        return plugin

    if embedder is None:
        raise ConfigError(
            "chunking.strategy is 'semantic', which needs an embedding model to find "
            "sentence boundaries; configure an embedding provider or choose another strategy"
        )
    return SemanticChunker(
        embedder,
        chunk_size=size,
        overlap=overlap,
        counter=counter,
        percentile=settings.chunking.semantic_percentile,
    )
