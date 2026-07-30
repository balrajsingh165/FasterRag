"""Bridge from an async embedding adapter to the synchronous chunker interface.

Semantic chunking needs sentence vectors, but chunking runs in the CPU worker pool where
there is no event loop to await on — so the chunkers take a synchronous
:class:`~fasterrag.core.chunking.SentenceEmbedder`. This bridge adapts either side.

A local model is called directly, avoiding an event loop entirely. A remote provider is
driven through a private loop, which blocks the calling worker process. That is the
correct trade: a worker process is exactly what should block, and the API's event loop is
never involved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder
from fasterrag.errors import EmbedError

__all__ = ["SentenceEmbedderBridge"]


class SentenceEmbedderBridge:
    """Exposes an embedding adapter through the synchronous chunker interface."""

    def __init__(self, adapter: EmbeddingAdapter) -> None:
        """Wrap ``adapter``."""
        self._adapter = adapter

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per text.

        Raises:
            EmbedError: If called from inside a running event loop with a remote provider,
                which would deadlock. Chunking belongs in the CPU worker pool.
        """
        if isinstance(self._adapter, HuggingFaceEmbedder):
            return self._adapter.encode(texts)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._adapter.embed_documents(texts)).vectors

        raise EmbedError(
            "semantic chunking with a remote embedding provider cannot run inside an "
            "event loop; run chunking in the CPU worker pool",
            retryable=False,
        )
