"""Cohere embedding provider.

Cohere requires an ``input_type`` naming what is being embedded, and it genuinely changes
the vectors: passages must be embedded as ``search_document`` and queries as
``search_query``. Embedding both the same way is a silent retrieval-quality loss, which is
why the contract separates the two calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from fasterrag.adapters.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingResult,
    classify_provider_failure,
    require_extra,
    require_key,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError

__all__ = ["CohereEmbedder"]

_DOCUMENT_INPUT: Final = "search_document"
_QUERY_INPUT: Final = "search_query"


class CohereEmbedder(EmbeddingAdapter):
    """Embeds through Cohere's embed API."""

    provider = "cohere"

    def __init__(self, settings: Settings) -> None:
        """Build the adapter without opening a connection."""
        super().__init__(settings)
        self._client: Any | None = None
        self._timeout = settings.reliability.timeouts.embeddings_ms / 1000

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return the model version; the pinned model id is the most specific available."""
        return self.config.model

    @property
    def dimensions(self) -> int | None:
        """Return the configured dimensionality override, if any."""
        return self.config.dimensions

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from cohere import AsyncClient
            except ImportError as exc:
                raise require_extra(self.provider, "cohere", "cohere") from exc

            self._client = AsyncClient(
                api_key=require_key(self.config.api_key_env, self.provider),
                timeout=self._timeout,
            )
        return self._client

    async def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        """Embed one batch with the input type the model expects."""
        try:
            response = await self._connected().embed(
                texts=list(texts),
                model=self.config.model,
                input_type=input_type,
                embedding_types=["float"],
            )
        except ConfigError:
            raise
        # CRITICAL: the catch stays broad because the SDK raises a deep hierarchy of
        # transport and API errors. Every one must become a typed EmbedError so the worker
        # pool can decide whether to retry instead of a vendor exception escaping.
        except Exception as exc:
            raise classify_provider_failure(
                exc,
                provider=self.provider,
                operation="embed",
                key_env=self.config.api_key_env,
            ) from exc

        vectors = getattr(response.embeddings, "float_", None) or getattr(
            response.embeddings, "float", None
        )
        if vectors is None:
            raise EmbedError("cohere returned no float embeddings", retryable=False)
        return [list(vector) for vector in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages as search documents."""
        vectors: list[list[float]] = []
        for batch in self.batches(texts):
            vectors.extend(await self._embed(batch, _DOCUMENT_INPUT))

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query as a search query."""
        vectors = await self._embed([text], _QUERY_INPUT)
        return vectors[0]

    async def health(self) -> HealthStatus:
        """Report reachability with a minimal, non-destructive call."""
        try:
            await self._embed(["health check"], _QUERY_INPUT)
        except (ConfigError, EmbedError) as exc:
            return HealthStatus(healthy=False, detail=exc.detail)
        return HealthStatus(healthy=True, detail=f"{self.provider} reachable")

    async def close(self) -> None:
        """Close the client."""
        self._client = None
