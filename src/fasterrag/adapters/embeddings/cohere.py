"""Cohere embedding provider.

Cohere requires an ``input_type`` naming what is being embedded, and it genuinely changes
the vectors: passages must be embedded as ``search_document`` and queries as
``search_query``. Embedding both the same way is a silent retrieval-quality loss, which is
why the contract separates the two calls.

Cohere states what it billed on every embed response, and the adapter surfaces it as
``EmbeddingResult.total_tokens`` so spend is reconciled against the provider's own figure
rather than a local token estimate. Per the API reference for ``/v1/embed`` the response
carries an optional ``meta.billed_units`` whose ``input_tokens`` counts the billable text
units; the adapter reads that rather than ``meta.tokens``, because the invoice is charged
on the billed units and the two are allowed to differ.
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
    sum_reported,
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

    async def _embed(
        self, texts: Sequence[str], input_type: str
    ) -> tuple[list[list[float]], int | None]:
        """Embed one batch, returning vectors and the tokens Cohere says it billed."""
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
        return [list(vector) for vector in vectors], _billed_tokens(response)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages as search documents."""
        vectors: list[list[float]] = []
        billed: list[int | None] = []
        for batch in self.batches(texts):
            batch_vectors, tokens = await self._embed(batch, _DOCUMENT_INPUT)
            vectors.extend(batch_vectors)
            billed.append(tokens)

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
            total_tokens=sum_reported(billed),
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query as a search query."""
        vectors, _ = await self._embed([text], _QUERY_INPUT)
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


def _billed_tokens(response: Any) -> int | None:
    """Return the input tokens Cohere billed, or None when it stated none.

    Every level of ``meta`` is optional in the reference, so a missing one means "no
    receipt" rather than "free".
    """
    meta = getattr(response, "meta", None)
    billed = getattr(meta, "billed_units", None) if meta is not None else None
    tokens = getattr(billed, "input_tokens", None) if billed is not None else None
    return None if tokens is None else int(tokens)
