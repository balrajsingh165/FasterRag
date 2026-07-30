"""Ollama embedding provider — a local server, so no credential is required.

Ollama embeds one text per request, so the adapter issues a request per text rather than
pretending to batch. That is a property of the server, not a shortcut: reporting it
honestly is what lets the cost estimator and throughput numbers stay truthful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingResult,
    classify_provider_failure,
    require_extra,
)
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError

__all__ = ["OllamaEmbedder"]


class OllamaEmbedder(EmbeddingAdapter):
    """Embeds through a local Ollama server."""

    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        """Build the adapter without opening a connection."""
        super().__init__(settings)
        self._client: Any | None = None
        self._timeout = settings.reliability.timeouts.embeddings_ms / 1000
        self._host = settings.llm.base_url

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return the model version; the pulled model tag is the most specific available."""
        return self.config.model

    @property
    def dimensions(self) -> int | None:
        """Return the configured dimensionality override, if any."""
        return self.config.dimensions

    def _connected(self) -> Any:
        """Return the client, building it on first use."""
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise require_extra(self.provider, "ollama", "ollama") from exc

            self._client = AsyncClient(host=self._host, timeout=self._timeout)
        return self._client

    async def _embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        try:
            response = await self._connected().embeddings(model=self.config.model, prompt=text)
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
                key_env=None,
            ) from exc

        vector = response.get("embedding") if isinstance(response, dict) else None
        if not vector:
            raise EmbedError("ollama returned an empty embedding", retryable=False)
        return [float(value) for value in vector]

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages, one request per text."""
        vectors = [await self._embed_one(text) for text in texts]

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return await self._embed_one(text)

    async def health(self) -> HealthStatus:
        """Report reachability with a minimal, non-destructive call."""
        try:
            await self._embed_one("health check")
        except (ConfigError, EmbedError) as exc:
            return HealthStatus(healthy=False, detail=exc.detail)
        return HealthStatus(healthy=True, detail=f"{self.provider} reachable")

    async def close(self) -> None:
        """Release the client."""
        self._client = None
